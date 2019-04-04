#include <tvm/tvm.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/relay/pass.h>
#include <ATen/DLConvertor.h>
#include <torch/csrc/jit/custom_operator.h>
#include <topi/generic/injective.h>
#include <torch/csrc/jit/pass_manager.h>
#include <torch/csrc/jit/passes/graph_fuser.h>
#include <torch/csrc/jit/fuser/interface.h>
#include <torch/csrc/jit/fuser/kernel_cache.h>
#include <torch/csrc/jit/argument_spec.h>
#include <pybind11/pybind11.h>

using namespace torch::jit;

namespace torch {
namespace jit {

struct TVMObject {
  tvm::PackedFunc kernel_;
  tvm::PackedFunc set_input_;
  tvm::PackedFunc get_output_;
};

struct TORCH_API TVMCompiler {
  TVMCompiler(const Node* node);
  void run(Stack& stack);
  std::shared_ptr<Graph> subgraph_;
  std::unordered_map<CompleteArgumentSpec, TVMObject> cache_;
};

bool isSupported(Node* node);
tvm::relay::Function convertToRelay(std::shared_ptr<Graph> subgraph);

typedef tvm::relay::Op (*TVMOpFunctor)(Node*);
typedef const tvm::runtime::PackedFunc* (*TVMScheduleFunctor)();

std::unordered_map<Symbol, TVMOpFunctor>& getTVMOperatorMap() {
  static std::unordered_map<Symbol, TVMOpFunctor> map;
  return map;
}

struct RegisterTVMOperator {
  RegisterTVMOperator(std::vector<std::pair<Symbol, TVMOpFunctor>> ops) {
    for (const auto& pair : ops) {
      auto sym = std::get<0>(pair);
      auto op = std::get<1>(pair);
      getTVMOperatorMap()[sym] = op;
    }
  }
};

struct RegisterTVMOperatorSchedule {
  RegisterTVMOperatorSchedule(std::vector<std::pair<std::string, TVMScheduleFunctor>> scheds) {
    for (const auto& pair : scheds) {
      std::string name;
      TVMScheduleFunctor sched_f;
      std::tie(name, sched_f) = pair;
      auto reg = tvm::runtime::Registry::Get("relay.op._Register");
      AT_ASSERT(reg);
      auto sched = sched_f();
      AT_ASSERT(sched);
      (*reg)(name, "FTVMSchedule", *sched, 10);
    }
  }
};

RegisterTVMOperator reg({{Symbol::fromQualString("aten::add"),
                          [](Node* n) { return tvm::relay::Op::Get("add"); }},
                         {Symbol::fromQualString("aten::mul"),
                          [](Node* n) { return tvm::relay::Op::Get("multiply"); }}});

RegisterTVMOperatorSchedule reg_sched({
  {"add",
  []() { return tvm::runtime::Registry::Get("topi.generic.schedule_injective"); } },
  {"multiply",
  []() { return tvm::runtime::Registry::Get("topi.generic.schedule_injective"); } }
});

bool isSupported(Node* node) {
  auto map = getTVMOperatorMap();
  return map.find(node->kind()) != map.end();
}

tvm::relay::Op getOperator(Node* node) {
  AT_ASSERT(isSupported(node));
  return getTVMOperatorMap()[node->kind()](node);
}

TVMCompiler::TVMCompiler(const Node* node) {
  subgraph_ = node->g(attr::Subgraph);
}

void TVMCompiler::run(Stack& stack) {
  std::vector<IValue> inputs;
  for (const auto& input : subgraph_->inputs()) {
    inputs.emplace_back(stack.back());
    stack.pop_back();
  }

  CompleteArgumentSpec spec{false, ArrayRef<IValue>(inputs)};

  if (cache_.find(spec) == cache_.end()) {
    for (auto i = 0; i < inputs.size(); ++i) {
      auto ivalue_input = inputs[i];
      auto value_input = subgraph_->inputs()[i];
      AT_ASSERT(ivalue_input.isTensor());
      value_input->inferTypeFrom(ivalue_input.toTensor());
    }
    auto func = convertToRelay(subgraph_);
    auto pfb = tvm::runtime::Registry::Get("relay.build_module._BuildModule");
    AT_ASSERT(pfb);
    tvm::runtime::Module build_mod = (*pfb)();
    auto build_f = build_mod.GetFunction("build", false);
    auto json_f = build_mod.GetFunction("get_graph_json", false);
    auto mod_f = build_mod.GetFunction("get_module", false);
    build_f(func, "llvm", "llvm");
    std::string json = json_f();
    tvm::runtime::Module mod = mod_f();
    // TODO support gpu
    TVMContext cpu_ctx;
    cpu_ctx.device_type = kDLCPU;
    cpu_ctx.device_id = 0;
    auto pfr = tvm::runtime::Registry::Get("tvm.graph_runtime.create");
    AT_ASSERT(pfr);
    tvm::runtime::Module run_mod =
        (*pfr)(json, mod, (int)cpu_ctx.device_type, (int)cpu_ctx.device_id);
    cache_[spec].set_input_ = run_mod.GetFunction("set_input", false);
    cache_[spec].kernel_ = run_mod.GetFunction("run", false);
    cache_[spec].get_output_ = run_mod.GetFunction("get_output", false);
  }

  int i = 0;
  for (const auto& input : inputs) {
    auto tensor = input.toTensor();
    auto dl_tensor = at::toDLPack(tensor);
    cache_[spec].set_input_(i++, tvm::runtime::NDArray::FromDLPack(dl_tensor));
  }
  cache_[spec].kernel_();
  i = 0;
  for (const auto& output : subgraph_->outputs()) {
    tvm::runtime::NDArray ret_val = cache_[spec].get_output_(i++);
    auto dl_tensor = ret_val.ToDLPack();
    auto d = (float*)dl_tensor->dl_tensor.data;
    auto tensor = at::fromDLPack(dl_tensor);
    auto var = torch::autograd::make_variable(tensor);
    stack.push_back(IValue(var));
  }
}

tvm::relay::Expr createValues(Node* node, tvm::Array<tvm::relay::Expr> inputs) {
  auto op = getOperator(node);
  auto out = tvm::relay::CallNode::make(op, inputs, tvm::Attrs(), {});
  return out;
}

tvm::relay::Function convertToRelay(std::shared_ptr<Graph> subgraph) {
  auto normalized = torch::jit::fuser::normalizeGraphForCache(subgraph);
  auto key = torch::jit::fuser::store(normalized);
  auto f = torch::jit::fuser::retrieve(key);
  std::unordered_map<Value*, tvm::relay::Expr> value_map;

  for (const auto& input : subgraph->inputs()) {
    AT_ASSERT(input->isCompleteTensor());
    auto pt_t = input->type()->cast<CompleteTensorType>();
    tvm::Array<HalideIR::Expr> sizes;
    for (const auto& size : pt_t->sizes()) {
      sizes.push_back(HalideIR::Expr(size));
    }
    // TODO: support non-float inputs
    auto t = tvm::relay::TensorTypeNode::make(sizes, ::tvm::Float(32));
    auto v = tvm::relay::VarNode::make(input->uniqueName(), t);
    value_map[input] = v;
  }

  auto frontier = subgraph->inputs().vec();
  // TODO error handle incorrectly formed graphs (not dominated by frontier)
  while (frontier.size()) {
    std::vector<Value*> new_frontier = {};
    for (const auto& value : frontier) {
      auto uses = value->uses();
      for (const auto& use : uses) {
        tvm::Array<tvm::relay::Expr> relay_inputs;
        auto skip_user = false;
        for (const auto& input : use.user->inputs()) {
          if (value_map.find(input) == value_map.end()) {
            skip_user = true;
            break;
          }
          relay_inputs.push_back(value_map[input]);
        }
        if (skip_user) { continue; }
        // Things like prim::Return
        if (use.user->outputs().size() < 1) {
          continue;
        }
        // TODO handle multiple outputs
        AT_ASSERT(use.user->outputs().size() == 1);
        value_map[use.user->output()] = createValues(use.user, relay_inputs);
        new_frontier.emplace_back(use.user->output());
      }
    }
    frontier = new_frontier;
  }

  AT_ASSERT(subgraph->outputs().size() == 1);
  auto output = subgraph->outputs().at(0);
  AT_ASSERT(value_map.find(output) != value_map.end());
  return tvm::relay::FunctionNode::make(
      tvm::relay::FreeVars(value_map[output]),
      value_map[output],
      tvm::relay::Type(),
      {});
}

static auto tvm_sym = Symbol::fromQualString("tvm::CompilationGroup");
static auto options = OperatorOptions().aliasAnalysis(AliasAnalysisKind::EXTRACTOR);
RegisterOperators reg_tvm({Operator(tvm_sym, [](const Node* node) {
  auto cc = std::make_shared<TVMCompiler>(node);
  return [cc](Stack& stack) {
    cc->run(stack);
    return 0;
  };
}, options)});

} // jit
} // torch

namespace py = pybind11;

void tvmPass(std::shared_ptr<Graph>&g) {
  torch::jit::overrideCanFuseOnCPU(true);
  CustomFuseGraph(g, torch::jit::isSupported, tvm_sym);
  torch::jit::overrideCanFuseOnCPU(false);
  return;
}

PYBIND11_MODULE(torch_tvm, m) {
  RegisterPass p(tvmPass);
  m.doc() = "This module does nothing but register a TVM backend.";
}

