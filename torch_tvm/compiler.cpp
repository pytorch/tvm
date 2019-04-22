#include "compiler.h"
#include "operators.h"

#include <ATen/DLConvertor.h>
#include <torch/csrc/jit/constants.h>
#include <torch/csrc/jit/fuser/kernel_cache.h>

using namespace torch::jit;

tvm::relay::Var TVMCompiler::convertToRelay(const Value* val) {
  if (val->isCompleteTensor()) {
    auto pt_t = val->type()->cast<CompleteTensorType>();
    tvm::Array<HalideIR::Expr> sizes;
    for (const auto& size : pt_t->sizes()) {
      sizes.push_back(HalideIR::Expr(size));
    }
    // TODO: support non-float tensors
    auto t = tvm::relay::TensorTypeNode::make(sizes, ::tvm::Float(32));
    auto v = tvm::relay::VarNode::make(val->uniqueName(), t);
    return v;
  }
  AT_ASSERT(0);
}

tvm::relay::Expr TVMCompiler::convertToRelay(const IValue& val) {
  if (val.isDouble()) {
    auto x = tvm::runtime::NDArray::Empty(
        {}, tvm::runtime::String2TVMType("float32"), ctx_);
    reinterpret_cast<float*>(x->data)[0] = val.toDouble();
    auto v = tvm::relay::ConstantNode::make(x);
    return v;
  }
  if (val.isInt()) {
    auto x = tvm::runtime::NDArray::Empty(
        {}, tvm::runtime::String2TVMType("int32"), ctx_);
    reinterpret_cast<int32_t*>(x->data)[0] = val.toInt();
    auto v = tvm::relay::ConstantNode::make(x);
    return v;
  }
  if (val.isBool()) {
    auto x = tvm::runtime::NDArray::Empty(
        {}, tvm::runtime::String2TVMType("bool"), ctx_);
    reinterpret_cast<int32_t*>(x->data)[0] = val.toBool();
    auto v = tvm::relay::ConstantNode::make(x);
    return v;
  }
  // TODO Add None type to Relay
  if (val.isNone()) {
    auto x = tvm::runtime::NDArray::Empty(
        {}, tvm::runtime::String2TVMType("int32"), ctx_);
    reinterpret_cast<int32_t*>(x->data)[0] = 0;
    auto v = tvm::relay::ConstantNode::make(x);
    return v;
  }
  if (val.isIntList()) {
    tvm::Array<tvm::relay::Expr> tuple_elems;
    for (const auto& elem : val.toIntList()->elements()) {
      auto x = tvm::runtime::NDArray::Empty(
          {}, tvm::runtime::String2TVMType("int32"), ctx_);
      reinterpret_cast<int32_t*>(x->data)[0] = elem;
      auto v = tvm::relay::ConstantNode::make(x);
      tuple_elems.push_back(v);
    }
    return tvm::relay::TupleNode::make(tuple_elems);
  }
  AT_CHECK(
      0, "Cannot convert value ", val, " to Relay yet.  Please file a bug.\n");
}

tvm::relay::Function TVMCompiler::convertToRelay(
    std::shared_ptr<Graph> subgraph) {
  std::unordered_map<Value*, tvm::relay::Expr> value_map;
  tvm::Array<tvm::relay::Var> input_vars;

  for (const auto& input : subgraph->inputs()) {
    AT_ASSERT(input->isCompleteTensor());
    auto v = convertToRelay(input);
    input_vars.push_back(v);
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
            // We may be dealing with a constant, handle that here
            auto optional_ivalue = toIValue(input);
            if (!optional_ivalue.has_value()) {
              skip_user = true;
              break;
            } else {
              value_map[input] = convertToRelay(optional_ivalue.value());
            }
          }
          relay_inputs.push_back(value_map[input]);
        }
        if (skip_user) {
          continue;
        }
        // Things like prim::Return
        if (use.user->outputs().size() < 1) {
          continue;
        }
        // if there are 2+ outputs, getOperator returns a tuple
        if (use.user->outputs().size() == 1) {
          value_map[use.user->output()] = getOperator(use.user, relay_inputs);
          new_frontier.emplace_back(use.user->output());
        } else {
          auto tuple = getOperator(use.user, relay_inputs);
          int index = 0;
          for (const auto& output : use.user->outputs()) {
            auto n = tvm::make_node<tvm::relay::TupleGetItemNode>();
            n->tuple = tuple;
            n->index = index;
            value_map[output] = tvm::relay::TupleGetItem(n);
            index++;
            new_frontier.emplace_back(output);
          }
        }
      }
    }
    frontier = new_frontier;
  }

  AT_ASSERT(subgraph->outputs().size() == 1);
  auto output = subgraph->outputs().at(0);
  AT_ASSERT(value_map.find(output) != value_map.end());
  tvm::Array<tvm::relay::Var> free_vars =
      tvm::relay::FreeVars(value_map[output]);
  AT_ASSERT(free_vars.size() == input_vars.size());

  return tvm::relay::FunctionNode::make(
      input_vars, value_map[output], tvm::relay::Type(), {});
}

TVMCompiler::TVMCompiler(const Node* node) {
  // TODO support gpu
  ctx_.device_type = kDLCPU;
  ctx_.device_id = 0;
  subgraph_ = node->g(attr::Subgraph);
}

void TVMCompiler::run(Stack& stack) {
  std::unordered_map<Value*, IValue*> value_to_ivalue;
  std::vector<IValue> inputs;

  // Reverse the stack
  for (const auto& input : subgraph_->inputs()) {
    inputs.emplace(inputs.begin(), stack.back());
    stack.pop_back();
  }

  for (auto i = 0; i < inputs.size(); ++i) {
    auto value_input = subgraph_->inputs()[i];
    value_to_ivalue[value_input] = &inputs[i];
  }

  CompleteArgumentSpec spec{false, ArrayRef<IValue>(inputs)};

  if (cache_.find(spec) == cache_.end()) {
    for (auto& kv : value_to_ivalue) {
      kv.first->inferTypeFrom(kv.second->toTensor());
    }
    auto func = convertToRelay(subgraph_);
    auto pfb = tvm::runtime::Registry::Get("relay.build_module._BuildModule");
    AT_ASSERT(pfb);
    tvm::runtime::Module build_mod = (*pfb)();
    auto build_f = build_mod.GetFunction("build", false);
    auto json_f = build_mod.GetFunction("get_graph_json", false);
    auto mod_f = build_mod.GetFunction("get_module", false);
    build_f(func, "llvm", "llvm -mcpu=core-avx2");
    std::string json = json_f();
    tvm::runtime::Module mod = mod_f();
    auto pfr = tvm::runtime::Registry::Get("tvm.graph_runtime.create");
    AT_ASSERT(pfr);
    tvm::runtime::Module run_mod =
        (*pfr)(json, mod, (int)ctx_.device_type, (int)ctx_.device_id);
    cache_[spec].set_input = run_mod.GetFunction("set_input", false);
    cache_[spec].kernel = run_mod.GetFunction("run", false);
    cache_[spec].get_output = run_mod.GetFunction("get_output", false);
  }

  for (auto i = 0; i < subgraph_->inputs().size(); ++i) {
    auto* ivalue = value_to_ivalue[subgraph_->inputs()[i]];
    auto tensor = ivalue->toTensor();
    auto dl_tensor = at::toDLPack(tensor);
    cache_[spec].set_input(i, tvm::runtime::NDArray::FromDLPack(dl_tensor));
  }

  cache_[spec].kernel();

  int i = 0;
  for (const auto& output : subgraph_->outputs()) {
    tvm::runtime::NDArray ret_val = cache_[spec].get_output(i++);
    auto dl_tensor = ret_val.ToDLPack();
    auto tensor = at::fromDLPack(dl_tensor);
    auto var = torch::autograd::make_variable(tensor);
    stack.push_back(IValue(var));
  }
}
