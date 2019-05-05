#include <pybind11/pybind11.h>
#include <torch/csrc/autograd/record_function.h>
#include <torch/csrc/jit/custom_operator.h>
#include <torch/csrc/jit/fuser/interface.h>
#include <torch/csrc/jit/operator_options.h>
#include <torch/csrc/jit/pass_manager.h>
#include <torch/csrc/jit/passes/graph_fuser.h>
#include <torch/csrc/jit/operator.h>

#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include "compiler.h"
#include "operators.h"

namespace py = pybind11;
using namespace torch::jit;

PYBIND11_MODULE(_torch_tvm, m) {
  auto tvm_sym = Symbol::fromQualString("tvm::CompilationGroup");

  // Register the tvm::CompilationGroup operator
  auto options = OperatorOptions().aliasAnalysis(AliasAnalysisKind::PURE);
  RegisterOperators op({Operator(
      tvm_sym,
      [](const Node* node) {
        auto cc = std::make_shared<TVMCompiler>(node);
        return [cc](Stack& stack) {
          RECORD_FUNCTION("TVM", std::vector<c10::IValue>());
          cc->run(stack);
          return 0;
        };
      },
      options)});

  // Register the pass that fuses parts of the graph into
  // a tvm::CompilationGroup
  RegisterPass pass([tvm_sym](std::shared_ptr<Graph>& g) {
    if (isEnabled()) {
      CustomFuseGraph(g, isSupported, tvm_sym);
    }
  });

  // python API to enable and disable tvm fusion
  m.def("enable", []() { setEnabled(true); });
  m.def("disable", []() { setEnabled(false); });

  m.doc() = "This module does nothing but register a TVM backend.";
}

TVM_REGISTER_GLOBAL("torch_tvm.register_operator")
    .set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* rv) {
      AT_CHECK(args.size() >= 2);
      std::string name = args[0];
      AT_ASSERT(args[1].type_code() == kFuncHandle);
      tvm::PackedFunc* f =
          new tvm::PackedFunc(args[1].operator tvm::PackedFunc());
      //auto& reg = tvm::relay::OpRegistry::Registry()
      //                ->__REGISTER_OR_GET__("torch." + name)
      //                .set_name();
      //reg.set_attr("FTVMCompute", *f, 10);
      //???
      //do this instead
      //https://github.com/dmlc/tvm/blob/cffb4fba03ea582417e2630bd163bca773756af6/src/relay/ir/op.cc#L133
      //???
      // tvm::Operation operation = args[1];
      //.set_attr<FTVMCompute>("FTVMCompute",
      // tvm::Expr expr = compute.body;
      // TODO use function_schema constructor and extract num return args from
      // *f()
      tvm::Tensor t = (*f)();
      tvm::Operation o = t->op;
      auto tvm_inputs = o->InputTensors();

      auto ident = [](const tvm::Array<tvm::relay::Type>& types,
          int num_inputs,
          const tvm::Attrs& attrs,
          const tvm::relay::TypeReporter& reporter) -> bool {
        for (size_t i = 1; i < types.size(); ++i) {
          reporter->Assign(types[i], types[0]);
        }
        return true;
      };
      auto& reg = tvm::relay::OpRegistry::Registry()
                      ->__REGISTER_OR_GET__("torch." + name)
                      .set_name();
      reg.add_type_rel("Identity", ident);
      reg.set_num_inputs(tvm_inputs.size());

      RegisterTVMOperator reg_op(
          {{Symbol::fromQualString("tvm::" + name),
            [name](Node* node, tvm::Array<tvm::relay::Expr> inputs) {
              auto op = tvm::relay::Op::Get("torch." + name);
              AT_ASSERT(op);
              auto call = tvm::relay::CallNode::make(op, inputs, tvm::Attrs(), {});
              auto pf = tvm::runtime::Registry::Get("relay._expr.AsText");
              std::string func_debug_str = (*pf)(call->op, false, nullptr);
              std::cerr << "expr is " << func_debug_str << "\n";
              //auto ftype = tvm::relay::GetType(call->op);
              //std::cerr << "GOT FTYPE " << ftype << "\n";
              return call;
            }}});

      auto sym = Symbol::fromQualString("tvm::" + name);
      auto options = OperatorOptions().aliasAnalysis(AliasAnalysisKind::PURE);

      std::vector<Argument> torch_inputs;
      for (const auto& tvm_input : tvm_inputs) {
        torch_inputs.emplace_back();
      }

      auto oo = Operator(
          FunctionSchema(
            "tvm::" + name,
            "",
            torch_inputs,
            {Argument()},
            false,
            false
            ),
          [](const Node* node) {
            return [](Stack& stack) {
              AT_CHECK(0, "TODO")
              return 0;
            };
          },
          options);
      RegisterOperators op({ oo });
    });
