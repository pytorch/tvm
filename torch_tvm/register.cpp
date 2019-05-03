#include <pybind11/pybind11.h>
#include <torch/csrc/autograd/record_function.h>
#include <torch/csrc/jit/custom_operator.h>
#include <torch/csrc/jit/fuser/interface.h>
#include <torch/csrc/jit/operator_options.h>
#include <torch/csrc/jit/pass_manager.h>
#include <torch/csrc/jit/passes/graph_fuser.h>

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
  m.def("enable", [](){
      setEnabled(true);
  });
  m.def("disable", [](){
      setEnabled(false);
  });


  m.doc() = "This module does nothing but register a TVM backend.";
}

TVM_REGISTER_GLOBAL("torch_tvm.test")
.set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* rv) {
    *rv = 1337;
    });
