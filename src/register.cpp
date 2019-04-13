#include <pybind11/pybind11.h>
#include <torch/csrc/jit/custom_operator.h>
#include <torch/csrc/jit/fuser/interface.h>
#include <torch/csrc/jit/operator_options.h>
#include <torch/csrc/jit/pass_manager.h>
#include <torch/csrc/jit/passes/graph_fuser.h>

#include "compiler.h"
#include "operators.h"

namespace py = pybind11;
using namespace torch::jit;

PYBIND11_MODULE(torch_tvm, m) {
  auto tvm_sym = Symbol::fromQualString("tvm::CompilationGroup");

  // Register the tvm::CompilationGroup operator
  auto options = OperatorOptions().aliasAnalysis(AliasAnalysisKind::PURE);
  RegisterOperators op({Operator(
      tvm_sym,
      [](const Node* node) {
        auto cc = std::make_shared<TVMCompiler>(node);
        return [cc](Stack& stack) {
          cc->run(stack);
          return 0;
        };
      },
      options)});

  // Register the pass that fuses parts of the graph into
  // a tvm::CompilationGroup
  RegisterPass pass([tvm_sym](std::shared_ptr<Graph>& g) {
    CustomFuseGraph(g, isSupported, tvm_sym);
  });

  // python API to enable and disable tvm fusion
  m.def("enable_tvm_fusion", [](){
      setTVMFusion(true);
  });
  m.def("disable_tvm_fusion", [](){
      setTVMFusion(false);
  });

  m.doc() = "This module does nothing but register a TVM backend.";
}
