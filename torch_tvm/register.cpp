#include <pybind11/pybind11.h>
#include <torch/csrc/autograd/record_function.h>
#include <torch/csrc/jit/custom_operator.h>
#include <torch/csrc/jit/operator_options.h>
#include <torch/csrc/jit/pass_manager.h>
#include <torch/csrc/jit/passes/graph_fuser.h>

#include "compiler.h"
#include "operators.h"

namespace py = pybind11;
using namespace torch::jit;

static bool fusion_enabled = false;
static int opt_level = PT_TVM_DEFAULT_OPT_LEVEL;
static std::string device_type = PT_TVM_DEFAULT_DEVICE_TYPE;
static std::string device = PT_TVM_DEFAULT_DEVICE;
static std::string host = PT_TVM_DEFAULT_HOST;
PYBIND11_MODULE(_torch_tvm, m) {
  auto tvm_sym = Symbol::fromQualString("tvm::CompilationGroup");

  // Register the tvm::CompilationGroup operator
  auto options = OperatorOptions().aliasAnalysis(AliasAnalysisKind::PURE);
  RegisterOperators op({Operator(
      tvm_sym,
      [](const Node* node) {
        auto cc = std::make_shared<TVMCompiler>(
            node, opt_level, device_type, device, host);
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
    if (fusion_enabled) {
      CustomFuseGraph(g, isSupported, tvm_sym);
    }
  });

  // python API to enable and disable tvm fusion
  m.def(
      "enable",
      [](int opt_level_,
         std::string device_type_,
         std::string device_,
         std::string host_) {
        fusion_enabled = true;
        opt_level = opt_level_;
        device_type = device_type_;
        device = device_;
        host = host_;
      },
      py::arg("opt_level") = PT_TVM_DEFAULT_OPT_LEVEL,
      py::arg("device_type") = PT_TVM_DEFAULT_DEVICE_TYPE,
      py::arg("device") = PT_TVM_DEFAULT_DEVICE,
      py::arg("host") = PT_TVM_DEFAULT_HOST);

  m.def("disable", []() { fusion_enabled = false; });

  m.doc() = "This module does nothing but register a TVM backend.";
}
