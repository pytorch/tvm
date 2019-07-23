#include <pybind11/pybind11.h>
#include <torch/csrc/autograd/record_function.h>
#include <torch/csrc/jit/custom_operator.h>
#include <torch/csrc/jit/operator_options.h>
#include <torch/csrc/jit/pass_manager.h>
#include <torch/csrc/jit/pybind_utils.h>

#include "compiler.h"
#include "fuse_linear.h"
#include "fusion_pass.h"

namespace py = pybind11;
using namespace torch::jit;

// control if we enable tvm fusion or not
static bool fusion_enabled = false;
// control if the run mode is strict or not, if it's strict, we throw to
// user with the relevant conversion errors, otherwise we bail out to JIT
static bool strict = false;

static int opt_level = 2;
static std::string device_type = "cpu";
static std::string device = "llvm -mcpu=core-avx2";
static std::string host = "llvm -mcpu=core-avx2";

static std::unordered_map<size_t, tvm::relay::Expr> relay_exprs;
static size_t relay_exprs_uuid = 0;

PYBIND11_MODULE(_torch_tvm, m) {
  // Register the tvm::CompilationGroup operator
  auto options = c10::OperatorOptions();
  options.setAliasAnalysis(AliasAnalysisKind::PURE);
  RegisterOperators op({Operator(
      getTVMSymbol(),
      [](const Node* node) {
        auto cc = std::make_shared<TVMCompiler>(
            node, opt_level, strict, device_type, device, host);
        return [cc](Stack& stack) {
          RECORD_FUNCTION("TVM", std::vector<c10::IValue>());
          cc->run(stack);
          return 0;
        };
      },
      options)});

  // Register the pass that fuses parts of the graph into
  // a tvm::CompilationGroup
  RegisterPass pass([](std::shared_ptr<Graph>& g) {
    if (fusion_enabled) {
      FuseLinear(g);
      FuseSupportedOps(g);
    }
  });

  // python API to enable and disable tvm fusion
  m.def(
      "enable",
      [](int opt_level_,
         bool strict_,
         std::string device_type_,
         std::string device_,
         std::string host_) {
        fusion_enabled = true;
        strict = strict_;
        opt_level = opt_level_;
        device_type = device_type_;
        device = device_;
        host = host_;
      },
      py::arg("opt_level") = 2,
      py::arg("strict") = false,
      py::arg("device_type") = "cpu",
      py::arg("device") = "llvm -mcpu=core-avx2",
      py::arg("host") = "llvm -mcpu=core-avx2");

  m.def("disable", []() { fusion_enabled = false; });

  m.def(
      "_push_relay_expr",
      [](std::shared_ptr<Graph> g, std::vector<at::Tensor> inputs) {
        size_t count = 0;
        for (auto node : g->nodes()) {
          count++;
        }
        TORCH_CHECK(
            count == 1,
            "This program cannot be exported as a single Relay expression.");
        for (auto node : g->nodes()) {
          if (node->kind() == getTVMSymbol()) {
            std::vector<Value*> v;
            auto subgraph = node->g(attr::Subgraph);
            TORCH_CHECK(
                subgraph->inputs().size() == inputs.size(),
                "Expected ",
                subgraph->inputs().size(),
                " inputs");
            for (auto i = 0; i < inputs.size(); ++i) {
              subgraph->inputs()[i]->inferTypeFrom(inputs[i]);
            }
            TVMContext ctx;
            ctx.device_type = kDLCPU;
            ctx.device_id = 0;
            auto expr = TVMCompiler::convertToRelay(subgraph, ctx);
            relay_exprs[++relay_exprs_uuid] = expr;
            return relay_exprs_uuid;
          } else {
            TORCH_CHECK(
                0,
                "This program contains non-Relay expressions that cannot be exported.");
          }
        }
        return 0UL;
      });

  m.doc() = "This module does nothing but register a TVM backend.";
}

TVM_REGISTER_GLOBAL("torch_tvm._pop_relay_expr")
    .set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* rv) {
      size_t id = args[0];
      *rv = relay_exprs[id]; //.top();
      relay_exprs.erase(id);
    });
