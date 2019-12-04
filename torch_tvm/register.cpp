#include <pybind11/pybind11.h>
#include <torch/csrc/autograd/record_function.h>
#include <torch/csrc/jit/custom_operator.h>
#include <torch/csrc/jit/operator_options.h>
#include <torch/csrc/jit/pass_manager.h>
#include <torch/csrc/jit/pybind_utils.h>

#include "register.h"

namespace py = pybind11;

// control if we enable tvm fusion or not
static bool fusion_enabled = false;

static std::unordered_map<size_t, tvm::relay::Expr> relay_exprs;
static size_t relay_exprs_uuid = 0;

PYBIND11_MODULE(_torch_tvm, m) {
  std::function<bool()> is_enabled = []() { return fusion_enabled; };
  tvm::torch_tvm_enable(is_enabled);
  // python API to enable and disable tvm fusion
  m.def(
      "enable",
      [](int opt_level_,
         bool strict_,
         bool debug_,
         bool debug_runtime_,
         std::string device_type_,
         std::string device_,
         std::string host_,
         int device_id_,
         bool is_training_) {
        fusion_enabled = true;
        tvm::set_build_config(
            opt_level_, strict_, debug_, debug_runtime_, device_type_, device_,
            host_, device_id_, is_training_);
      },
      py::arg("opt_level") = 2,
      py::arg("strict") = false,
      py::arg("debug") = false,
      py::arg("debug_runtime") = false,
      py::arg("device_type") = "cpu",
      py::arg("device") = "llvm -mcpu=core-avx2",
      py::arg("host") = "llvm -mcpu=core-avx2",
      py::arg("device_id") = 0,
      py::arg("is_training") = false);

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
