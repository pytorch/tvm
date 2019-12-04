#include <functional>

#include <torch/csrc/autograd/record_function.h>
#include <torch/csrc/jit/custom_operator.h>
#include <torch/csrc/jit/operator_options.h>
#include <torch/csrc/jit/pass_manager.h>

#include "compiler.h"
#include "debug_utils.h"
#include "fuse_concat.h"
#include "fuse_linear.h"
#include "fusion_pass.h"
#include "remove_dropout.h"

namespace tvm {
namespace {
// control if the run mode is strict or not, if it's strict, we throw to
// user with the relevant conversion errors, otherwise we bail out to JIT
static bool strict = false;
static int opt_level = 2;
static bool debug = false;
static bool debug_runtime = false;
static std::string device_type = "cpu";
static std::string device = "llvm -mcpu=core-avx2";
static std::string host = "llvm -mcpu=core-avx2";
static int device_id = 0;
static bool is_training_mode = false;

void registerTVMOp() {
  auto options = c10::OperatorOptions();
  options.setAliasAnalysis(AliasAnalysisKind::PURE_FUNCTION);
  torch::jit::RegisterOperators op({torch::jit::Operator(
      getTVMSymbol(),
      [](const torch::jit::Node* node) -> torch::jit::Operation {
        auto cc = std::make_shared<TVMCompiler>(
            node, opt_level, strict, debug, debug_runtime, device_type, device,
            host, device_id);
        return [cc](Stack& stack) {
          RECORD_FUNCTION("TVM", std::vector<c10::IValue>());
          cc->run(stack);
          return 0;
        };
      },
      options)});
}

void set_build_config(
    int opt_level_,
    bool strict_,
    bool debug_,
    bool debug_runtime_,
    const std::string& device_type_,
    const std::string& device_,
    const std::string& host_,
    int device_id_,
    bool is_training_) {
  opt_level = opt_level_;
  strict = strict_;
  debug = debug_;
  debug_runtime = debug_runtime_;
  device_type = device_type_;
  device = device_;
  host = host_;
  device_id = device_id_;
  is_training_mode = is_training_;
}

bool is_training() {
  return is_training_mode;
}

void torch_tvm_enable(std::function<bool()> enableTVMCompile) {
  registerTVMOp();
  torch::jit::RegisterPass pass(
      [enableTVMCompile =
           std::move(enableTVMCompile)](std::shared_ptr<torch::jit::Graph>& g) {
        if (enableTVMCompile()) {
          FuseLinear(g);
          FuseConcat(g);
          RemoveDropout(g);
          FuseSupportedOps(g);
        }
  });
}

} // namespace
} // namespace tvm
