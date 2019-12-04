#pragma once

#include <torch/csrc/jit/argument_spec.h>
#include <torch/csrc/jit/interpreter.h>
#include <torch/csrc/jit/ir.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/relay/analysis.h>
#include <tvm/build_module.h>
#include <tvm/operation.h>
#include <ATen/Tensor.h>

#include <vector>

#include "memory_utils.h"
#include "debug_utils.h"

struct TVMGraphInputInfo {
  TVMGraphInputInfo(bool is_param_, std::string tvm_var_name_) {
    is_param = is_param_;
    tvm_var_name = std::move(tvm_var_name_);
  }
  TVMGraphInputInfo(bool is_param_, std::string&& tvm_var_name_) {
    is_param = is_param_;
    tvm_var_name = tvm_var_name_;
  }
  std::string tvm_var_name;
  bool is_param;
  // DLManagedTensorPtr = unique_ptr<DLManagedTensor, DLManagedTensorDeleter>
  torch_tvm::utils::DLManagedTensorPtr tvm_tensor;
};

struct TVMObject {
  tvm::PackedFunc kernel;
  tvm::PackedFunc set_input;
  tvm::PackedFunc get_output;
  tvm::PackedFunc setup_external_storage;
  // Map input indices to values in the subgraph
  // Plus indicates if the corresponding value is immutable,
  // e.g., a parameter such as weight.
  std::unordered_map<torch::jit::Value*, TVMGraphInputInfo> input_values;
  void populateParamTVMTensors(
      const std::unordered_map<torch::jit::Value*,
      torch::jit::IValue>& value_to_ivalue);
  tvm::Map<std::string, tvm::relay::Constant> generateParamConstantMap();
};

struct TVMCompiler {
  explicit TVMCompiler(
      const torch::jit::Node* node,
      int opt_level = 2,
      bool strict = false,
      bool debug = false,
      bool debug_runtime = false,
      std::string device_type = "cpu",
      std::string device = "llvm",
      std::string host = "llvm",
      int device_id = 0);
  void run(torch::jit::Stack& stack);

 private:
  std::shared_ptr<torch::jit::Graph> subgraph_;
  std::unordered_map<torch::jit::CompleteArgumentSpec, TVMObject> cache_;
  TVMContext ctx_;
  int opt_level_;
  bool strict_;
  bool debug_;
  bool debug_runtime_;
  std::string device_type_;
  std::string device_;
  std::string host_;
  int device_id_;
  tvm::runtime::Module build_mod_;
  DebugLogger debug_logger_;
  // Not used in OSS
  at::Tensor activation_buffer_;
  std::vector<int64_t> activation_buffer_shape_;
  int64_t max_activation_buffer_size_{0};

  std::string handle_str_;
  std::unique_ptr<torch::jit::InterpreterState> fallback_interpreter_;
  // stores argument specs for which we couldn't compile and fall back to the
  // interpreter
  std::unordered_set<torch::jit::CompleteArgumentSpec> bad_specs_;

  std::string getTVMCompilerHandle(std::shared_ptr<torch::jit::Graph> subgraph);

 public:
  static tvm::relay::Var convertToRelay(torch::jit::Value* val, TVMContext ctx);
  static tvm::relay::Expr convertToRelay(
      const torch::jit::IValue& val,
      TVMContext ctx);
  static tvm::relay::Function convertToRelay(
      std::shared_ptr<torch::jit::Graph> subgraph,
      TVMContext ctx,
      std::unordered_map<torch::jit::Value*, TVMGraphInputInfo>*
      input_values = nullptr);

#ifdef TVM_USE_FB_GRAPH_RUNTIME
  void allocateMemoryAndSetParams(
      TVMObject& obj,
      const tvm::Map<std::string, tvm::relay::Constant>& params,
      const std::string& json_str);
#endif
};
