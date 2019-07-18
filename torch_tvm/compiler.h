#pragma once

#include <torch/csrc/jit/argument_spec.h>
#include <torch/csrc/jit/ir.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/relay/analysis.h>
#include <tvm/build_module.h>
#include <tvm/tvm.h>
#include <vector>

struct TVMObject {
  tvm::PackedFunc kernel;
  tvm::PackedFunc set_input;
  tvm::PackedFunc get_output;
  // Map input indices to values in the subgraph
  std::vector<torch::jit::Value*> input_values;
  bool invalid = true;
};

struct TVMCompiler {
  TVMCompiler(
      const torch::jit::Node* node,
      int opt_level = 2,
      bool strict = false,
      std::string device_type = "cpu",
      std::string device = "llvm",
      std::string host = "llvm");
  void run(torch::jit::Stack& stack);

 private:
  std::shared_ptr<torch::jit::Graph> subgraph_;
  c10::optional<TVMObject> cache_;
  TVMContext ctx_;
  int opt_level_;
  bool strict_;
  std::string device_type_;
  std::string device_;
  std::string host_;
  tvm::runtime::Module build_mod_;

 public:
  static tvm::relay::Var convertToRelay(torch::jit::Value* val, TVMContext ctx);
  static tvm::relay::Expr convertToRelay(
      const torch::jit::IValue& val,
      TVMContext ctx);
  static tvm::relay::Function convertToRelay(
      std::shared_ptr<torch::jit::Graph> subgraph,
      TVMContext ctx,
      std::vector<torch::jit::Value*>* input_values = nullptr);
};
