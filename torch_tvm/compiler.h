#pragma once

#include <torch/csrc/jit/argument_spec.h>
#include <torch/csrc/jit/ir.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/relay/pass.h>
#include <tvm/tvm.h>
#include <vector>

struct TVMObject {
  tvm::PackedFunc kernel;
  tvm::PackedFunc set_input;
  tvm::PackedFunc get_output;
};

struct TVMCompiler {
  TVMCompiler(
      const torch::jit::Node* node,
      int opt_level = 2,
      std::string device_type = "cpu",
      std::string device = "llvm",
      std::string host = "llvm");
  void run(torch::jit::Stack& stack);

 private:
  std::shared_ptr<torch::jit::Graph> subgraph_;
  std::unordered_map<torch::jit::CompleteArgumentSpec, TVMObject> cache_;
  TVMContext ctx_;
  int opt_level_;
  std::string device_type_;
  std::string device_;
  std::string host_;

  tvm::relay::Var convertToRelay(torch::jit::Value* val);
  tvm::relay::Expr convertToRelay(const torch::jit::IValue& val);
  tvm::relay::Function convertToRelay(
      std::shared_ptr<torch::jit::Graph> subgraph, std::vector<torch::jit::Value*>* input_values);
};
