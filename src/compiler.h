#pragma once

#include <torch/csrc/jit/argument_spec.h>
#include <torch/csrc/jit/ir.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/relay/pass.h>
#include <tvm/tvm.h>

struct TVMObject {
  tvm::PackedFunc kernel;
  tvm::PackedFunc set_input;
  tvm::PackedFunc get_output;
};

struct TVMCompiler {
  TVMCompiler(const torch::jit::Node* node);
  void run(torch::jit::Stack& stack);

 private:
  std::shared_ptr<torch::jit::Graph> subgraph_;
  std::unordered_map<torch::jit::CompleteArgumentSpec, TVMObject> cache_;
  TVMContext ctx_;

  tvm::relay::Var convertToRelay(const torch::jit::Value* val);
  tvm::relay::Expr convertToRelay(const torch::jit::IValue& val);
  tvm::relay::Function convertToRelay(
      std::shared_ptr<torch::jit::Graph> subgraph);
};
