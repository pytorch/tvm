#pragma once

#include <torch/csrc/jit/argument_spec.h>
#include <torch/csrc/jit/ir.h>
#include <tvm/tvm.h>

struct TVMObject {
  tvm::PackedFunc kernel;
  tvm::PackedFunc set_input;
  tvm::PackedFunc get_output;
};

struct TVMCompiler {
  TVMCompiler(const torch::jit::Node* node);
  void run(torch::jit::Stack& stack);
  std::shared_ptr<torch::jit::Graph> subgraph_;
  std::unordered_map<torch::jit::CompleteArgumentSpec, TVMObject> cache_;
};
