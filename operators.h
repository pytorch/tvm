#pragma once
#include <tvm/tvm.h>
#include <torch/csrc/jit/ir.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>

bool isSupported(torch::jit::Node* node);
tvm::relay::Op getOperator(torch::jit::Node* node);

typedef tvm::relay::Op (*TVMOpFunctor)(torch::jit::Node*);
typedef const tvm::runtime::PackedFunc* (*TVMScheduleFunctor)();

struct RegisterTVMOperator {
  RegisterTVMOperator(std::vector<std::pair<torch::jit::Symbol, TVMOpFunctor>> ops);
};

struct RegisterTVMOperatorSchedule {
  RegisterTVMOperatorSchedule(std::vector<std::pair<std::string, TVMScheduleFunctor>> scheds);
};

