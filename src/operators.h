#pragma once
#include <torch/csrc/jit/ir.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/tvm.h>

void setTVMFusion(bool flag);
bool isSupported(torch::jit::Node* node);
tvm::relay::Expr getOperator(
    torch::jit::Node* node,
    tvm::Array<tvm::relay::Expr> inputs);

using TVMOpFunctor = std::function<tvm::relay::Expr(
    torch::jit::Node* node,
    tvm::Array<tvm::relay::Expr> inputs)>;
using TVMScheduleFunctor = std::function<const tvm::runtime::PackedFunc*()>;

struct RegisterTVMOperator {
  RegisterTVMOperator(
      std::vector<std::pair<torch::jit::Symbol, TVMOpFunctor>> ops);
};

struct RegisterTVMOperatorSchedule {
  RegisterTVMOperatorSchedule(
      std::vector<std::pair<std::string, TVMScheduleFunctor>> scheds);
};
