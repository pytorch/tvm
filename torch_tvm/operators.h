#pragma once
#include <torch/csrc/jit/ir.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>

bool isSupported(torch::jit::Node* node);
tvm::relay::Expr getOperator(
    torch::jit::Node* node,
    tvm::Array<tvm::relay::Expr> inputs);

bool relayIsNone(tvm::relay::Expr e);
uint64_t getNoneSentinel();

using TVMOpFunctor = std::function<tvm::relay::Expr(
    torch::jit::Node* node,
    tvm::Array<tvm::relay::Expr> inputs)>;
using TVMScheduleFunctor = std::function<const tvm::runtime::PackedFunc*()>;

struct TVMOpMap {
  TVMOpMap(torch::jit::Symbol sym_, TVMOpFunctor fn_, std::string name_ = "")
      : sym(sym_), fn(fn_), name(name_) {}

  torch::jit::Symbol sym;
  TVMOpFunctor fn;
  std::string name;
};

struct RegisterTVMOperator {
  RegisterTVMOperator(std::vector<TVMOpMap> ops);
};

struct RegisterTVMOperatorSchedule {
  RegisterTVMOperatorSchedule(
      std::vector<std::pair<std::string, TVMScheduleFunctor>> scheds);
};
