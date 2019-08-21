#pragma once
#include <torch/csrc/jit/ir.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>

#define PARAM_INDICES_convolution {1, 2}
#define PARAM_INDICES_layer_norm {2, 3}
#define PARAM_INDICES_linear {1, 2}

#define PARAM_INDICES(op_name) PARAM_INDICES_##op_name

bool isSupported(torch::jit::Node* node);
tvm::relay::Expr getOperator(
    torch::jit::Node* node,
    tvm::Array<tvm::relay::Expr> inputs);

bool relayIsNone(tvm::relay::Expr e);
uint64_t getNoneSentinel();

const std::vector<int32_t>& getParamIndices(torch::jit::Node* node);

using TVMOpFunctor = std::function<tvm::relay::Expr(
    torch::jit::Node* node,
    tvm::Array<tvm::relay::Expr> inputs)>;
using TVMScheduleFunctor = std::function<const tvm::runtime::PackedFunc*()>;

struct TVMOpMap {
  TVMOpMap(torch::jit::Symbol sym_, TVMOpFunctor fn_, std::string name_ = ""
      ,std::vector<int32_t> param_indices_={})
      : sym(sym_), fn(fn_), name(name_), param_indices(param_indices_){}

  torch::jit::Symbol sym;
  TVMOpFunctor fn;
  std::vector<int32_t> param_indices;
  std::string name;
};

struct RegisterTVMOperator {
  RegisterTVMOperator(std::vector<TVMOpMap> ops);
};

struct RegisterTVMOperatorSchedule {
  RegisterTVMOperatorSchedule(
      std::vector<std::pair<std::string, TVMScheduleFunctor>> scheds);
};
