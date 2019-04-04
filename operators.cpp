#include "operators.h"

using namespace torch::jit;

std::unordered_map<Symbol, TVMOpFunctor>& getTVMOperatorMap() {
  static std::unordered_map<Symbol, TVMOpFunctor> map;
  return map;
}

RegisterTVMOperator::RegisterTVMOperator(
    std::vector<std::pair<Symbol, TVMOpFunctor>> ops) {
  for (const auto& pair : ops) {
    auto sym = std::get<0>(pair);
    auto op = std::get<1>(pair);
    getTVMOperatorMap()[sym] = op;
  }
}
RegisterTVMOperatorSchedule::RegisterTVMOperatorSchedule(
    std::vector<std::pair<std::string, TVMScheduleFunctor>> scheds) {
  for (const auto& pair : scheds) {
    std::string name;
    TVMScheduleFunctor sched_f;
    std::tie(name, sched_f) = pair;
    auto reg = tvm::runtime::Registry::Get("relay.op._Register");
    AT_ASSERT(reg);
    auto sched = sched_f();
    AT_ASSERT(sched);
    (*reg)(name, "FTVMSchedule", *sched, 10);
  }
}

RegisterTVMOperator reg({
    {Symbol::fromQualString("aten::add"),
     [](Node* node, tvm::Array<tvm::relay::Expr> inputs) {
       auto op = tvm::relay::Op::Get("add");
       AT_ASSERT(inputs.size() == 2);
       auto out = tvm::relay::CallNode::make(op, inputs, tvm::Attrs(), {});
       return out;
     }},
    {Symbol::fromQualString("aten::mul"),
     [](Node* node, tvm::Array<tvm::relay::Expr> inputs) {
       auto op = tvm::relay::Op::Get("multiply");
       auto out = tvm::relay::CallNode::make(op, inputs, tvm::Attrs(), {});
       return out;
     }},
});

RegisterTVMOperatorSchedule reg_sched(
    {{"add",
      []() {
        return tvm::runtime::Registry::Get("topi.generic.schedule_injective");
      }},
     {"multiply", []() {
        return tvm::runtime::Registry::Get("topi.generic.schedule_injective");
      }}});

bool isSupported(Node* node) {
  auto map = getTVMOperatorMap();
  return map.find(node->kind()) != map.end();
}

tvm::relay::Expr getOperator(Node* node, tvm::Array<tvm::relay::Expr> inputs) {
  AT_ASSERT(isSupported(node));
  return getTVMOperatorMap()[node->kind()](node, inputs);
}
