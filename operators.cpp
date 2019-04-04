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

RegisterTVMOperator reg({{Symbol::fromQualString("aten::add"),
                          [](Node* n) { return tvm::relay::Op::Get("add"); }},
                         {Symbol::fromQualString("aten::mul"), [](Node* n) {
                            return tvm::relay::Op::Get("multiply");
                          }}});

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

tvm::relay::Op getOperator(Node* node) {
  AT_ASSERT(isSupported(node));
  return getTVMOperatorMap()[node->kind()](node);
}
