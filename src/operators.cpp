#include "operators.h"

using namespace torch::jit;

std::unordered_map<std::string, TVMScheduleFunctor>& getTVMScheduleMap() {
  static std::unordered_map<std::string, TVMScheduleFunctor> map;
  return map;
}

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

// This must be done lazily to prevent SIOF
void registerSchedule(std::string name) {
  AT_ASSERT(getTVMScheduleMap().find(name) != getTVMScheduleMap().end());
  TVMScheduleFunctor sched_f = getTVMScheduleMap()[name];
  auto reg = tvm::runtime::Registry::Get("relay.op._Register");
  AT_ASSERT(reg);
  auto sched = sched_f();
  // Relay does not provide a good API for querying the status of schedules
  if (sched) {
    (*reg)(name, "FTVMSchedule", *sched, 10);
    getTVMScheduleMap()[name] = []() { return nullptr; };
  }
}

RegisterTVMOperatorSchedule::RegisterTVMOperatorSchedule(
    std::vector<std::pair<std::string, TVMScheduleFunctor>> scheds) {
  for (const auto& pair : scheds) {
    auto name = std::get<0>(pair);
    auto sched_f = std::get<1>(pair);
    getTVMScheduleMap()[name] = sched_f;
  }
}

RegisterTVMOperator reg({
    {Symbol::fromQualString("aten::add"),
     [](Node* node, tvm::Array<tvm::relay::Expr> inputs) {
       auto op = tvm::relay::Op::Get("add");
       registerSchedule("add");
       AT_ASSERT(inputs.size() == 3);
       tvm::Array<tvm::relay::Expr> add_inputs = {inputs[0], inputs[1]};
       // Handle pytorch's value argument in add
       auto value = inputs[2].as<tvm::relay::ConstantNode>();
       AT_ASSERT(
           value->is_scalar() &&
           reinterpret_cast<int*>(value->data->data)[0] == 1);
       auto out = tvm::relay::CallNode::make(op, add_inputs, tvm::Attrs(), {});
       return out;
     }},
    {Symbol::fromQualString("aten::mul"),
     [](Node* node, tvm::Array<tvm::relay::Expr> inputs) {
       auto op = tvm::relay::Op::Get("multiply");
       registerSchedule("multiply");
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
  auto can_handle = map.find(node->kind()) != map.end();
  if (node->kind() == prim::Constant) { can_handle = true; }
  return can_handle;
}

tvm::relay::Expr getOperator(Node* node, tvm::Array<tvm::relay::Expr> inputs) {
  AT_ASSERT(isSupported(node));
  return getTVMOperatorMap()[node->kind()](node, inputs);
}
