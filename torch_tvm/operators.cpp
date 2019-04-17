#include "operators.h"
#include <tvm/relay/attrs/nn.h>

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

template <typename T>
T relayToConstant(tvm::relay::Expr e) {
  auto c = e.as<tvm::relay::ConstantNode>();
  AT_ASSERT(c->is_scalar());
  return static_cast<T*>(c->data->data)[0];
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
       // registerSchedule("add");
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
    {Symbol::fromQualString("aten::conv2d"),
     [](Node* node, tvm::Array<tvm::relay::Expr> inputs) {
       static const tvm::relay::Op& op = tvm::relay::Op::Get("nn.conv2d");
       tvm::Array<tvm::relay::Expr> new_inputs = {
           inputs[0],
           inputs[1],
       };
       auto conv_attrs = tvm::make_node<tvm::relay::Conv2DAttrs>();
       conv_attrs->groups = 1;
       conv_attrs->data_layout = "NCHW";
       conv_attrs->kernel_layout = "OIHW";
       conv_attrs->kernel_size = {3, 3};
       conv_attrs->padding = {0, 0};
       conv_attrs->strides = {1, 1};
       conv_attrs->dilation = {1, 1};

       auto out = tvm::relay::CallNode::make(
           op, new_inputs, tvm::Attrs(conv_attrs), {});
       return out;
     }},
    {Symbol::fromQualString("aten::batch_norm"),
     [](Node* node, tvm::Array<tvm::relay::Expr> inputs) {
       auto op = tvm::relay::Op::Get("nn.batch_norm");

       auto& broadcast = tvm::relay::Op::Get("broadcast_to_like");
       tvm::Array<tvm::relay::Expr> bn_inputs = {
           inputs[0],
           tvm::relay::CallNode::make(
               broadcast, {inputs[1], inputs[3]}, tvm::Attrs(), {}),
           tvm::relay::CallNode::make(
               broadcast, {inputs[1], inputs[3]}, tvm::Attrs(), {}),
           inputs[3],
           inputs[4],
       };
       auto attrs = tvm::make_node<tvm::relay::BatchNormAttrs>();
       auto eps = relayToConstant<double>(inputs[7]);
       attrs->epsilon = eps;
       attrs->axis = 1;
       // TODO handle gamma and beta correctly
       attrs->center = false;
       attrs->scale = false;
       auto out =
           tvm::relay::CallNode::make(op, bn_inputs, tvm::Attrs(attrs), {});
       auto n = tvm::make_node<tvm::relay::TupleGetItemNode>();
       n->tuple = std::move(out);
       n->index = 0;
       return tvm::relay::TupleGetItem(n);
     }},
    {Symbol::fromQualString("aten::mul"),
     [](Node* node, tvm::Array<tvm::relay::Expr> inputs) {
       auto op = tvm::relay::Op::Get("multiply");
       // registerSchedule("multiply");
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


// flag to control whether to enable tvm fusion, default to false
static bool tvm_fusion_enabled = false;

void setEnabled(bool flag) {
  tvm_fusion_enabled = flag;
}

bool isSupported(Node* node) {
  if (!tvm_fusion_enabled) {
    return false;
  }
  auto map = getTVMOperatorMap();
  auto can_handle = map.find(node->kind()) != map.end();
  if (node->kind() == prim::Constant) { can_handle = true; }
  return can_handle;
}

tvm::relay::Expr getOperator(Node* node, tvm::Array<tvm::relay::Expr> inputs) {
  AT_ASSERT(isSupported(node));
  return getTVMOperatorMap()[node->kind()](node, inputs);
}
