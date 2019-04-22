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
bool isConstant(tvm::relay::Expr e) {
  auto c = e.as<tvm::relay::ConstantNode>();
  return !!c;
}

template <typename T>
T relayToConstant(tvm::relay::Expr e) {
  auto c = e.as<tvm::relay::ConstantNode>();
  AT_ASSERT(c);
  AT_ASSERT(c->is_scalar());
  return static_cast<T*>(c->data->data)[0];
}

bool relayIsNone(tvm::relay::Expr e) {
  if (!isConstant(e)) {
    return false;
  }
  auto c = e.as<tvm::relay::ConstantNode>();
  if (!c->is_scalar()) {
    return false;
  }
  auto val = static_cast<uint64_t*>(c->data->data)[0];
  return val == getNoneSentinel();
}

uint64_t getNoneSentinel() {
  return 0xe4fa3adecabcf036;
}

tvm::Array<tvm::relay::IndexExpr> relayToIntList(tvm::relay::Expr e) {
  auto t = e.as<tvm::relay::TupleNode>();
  tvm::Array<tvm::relay::IndexExpr> elems;
  for (auto c: t->fields) {
    int elem = relayToConstant<int>(c);
    elems.push_back(elem);
  }
  return elems;
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
    {Symbol::fromQualString("aten::_convolution"),
     [](Node* node, tvm::Array<tvm::relay::Expr> inputs) {
       bool is_transpose = relayToConstant<bool>(inputs[6]);
       // check the operator to emit base on is_transpose
       auto op = tvm::relay::Op::Get("nn.conv2d");
       if (is_transpose) {
          op = tvm::relay::Op::Get("nn.conv2d_transpose");
       }

       // input and filter
       tvm::Array<tvm::relay::Expr> new_inputs = {
           inputs[0],
           inputs[1],
       };

       auto conv_attrs = tvm::make_node<tvm::relay::Conv2DAttrs>();
       conv_attrs->groups = relayToConstant<int>(inputs[8]);
       conv_attrs->data_layout = "NCHW";
       conv_attrs->kernel_layout = "OIHW";
       // kernel size information should already embed in the weight shape
       // TODO: figure out kernel size usage in tvm
       conv_attrs->kernel_size = tvm::NullValue<tvm::Array<tvm::relay::IndexExpr>>();
       conv_attrs->strides = relayToIntList(inputs[3]);
       conv_attrs->padding = relayToIntList(inputs[4]);
       conv_attrs->dilation = relayToIntList(inputs[5]);

       auto out = tvm::relay::CallNode::make(
           op, new_inputs, tvm::Attrs(conv_attrs), {});

       // Check if bias node is a var or constant (denoting a None currently),
       // if bias is present, emit an additional bias_add node.
       // TODO: better check when relay has None type
       auto bias_is_none = inputs[2].as<tvm::relay::ConstantNode>();
       if (!bias_is_none) {
         auto bias_add_op = tvm::relay::Op::Get("nn.bias_add");
         auto bias_add_attrs = tvm::make_node<tvm::relay::BiasAddAttrs>();
         bias_add_attrs->axis = 1;
         return tvm::relay::CallNode::make(
             bias_add_op, {out, inputs[2]}, tvm::Attrs(bias_add_attrs), {});
       }
       return out;
     }},
    {Symbol::fromQualString("aten::batch_norm"),
     [](Node* node, tvm::Array<tvm::relay::Expr> inputs) -> tvm::relay::Expr {
       auto op = tvm::relay::Op::Get("nn.batch_norm");
       auto attrs = tvm::make_node<tvm::relay::BatchNormAttrs>();
       auto eps = relayToConstant<double>(inputs[7]);
       attrs->epsilon = eps;
       attrs->axis = 1;

       if (relayIsNone(inputs[1])) {
         attrs->scale = false;
       } else {
         attrs->scale = true;
       }
       if (relayIsNone(inputs[2])) {
         attrs->center = false;
       } else {
         attrs->center = true;
       }

       auto& broadcast = tvm::relay::Op::Get("broadcast_to_like");
       tvm::Array<tvm::relay::Expr> bn_inputs = {
           inputs[0],
           tvm::relay::CallNode::make(
               broadcast, {inputs[1], inputs[3]}, tvm::Attrs(), {}),
           tvm::relay::CallNode::make(
               broadcast, {inputs[2], inputs[3]}, tvm::Attrs(), {}),
           inputs[3],
           inputs[4],
       };

       auto out =
           tvm::relay::CallNode::make(op, bn_inputs, tvm::Attrs(attrs), {});
       if (node->outputs().size() > 1) {
         return out;
       }
       auto n = tvm::make_node<tvm::relay::TupleGetItemNode>();
       n->tuple = std::move(out);
       n->index = 0;
       return tvm::relay::TupleGetItem(n);
     }},
    {Symbol::fromQualString("aten::relu"),
     [](Node* node, tvm::Array<tvm::relay::Expr> inputs) {
       auto op = tvm::relay::Op::Get("nn.relu");
       auto out = tvm::relay::CallNode::make(op, inputs, tvm::Attrs(), {});
       return out;
     }},
    {Symbol::fromQualString("aten::mul"),
     [](Node* node, tvm::Array<tvm::relay::Expr> inputs) {
       auto op = tvm::relay::Op::Get("multiply");
       // registerSchedule("multiply");
       auto out = tvm::relay::CallNode::make(op, inputs, tvm::Attrs(), {});
       return out;
     }},
    {Symbol::fromQualString("aten::avg_pool2d"),
     [](Node* node, tvm::Array<tvm::relay::Expr> inputs) {
       auto op = tvm::relay::Op::Get("nn.avg_pool2d");
       auto pool_attrs = tvm::make_node<tvm::relay::AvgPool2DAttrs>();
       pool_attrs->pool_size = relayToIntList(inputs[1]);
       auto strides = relayToIntList(inputs[2]);
       if (strides.size() == 0) {
         // pytorch avg_pool2d semantic: strides default to pool size
         pool_attrs->strides = pool_attrs->pool_size;
       } else {
         pool_attrs->strides = strides;
       }
       pool_attrs->padding = relayToIntList(inputs[3]);
       pool_attrs->layout = "NCHW";
       pool_attrs->ceil_mode = relayToConstant<bool>(inputs[4]);
       pool_attrs->count_include_pad = relayToConstant<bool>(inputs[5]);

       auto out = tvm::relay::CallNode::make(op, {inputs[0]}, tvm::Attrs(pool_attrs), {});
       return out;
     }},
    {Symbol::fromQualString("aten::max_pool2d"),
     [](Node* node, tvm::Array<tvm::relay::Expr> inputs) {
       auto pool_attrs = tvm::make_node<tvm::relay::MaxPool2DAttrs>();
       pool_attrs->pool_size = relayToIntList(inputs[1]);
       auto strides = relayToIntList(inputs[2]);
       if (strides.size() == 0) {
         // pytorch max_pool2d semantic: strides default to pool size
         pool_attrs->strides = pool_attrs->pool_size;
       } else {
         pool_attrs->strides = strides;
       }
       pool_attrs->padding = relayToIntList(inputs[3]);
       pool_attrs->layout = "NCHW";
       // TODO: tvm has no dialtion but pytorch has, handle dilation
       pool_attrs->ceil_mode = relayToConstant<bool>(inputs[5]);

       static const tvm::relay::Op& op = tvm::relay::Op::Get("nn.max_pool2d");
       auto out = tvm::relay::CallNode::make(op, {inputs[0]}, tvm::Attrs(pool_attrs), {});
       return out;
     }},
});

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
