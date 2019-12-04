#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/module.h>
#include <tvm/runtime/device_api.h>
#include <tvm/data_layout.h>

#include <torch/csrc/autograd/record_function.h>
#include <torch/csrc/jit/custom_operator.h>
#include <torch/csrc/jit/operator_options.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

#include "custom_tvm_ops/cpp/relay/custom_layer_norm_attrs.h"
#include "custom_tvm_ops/cpp/relay/quantize_attrs.h"
#include "custom_tvm_ops/cpp/relay/weight_pack_attrs.h"
#include "custom_tvm_ops/cpp/relay/utils.h"
#include "compiler.h"
#include "fusion_pass.h" // tvm_sym
#include "operators.h"

using namespace torch::jit;

// Because tvm does not expose this API.
namespace tvm {
namespace relay {
Expr InferType(const Expr& expr, const Module& mod_ref);
}
}

namespace {
const tvm::relay::TensorTypeNode getExprType(const tvm::relay::Expr& input) {
  auto mod = tvm::relay::ModuleNode::FromExpr(input);
  auto input_type = tvm::relay::InferType(input, mod);
  auto checked_type = input_type->type_as<tvm::relay::TensorTypeNode>();
  TORCH_INTERNAL_ASSERT(checked_type);
  return *checked_type;
}

template <typename T>
T relayToConstant(tvm::relay::Expr e) {
  auto c = e.as<tvm::relay::ConstantNode>();
  TORCH_INTERNAL_ASSERT(c);
  TORCH_INTERNAL_ASSERT(c->is_scalar());
  // TODO: Better error handling on checking if T is same as
  // c->data->dtype. Unfortunately DLDataType is enum of
  // int, uint and float.

  T val {0};
  tvm::runtime::DeviceAPI::Get(c->data->ctx)->CopyDataFromTo(
      c->data->data, 0,
      &val, 0,
      sizeof(val),
      c->data->ctx,
      cpuContext(),
      c->data->dtype, nullptr);
  return val;
}

bool isConstant(tvm::relay::Expr e) {
  auto c = e.as<tvm::relay::ConstantNode>();
  return !!c;
}

template <typename T>
tvm::Array<T> relayToArray(tvm::relay::Expr e) {
  auto t = e.as<tvm::relay::TupleNode>();
  tvm::Array<T> elems;
  for (auto c : t->fields) {
    int elem = relayToConstant<int>(c);
    elems.push_back(elem);
  }
  return elems;
}

tvm::relay::Expr getCustomDense(tvm::relay::Expr data, tvm::relay::Expr weight) {
  auto weight_pack_attrs = tvm::make_node<tvm::relay::WeightPackAttrs>();
  const auto weight_type = getExprType(weight);
  const auto weight_shape = weight_type.shape;
  auto N = (weight_shape[0].as<tvm::IntImm>())->value;
  weight_pack_attrs->pack_width = tvm::relay::helper::get_pack_width(N);
  auto packed_weight = tvm::relay::CallNode::make(
      tvm::relay::Op::Get("nn.dense_weight_pack"),
      {weight},
      tvm::Attrs(weight_pack_attrs),
      {});

  auto dense_attrs = tvm::make_node<tvm::relay::DenseAttrs>();
  auto out = tvm::relay::CallNode::make(
      tvm::relay::Op::Get("nn.custom_dense"),
      {data, packed_weight},
      tvm::Attrs(dense_attrs),
      {});

  return out;
}

tvm::relay::Expr insertInt8WeightTransform(tvm::relay::Expr weight, const int64_t N) {
  std::string out_layout;
  if (N > 16) {
    TORCH_CHECK(N % 16 == 0, "For weights with dim 0 > 16 it must be multiple of 16, got:", N);
    out_layout = "NK16n4k";
  } else {
    out_layout = "NK" + std::to_string(N) + "n4k";
  }
  tvm::Layout src_layout("NK");
  tvm::Layout dst_layout(out_layout);
  auto& transform_op = tvm::relay::Op::Get("layout_transform");
  auto attrs = tvm::make_node<tvm::relay::LayoutTransformAttrs>();
  attrs->src_layout = src_layout.name();
  attrs->dst_layout = dst_layout.name();
  return tvm::relay::CallNode::make(transform_op, {weight}, tvm::Attrs{attrs});
}

tvm::relay::Expr lowerFbgemmLinearInt8Acc32FP32(tvm::Array<tvm::relay::Expr> inputs) {
  // inputs[0]: data_fp32
  // inputs[1]: weight_int8
  // inputs[2]: packed_weight
  // inputs[3]: weight_acc
  // inputs[4]: weight_scale
  // inputs[5]: weight_zp
  // inputs[6]: bias
  TORCH_CHECK(
      inputs.size() == 7, "Given input size from quantized linear", inputs.size(), " inputs");
  auto op_find_minmax = tvm::relay::Op::Get("nn.quantize_findminmax");
  auto op_choose_q_params = tvm::relay::Op::Get("nn.choose_quantize_params");
  auto op_data_q = tvm::relay::Op::Get("nn.quantize_data_int8_quantize");
  auto op_data_row_offset = tvm::relay::Op::Get("nn.quantize_data_int8_row_offset");
  auto op_data_deq = tvm::relay::Op::Get("nn.quantize_data_mm_dequantize");

  const auto input0_type = getExprType(inputs[0]);
  const auto input1_type = getExprType(inputs[1]);

  auto shape0 = input0_type.shape;
  auto shape1 = input1_type.shape;
  TORCH_CHECK(shape0.size() >= 2, "Input must have at least 2 dims.");
  TORCH_CHECK(shape1.size() == 2, "Weight input must have 2 dims.");
  auto reshaped_input = inputs[0];
  if (shape0.size() > 2) {
    uint64_t num_elements_to_flatten = 1;
    int64_t i = 0;
    for (;i < shape0.size()-1; ++i) {
      auto d = (shape0[i].as<tvm::IntImm>())->value;
      num_elements_to_flatten *= d;
    }
    auto last_dim = (shape0[i].as<tvm::IntImm>())->value;
    auto attrs = tvm::make_node<tvm::relay::ReshapeAttrs>();
    auto& newshape = attrs->newshape;
    newshape.push_back(tvm::Integer(num_elements_to_flatten));
    newshape.push_back(tvm::Integer(last_dim));
    attrs->reverse = false;
    auto op = tvm::relay::Op::Get("reshape");
    reshaped_input =
        tvm::relay::CallNode::make(op, {inputs[0]}, tvm::Attrs(attrs), {});
  }

  auto min_max_call = tvm::relay::CallNode::make(op_find_minmax, {reshaped_input}, tvm::Attrs(), {});
  const tvm::relay::Expr& data_min = tvm::relay::TupleGetItemNode::make(min_max_call, 0);
  const tvm::relay::Expr& data_max = tvm::relay::TupleGetItemNode::make(min_max_call, 1);

  auto scheme_attrs = tvm::make_node<tvm::relay::QuantizeSchemeAttrs>();
  scheme_attrs->precision = 8;
  scheme_attrs->is_signed = false;

  auto q_params = tvm::relay::CallNode::make(op_choose_q_params, {data_min, data_max}, tvm::Attrs(scheme_attrs), {});
  const tvm::relay::Expr& data_zp = tvm::relay::TupleGetItemNode::make(q_params, 0);
  const tvm::relay::Expr& data_scale = tvm::relay::TupleGetItemNode::make(q_params, 1);

  // data, zero_point, scale
  auto q_data = tvm::relay::CallNode::make(op_data_q, {reshaped_input, data_zp, data_scale},
                                                tvm::Attrs(scheme_attrs), {});
  auto q_data_row_offset = tvm::relay::CallNode::make(op_data_row_offset,
      {q_data}, tvm::Attrs(), {});

  const auto weight_type = getExprType(inputs[1]);
  auto weight_shape = weight_type.shape;
  int N = (weight_shape[0].as<tvm::IntImm>())->value;
  tvm::relay::Expr packed_weight = insertInt8WeightTransform(inputs[1], N);

  tvm::Array<tvm::relay::Expr> deq_inputs = {q_data, packed_weight, inputs[3],
    q_data_row_offset, data_scale, data_zp};

  auto params_attrs = tvm::make_node<tvm::relay::QuantizedParamsAttrs>();
  params_attrs->w_scale= static_cast<double>(relayToConstant<float>(inputs[4]));
  params_attrs->w_zp = relayToConstant<int>(inputs[5]);
  params_attrs->N = N;

  auto mm = tvm::relay::CallNode::make(op_data_deq, deq_inputs, tvm::Attrs(params_attrs), {});

  if (shape0.size() > 2) {
    auto attrs = tvm::make_node<tvm::relay::ReshapeAttrs>();
    for (int64_t i = 0;i < shape0.size() - 1; ++i) {
      attrs->newshape.push_back((shape0[i].as<tvm::IntImm>())->value);
    }
    attrs->newshape.push_back((shape1[0].as<tvm::IntImm>())->value);
    attrs->reverse = false;
    auto op = tvm::relay::Op::Get("reshape");
    mm = tvm::relay::CallNode::make(op, {mm}, tvm::Attrs(attrs), {});
  }

  auto bias_add_op = tvm::relay::Op::Get("nn.bias_add");
  auto bias_add_attrs = tvm::make_node<tvm::relay::BiasAddAttrs>();
  bias_add_attrs->axis = -1;
  return tvm::relay::CallNode::make(bias_add_op, {mm, inputs[6]}, tvm::Attrs(bias_add_attrs), {});
}

tvm::relay::Expr lowerLinear(tvm::Array<tvm::relay::Expr> inputs) {
  const auto input0_type = getExprType(inputs[0]);
  const auto input1_type = getExprType(inputs[1]);

  auto shape0 = input0_type.shape;
  auto shape1 = input1_type.shape;
  TORCH_CHECK(shape0.size() >= 2, "Input must have at least 2 dims.");
  TORCH_CHECK(shape1.size() == 2, "Weight input must have 2 dims.");
  auto reshaped_input = inputs[0];
  if (shape0.size() > 2) {
    uint64_t num_elements_to_flatten = 1;
    int64_t i = 0;
    for (;i < shape0.size()-1; ++i) {
      auto d = (shape0[i].as<tvm::IntImm>())->value;
      num_elements_to_flatten *= d;
    }
    auto last_dim = (shape0[i].as<tvm::IntImm>())->value;
    auto attrs = tvm::make_node<tvm::relay::ReshapeAttrs>();
    auto& newshape = attrs->newshape;
    newshape.push_back(tvm::Integer(num_elements_to_flatten));
    newshape.push_back(tvm::Integer(last_dim));
    attrs->reverse = false;
    auto op = tvm::relay::Op::Get("reshape");
    reshaped_input =
        tvm::relay::CallNode::make(op, {inputs[0]}, tvm::Attrs(attrs), {});
  }

  //Make sure both inputs have same batch dim.
  const auto batch_dim0 = (shape0[0].as<tvm::IntImm>())->value;
  const auto batch_dim1 = (shape0[0].as<tvm::IntImm>())->value;
  TORCH_CHECK((batch_dim0 == batch_dim1),
      "Input and weight must have same value in batch dim.");

  //TODO: Right now custom dense is only supported for cpu target.
  //Any other target will likely break, e.g. arm_cpu.
  //Future fix.
  tvm::relay::Expr out;
  out = getCustomDense(reshaped_input, inputs[1]);

  if (shape0.size() > 2) {
    auto attrs = tvm::make_node<tvm::relay::ReshapeAttrs>();
    for (int64_t i = 0;i < shape0.size() - 1; ++i) {
      attrs->newshape.push_back((shape0[i].as<tvm::IntImm>())->value);
    }
    attrs->newshape.push_back((shape1[0].as<tvm::IntImm>())->value);
    attrs->reverse = false;
    auto op = tvm::relay::Op::Get("reshape");
    out = tvm::relay::CallNode::make(op, {out}, tvm::Attrs(attrs), {});
  }

  if (!relayIsNone(inputs[2])) {
    const auto input2_type = getExprType(inputs[2]);
    auto shape2 = input2_type.shape;
    TORCH_CHECK(shape2.size() == 1, "Bias input must be 1 dim.");
    auto bias_add_op = tvm::relay::Op::Get("nn.bias_add");
    auto bias_add_attrs = tvm::make_node<tvm::relay::BiasAddAttrs>();
    bias_add_attrs->axis = -1;
    return tvm::relay::CallNode::make(
        bias_add_op, {out, inputs[2]}, tvm::Attrs(bias_add_attrs), {});
  }
  return out;
}

tvm::relay::Call insertDims(const tvm::relay::Expr& in, int64_t input_dims,
    int64_t num_dims_to_add) {
  auto expand_dims = tvm::relay::Op::Get("expand_dims");
  auto attrs = tvm::make_node<tvm::relay::ExpandDimsAttrs>();
  attrs->axis = input_dims;
  attrs->num_newaxis = num_dims_to_add;
  return tvm::relay::CallNode::make(expand_dims, {in}, tvm::Attrs(attrs), {});
}

tvm::relay::Call squeezeSingleDim(const tvm::relay::Expr& in,
    int32_t dim_to_squeeze) {
  auto squeeze = tvm::relay::Op::Get("squeeze");
  auto attrs = tvm::make_node<tvm::relay::SqueezeAttrs>();
  attrs->axis = {dim_to_squeeze};
  return tvm::relay::CallNode::make(squeeze, {in}, tvm::Attrs(attrs), {});
}

} // namespace

std::unordered_map<std::string, TVMScheduleFunctor>& getTVMScheduleMap() {
  static std::unordered_map<std::string, TVMScheduleFunctor> map;
  return map;
}

std::unordered_map<Symbol, TVMOpFunctor>& getTVMOperatorMap() {
  static std::unordered_map<Symbol, TVMOpFunctor> map;
  return map;
}

std::unordered_map<Symbol, std::vector<int32_t>>& getOpParamsMap() {
  static std::unordered_map<Symbol, std::vector<int32_t>> param_map;
  return param_map;
}

// These "wrapper" graphs are used to store the subgraphs
// that will be compiled during execution.
static std::list<Graph> wrapper_graphs;
RegisterTVMOperator::RegisterTVMOperator(std::vector<TVMOpMap> ops) {
  for (const auto& op : ops) {
    getTVMOperatorMap()[op.sym] = op.fn;
    getOpParamsMap()[op.sym] = op.param_indices;

    if (op.name != "") {
      auto torch_ops = getAllOperatorsFor(op.sym);

      for (const auto& torch_op : torch_ops) {
        auto schema = torch_op->schema();

        wrapper_graphs.emplace_back();
        auto& wrapper_graph = wrapper_graphs.back();
        std::vector<Value*> torch_inputs;
        for (const auto& inp : schema.arguments()) {
          torch_inputs.emplace_back(wrapper_graph.addInput());
        }
        Node* node =
            wrapper_graph.create(op.sym, torch_inputs, schema.returns().size());
        wrapper_graph.appendNode(node);
        wrapper_graph.registerOutput(node->output());

        node = SubgraphUtils::createSingletonSubgraph(node, getTVMSymbol());
        auto cc = std::make_shared<TVMCompiler>(node);

        // NB: We assume all relay ops are pure
        auto options = c10::OperatorOptions();
        options.setAliasAnalysis(AliasAnalysisKind::PURE_FUNCTION);
        // TODO: Pass in operator options somehow
        auto torch_operator = Operator(
            FunctionSchema(
                "tvm::" + op.name,
                "",
                schema.arguments(),
                schema.returns(),
                false,
                false),
            [cc](Stack& stack) {
              RECORD_FUNCTION("TVM", std::vector<c10::IValue>());
              cc->run(stack);
              return 0;
            });
        RegisterOperators torch_register_ops(
            std::vector<Operator>{torch_operator});
      }
    }
  }
}

// This must be done lazily to prevent SIOF
void registerSchedule(std::string name) {
  TORCH_INTERNAL_ASSERT(getTVMScheduleMap().find(name) != getTVMScheduleMap().end());
  TVMScheduleFunctor sched_f = getTVMScheduleMap()[name];
  auto reg = tvm::runtime::Registry::Get("relay.op._Register");
  TORCH_INTERNAL_ASSERT(reg);
  auto sched = sched_f();
  // Relay does not provide a good API for querying the status of schedules
  if (sched) {
    (*reg)(name, "FTVMSchedule", *sched, 10);
    getTVMScheduleMap()[name] = []() { return nullptr; };
  }
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
       TORCH_INTERNAL_ASSERT(inputs.size() == 3);
       tvm::Array<tvm::relay::Expr> add_inputs = {inputs[0], inputs[1]};
       // Handle pytorch's value argument in add
       auto value = inputs[2].as<tvm::relay::ConstantNode>();
       int val {0};
       tvm::runtime::DeviceAPI::Get(value->data->ctx)->CopyDataFromTo(
           value->data->data, 0,
           &val, 0,
           sizeof(val),
           value->data->ctx,
           cpuContext(),
           value->data->dtype, nullptr);
       TORCH_INTERNAL_ASSERT(value->is_scalar() && val == 1);

       auto out = tvm::relay::CallNode::make(op, add_inputs, tvm::Attrs(), {});
       return out;
     }},
    {Symbol::fromQualString("aten::max"),
     [](Node* node, tvm::Array<tvm::relay::Expr> inputs) {
       TORCH_INTERNAL_ASSERT(inputs.size() == 3);
       auto make_func_ptr = tvm::runtime::Registry::Get("relay.op._make.max");
       auto make_func = *make_func_ptr;
       auto axis = relayToConstant<int>(inputs[1]);
       auto keepdims = relayToConstant<bool>(inputs[2]);
       tvm::Array<tvm::Integer> axis_arr({tvm::Integer(axis)});
       auto max_values = make_func(inputs[0], axis_arr, keepdims, false);
       make_func_ptr = tvm::runtime::Registry::Get("relay.op._make.argmax");
       make_func = *make_func_ptr;
       auto max_indices= make_func(inputs[0], axis_arr, keepdims, false);
       return tvm::relay::Expr(tvm::relay::TupleNode::make({max_values, max_indices}));
     }},
    {Symbol::fromQualString("aten::add_"),
     [](Node* node, tvm::Array<tvm::relay::Expr> inputs) {
       auto op = tvm::relay::Op::Get("add");
       TORCH_INTERNAL_ASSERT(inputs.size() == 3);
       tvm::Array<tvm::relay::Expr> add_inputs = {inputs[0], inputs[1]};
       // Handle pytorch's value argument in add
       auto value = inputs[2].as<tvm::relay::ConstantNode>();
       int val {0};
       tvm::runtime::DeviceAPI::Get(value->data->ctx)->CopyDataFromTo(
           value->data->data, 0,
           &val, 0,
           sizeof(val),
           value->data->ctx,
           cpuContext(),
           value->data->dtype, nullptr);
       TORCH_INTERNAL_ASSERT(value->is_scalar() && val == 1);

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
       conv_attrs->strides = relayToArray<tvm::relay::IndexExpr>(inputs[3]);
       conv_attrs->padding = relayToArray<tvm::relay::IndexExpr>(inputs[4]);
       conv_attrs->dilation = relayToArray<tvm::relay::IndexExpr>(inputs[5]);

       conv_attrs->kernel_size =
           tvm::NullValue<tvm::Array<tvm::relay::IndexExpr>>();
       // For 1D conv need to expand dim. That means it has to be removed
       // as well.
       bool dim_added{false}; // For 1D conv need to expand dim.
       // If the input was a complete tensor type than we have information to
       // populate the kernel
       if (const tvm::relay::VarNode* var =
               inputs[1].as<tvm::relay::VarNode>()) {
         auto* w_t = var->type_annotation.as<tvm::relay::TensorTypeNode>();
         TORCH_INTERNAL_ASSERT(w_t);
         auto shape = w_t->shape;
         auto num_dims = shape.size();
         tvm::Array<tvm::relay::IndexExpr> w_sizes;
         if (num_dims < 4) {
           AT_CHECK(num_dims == 3,
               "Expected number of min dims for convolution is 3, found: ",
               num_dims);
           AT_CHECK(conv_attrs->strides.size() == 1,
               "Expected strides size for 1D conv is 1, found: ",
               conv_attrs->strides.size());
           AT_CHECK(conv_attrs->padding.size() == 1,
               "Expected strides size for 1D conv is 1, found: ",
               conv_attrs->padding.size());
           AT_CHECK(conv_attrs->dilation.size() == 1,
               "Expected strides size for 1D conv is 1, found: ",
               conv_attrs->dilation.size());
           new_inputs.Set(0, insertDims(inputs[0], num_dims, 1));
           new_inputs.Set(1, insertDims(inputs[1], num_dims, 1));
           w_sizes = {shape[2], tvm::relay::IndexExpr(1)};
           conv_attrs->strides.push_back(tvm::relay::IndexExpr(1));
           conv_attrs->padding.push_back(tvm::relay::IndexExpr(0));
           conv_attrs->dilation.push_back(tvm::relay::IndexExpr(1));
           dim_added = true;
         }
         else {
           w_sizes = {shape[2], shape[3]};
         }
         conv_attrs->kernel_size = w_sizes;
       }
       else {
         TORCH_INTERNAL_ASSERT(0, "Kernel information must be available.")
       }

       auto out = tvm::relay::CallNode::make(
           op, new_inputs, tvm::Attrs(conv_attrs), {});

       if (dim_added) {
         out = squeezeSingleDim(out, -1);
       }
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
     },
     "", PARAM_INDICES(convolution)},
    {Symbol::fromQualString("prim::FusedConcat"),
     [](Node* node, tvm::Array<tvm::relay::Expr> inputs) {
       auto op = tvm::relay::Op::Get("concatenate");
       auto concat_inputs = tvm::relay::TupleNode::make(inputs);
       auto attrs = tvm::make_node<tvm::relay::ConcatenateAttrs>();
       attrs->axis = node->i(attr::dim);
       auto out = tvm::relay::CallNode::make(op, {concat_inputs}, tvm::Attrs(attrs), {});
       return out;
     }},
    {Symbol::fromQualString("aten::layer_norm"),
     [](Node* node, tvm::Array<tvm::relay::Expr> inputs) -> tvm::relay::Expr {
       auto op = tvm::relay::Op::Get("nn.custom_layer_norm");
       TORCH_CHECK(
           inputs.size() == 6,
           "layer_norm received ",
           inputs.size(),
           " inputs");
       auto normalized_axis_shape = relayToArray<tvm::Integer>(inputs[1]);
       std::vector<int64_t> shape;
       shape.reserve(normalized_axis_shape.size());
       for (auto& dim : normalized_axis_shape) {
         shape.push_back(dim);
       }
       auto attrs = tvm::make_node<tvm::relay::CustomLayerNormAttrs>();
       attrs->num_axis_to_normalize = normalized_axis_shape.size();
       auto eps = static_cast<double>(relayToConstant<float>(inputs[4]));
       attrs->eps = eps;

       TVMContext ctx_;
       ctx_.device_type = kDLCPU;
       ctx_.device_id = 0;

       //TODO: Find a way to obtain type of data held by tensor.
       //Perhaps via invoking type inferencer.
       tvm::runtime::NDArray weight_temp, bias_temp;
       weight_temp = tvm::runtime::NDArray::Empty(shape, tvm::Float(32), ctx_);
       bias_temp = tvm::runtime::NDArray::Empty(shape, tvm::Float(32), ctx_);
       tvm::relay::Expr weight, bias;
       weight = tvm::relay::ConstantNode::make(weight_temp);
       bias = tvm::relay::ConstantNode::make(bias_temp);
       if (!relayIsNone(inputs[2])) {
         TORCH_CHECK(!relayIsNone(inputs[3]), "If weight tensor is present"
             "then bias tensor must be present as well.");
         attrs->affine = true;
         weight = inputs[2];
         bias = inputs[3];
       }

       tvm::Array<tvm::relay::Expr> ln_inputs = {
           inputs[0],
           weight,
           bias,
       };

       auto out =
           tvm::relay::CallNode::make(op, ln_inputs, tvm::Attrs(attrs), {});
       TORCH_CHECK(node->outputs().size() == 1);
       return out;
     },
     "", PARAM_INDICES(layer_norm)},
    {Symbol::fromQualString("aten::batch_norm"),
     [](Node* node, tvm::Array<tvm::relay::Expr> inputs) -> tvm::relay::Expr {
       auto op = tvm::relay::Op::Get("nn.batch_norm");
       TORCH_CHECK(
           inputs.size() == 9,
           "batch_norm received ",
           inputs.size(),
           " inputs");
       TORCH_CHECK(
           relayToConstant<bool>(inputs[5]) == false,
           "batch_norm is in training mode");
       auto attrs = tvm::make_node<tvm::relay::BatchNormAttrs>();
       auto eps = relayToConstant<float>(inputs[7]);
       attrs->epsilon = eps;
       attrs->axis = 1;
       attrs->scale = false;
       attrs->center = false;

       TVMContext ctx_;
       ctx_.device_type = kDLCPU;
       ctx_.device_id = 0;
       auto x = tvm::runtime::NDArray::Empty(
           {}, tvm::runtime::String2TVMType("float32"), ctx_);
       // Make this large to induce noticeable errors
       float x_val = 1337e10;
       tvm::runtime::DeviceAPI::Get(x->ctx)->CopyDataFromTo(
         &x_val, 0,
         x->data, 0,
         sizeof(x_val),
         cpuContext(),
         x->ctx,
         x->dtype, nullptr);

       tvm::relay::Expr v = tvm::relay::ConstantNode::make(x);

       auto& broadcast = tvm::relay::Op::Get("broadcast_to_like");
       tvm::relay::Expr weight = tvm::relay::CallNode::make(
           broadcast, {v, inputs[3]}, tvm::Attrs(), {});
       tvm::relay::Expr bias = tvm::relay::CallNode::make(
           broadcast, {v, inputs[3]}, tvm::Attrs(), {});

       // TODO check if pytorch semantics allow these to be broadcast
       if (!relayIsNone(inputs[1])) {
         attrs->scale = true;
         weight = inputs[1];
       }
       if (!relayIsNone(inputs[2])) {
         attrs->center = true;
         bias = inputs[2];
       }

       tvm::Array<tvm::relay::Expr> bn_inputs = {
           inputs[0],
           weight,
           bias,
           inputs[3],
           inputs[4],
       };

       auto out =
           tvm::relay::CallNode::make(op, bn_inputs, tvm::Attrs(attrs), {});
       TORCH_INTERNAL_ASSERT(node->outputs().size() == 1);
       if (node->outputs().size() == 2) {
         return out;
       }
       auto n = tvm::make_node<tvm::relay::TupleGetItemNode>();
       n->tuple = std::move(out);
       n->index = 0;
       return tvm::relay::TupleGetItem(n);
     }},
    {Symbol::fromQualString("aten::fbgemm_linear_int8_weight"),
     [](Node* node, tvm::Array<tvm::relay::Expr> inputs) {
       return lowerFbgemmLinearInt8Acc32FP32(inputs);
     },
     "", PARAM_INDICES(quantized_linear)},
    {Symbol::fromQualString("aten::relu_"),
     [](Node* node, tvm::Array<tvm::relay::Expr> inputs) {
       auto op = tvm::relay::Op::Get("nn.relu");
       auto out = tvm::relay::CallNode::make(op, inputs, tvm::Attrs(), {});
       return out;
     }},
    {Symbol::fromQualString("aten::relu"),
     [](Node* node, tvm::Array<tvm::relay::Expr> inputs) {
       auto op = tvm::relay::Op::Get("nn.relu");
       auto out = tvm::relay::CallNode::make(op, inputs, tvm::Attrs(), {});
       return out;
     },
     "relu"},
    {Symbol::fromQualString("aten::threshold_"),
     [](Node* node, tvm::Array<tvm::relay::Expr> inputs) {
       TORCH_CHECK(!relayIsNone(inputs[0]));
       TORCH_CHECK(!relayIsNone(inputs[1]));
       TORCH_CHECK(!relayIsNone(inputs[2]));
       auto d = relayToConstant<float>(inputs[1]);
       TORCH_CHECK(
           d < 1e-7, "aten::threshold_ only supported for threshold 0, got ", d);
       TORCH_CHECK(
           d > -1e-7,
           "aten::threshold_ only supported for threshold 0, got",
           d);
       d = relayToConstant<float>(inputs[2]);
       TORCH_CHECK(
           d < 1e-7, "aten::threshold_ only supported for value 0, got ", d);
       TORCH_CHECK(
           d > -1e-7, "aten::threshold_ only supported for value 0, got ", d);
       auto op = tvm::relay::Op::Get("nn.relu");
       auto out = tvm::relay::CallNode::make(op, {inputs[0]}, tvm::Attrs(), {});
       return out;
     }},
    {Symbol::fromQualString("aten::mul"),
     [](Node* node, tvm::Array<tvm::relay::Expr> inputs) {
       auto op = tvm::relay::Op::Get("multiply");
       auto out = tvm::relay::CallNode::make(op, inputs, tvm::Attrs(), {});
       return out;
     }},
    {Symbol::fromQualString("aten::clamp"),
     [](Node* node, tvm::Array<tvm::relay::Expr> inputs) {
       auto op = tvm::relay::Op::Get("clip");
       auto clip_attrs = tvm::make_node<tvm::relay::ClipAttrs>();
       clip_attrs->a_min = relayToConstant<float>(inputs[1]);
       clip_attrs->a_max = relayToConstant<float>(inputs[2]);
       auto out = tvm::relay::CallNode::make(
         op, {inputs[0]}, tvm::Attrs(clip_attrs), {});
       return out;
     }},
    {Symbol::fromQualString("aten::avg_pool2d"),
     [](Node* node, tvm::Array<tvm::relay::Expr> inputs) {
       auto op = tvm::relay::Op::Get("nn.avg_pool2d");
       auto pool_attrs = tvm::make_node<tvm::relay::AvgPool2DAttrs>();
       pool_attrs->pool_size = relayToArray<tvm::relay::IndexExpr>(inputs[1]);
       auto strides = relayToArray<tvm::relay::IndexExpr>(inputs[2]);
       if (strides.size() == 0) {
         // pytorch avg_pool2d semantic: strides default to pool size
         pool_attrs->strides = pool_attrs->pool_size;
       } else {
         pool_attrs->strides = strides;
       }
       pool_attrs->padding = relayToArray<tvm::relay::IndexExpr>(inputs[3]);
       pool_attrs->layout = "NCHW";
       pool_attrs->ceil_mode = relayToConstant<bool>(inputs[4]);
       pool_attrs->count_include_pad = relayToConstant<bool>(inputs[5]);

       auto out = tvm::relay::CallNode::make(
           op, {inputs[0]}, tvm::Attrs(pool_attrs), {});
       return out;
     }},
    {Symbol::fromQualString("aten::adaptive_avg_pool2d"),
     [](Node* node, tvm::Array<tvm::relay::Expr> inputs) {
       static const tvm::relay::Op& op =
           tvm::relay::Op::Get("contrib.adaptive_avg_pool2d");
       auto pool_attrs = tvm::make_node<tvm::relay::AdaptivePool2DAttrs>();
       pool_attrs->output_size = relayToArray<tvm::relay::IndexExpr>(inputs[1]);
       pool_attrs->layout = "NCHW";
       auto out = tvm::relay::CallNode::make(
           op, {inputs[0]}, tvm::Attrs(pool_attrs), {});
       return out;
     }},
    {Symbol::fromQualString("aten::max_pool2d"),
     [](Node* node, tvm::Array<tvm::relay::Expr> inputs) {
       auto pool_attrs = tvm::make_node<tvm::relay::MaxPool2DAttrs>();
       pool_attrs->pool_size = relayToArray<tvm::relay::IndexExpr>(inputs[1]);
       auto strides = relayToArray<tvm::relay::IndexExpr>(inputs[2]);
       if (strides.size() == 0) {
         // pytorch max_pool2d semantic: strides default to pool size
         pool_attrs->strides = pool_attrs->pool_size;
       } else {
         pool_attrs->strides = strides;
       }
       pool_attrs->padding = relayToArray<tvm::relay::IndexExpr>(inputs[3]);
       pool_attrs->layout = "NCHW";
       // TODO: tvm has no dialtion but pytorch has, handle dilation
       pool_attrs->ceil_mode = relayToConstant<bool>(inputs[5]);

       static const tvm::relay::Op& op = tvm::relay::Op::Get("nn.max_pool2d");
       auto out = tvm::relay::CallNode::make(
           op, {inputs[0]}, tvm::Attrs(pool_attrs), {});
       return out;
     }},
    {Symbol::fromQualString("aten::reshape"),
     [](Node* node, tvm::Array<tvm::relay::Expr> inputs) {
       auto op = tvm::relay::Op::Get("reshape");
       auto attrs = tvm::make_node<tvm::relay::ReshapeAttrs>();
       attrs->newshape = relayToArray<tvm::Integer>(inputs[1]);
       TORCH_INTERNAL_ASSERT(attrs->newshape.size() > 0);
       if (static_cast<int64_t>(attrs->newshape[0]) != -1) {
         LOG(WARNING) << "reshape with -1 as the first value has known incompatibility with PyTorch semantics.\n";
       }
       attrs->reverse = false;
       auto out =
           tvm::relay::CallNode::make(op, {inputs[0]}, tvm::Attrs(attrs), {});
       return out;
     }},
    {Symbol::fromQualString("aten::linear"),
     [](Node* node, tvm::Array<tvm::relay::Expr> inputs) {
       return lowerLinear(inputs);
     },
     "", PARAM_INDICES(linear)},
     {Symbol::fromQualString("aten::softmax"),
     [](Node* node, tvm::Array<tvm::relay::Expr> inputs) {
       auto softmax_op = tvm::relay::Op::Get("nn.softmax");
       auto softmax_attrs = tvm::make_node<tvm::relay::SoftmaxAttrs>();
       auto axis = relayToConstant<int64_t>(inputs[1]);
       softmax_attrs->axis = axis;
       return tvm::relay::CallNode::make(
         softmax_op,
         { inputs[0] },
         tvm::Attrs(softmax_attrs));
     }},
});

bool isSupported(Node* node) {
  auto map = getTVMOperatorMap();
  return map.find(node->kind()) != map.end();
}

const std::vector<int32_t>& getParamIndices(Node* node) {
  TORCH_INTERNAL_ASSERT(isSupported(node));
  return getOpParamsMap()[node->kind()];
}

tvm::relay::Expr getOperator(Node* node, tvm::Array<tvm::relay::Expr> inputs) {
  TORCH_INTERNAL_ASSERT(isSupported(node));
  return getTVMOperatorMap()[node->kind()](node, inputs);
}
