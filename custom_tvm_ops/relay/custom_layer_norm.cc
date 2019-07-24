#include <tvm/expr.h>
#include <tvm/operation.h>
#include <tvm/relay/op.h>
#include <relay/op/op_common.h>
#include <relay/op/type_relations.h>
#include <relay/pass/alter_op_layout.h>
#include "attrs/layer_norm_attrs.h"

#include <cstdint>

namespace tvm {
namespace relay {

Expr MakeCustomLayerNorm(Expr data, Expr gamma, Expr beta,
    const Array<Integer>& normalize_axis,
    const bool affine) {
  //CHECK(data.as<Tensor>()->shape.size() > normalize_axis.size());
  auto attrs = make_node<CustomLayerNormAttrs>();
  attrs->axis = normalize_axis;
  attrs->affine = affine;
  static const Op& op = Op::Get("nn.custom_layer_norm");
  return CallNode::make(op, {data, gamma, beta}, Attrs(attrs), {});
}

bool CustomLayerNormRel(
    const Array<Type>& types,
    int num_inputs, /* unused */
    const Attrs& attrs,
    const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 4);

  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    return false;
  }
  CHECK_GT(data->shape.size(), 1);
  int64_t data_size = data->shape.size();
  int64_t num_elements = 1;
  for (int64_t i = 0; i < data_size; ++i) {
    CHECK_LE(*as_const_int(data->shape[i]), std::numeric_limits<int>::max());
    num_elements *= *as_const_int(data->shape[i]);
    CHECK_LE(num_elements, std::numeric_limits<int>::max());
  }

  CHECK(data->dtype == Float(32));

  auto layer_norm_attrs_ptr = attrs.as<CustomLayerNormAttrs>();
  auto axis = layer_norm_attrs_ptr->axis;
  CHECK_GT(axis.size(), 0);
  // Right now no negative indexing supported.
  for (const auto& ax : axis) {
    CHECK_GT(*(as_const_int(ax)), 0);
    CHECK_LT(*(as_const_int(ax)), data->shape.size());
  }

  const auto* gamma = types[1].as<TensorTypeNode>();
  const auto* beta = types[2].as<TensorTypeNode>();
  if (gamma && beta) {
    CHECK_EQ(gamma->shape.size(), axis.size());
    CHECK_EQ(beta->shape.size(), axis.size());
    for (int64_t i = 0; i < axis.size(); ++i) {
      int64_t data_index = i + (data_size - axis.size());
      CHECK_EQ(*as_const_int(data->shape[data_index]),
          *as_const_int(gamma->shape[i]));
      CHECK_EQ(*as_const_int(data->shape[data_index]),
          *as_const_int(beta->shape[i]));
    }
  }
  reporter->Assign(types[3], TensorTypeNode::make(data->shape, Float(32)));
  return true;
}

TVM_REGISTER_API("relay.op.nn._make.custom_layer_norm")
    .set_body_typed(MakeCustomLayerNorm);

RELAY_REGISTER_OP("nn.custom_layer_norm")
    .describe(R"code(Applies the layer norm transformation with per element
    affine transform applied after normalization.

- **data**: `Tensor with N dims`
- **out**: `Tensor with N dims`

)code" TVM_ADD_FILELINE)
    .set_attrs_type_key("relay.attrs.CustomLayerNormAttrs")
    .set_num_inputs(3)
    .add_argument("data", "ND Tensor", "Input data.")
    .add_argument("gamma", "ND Tensor", "Input data.")
    .add_argument("beta", "ND Tensor", "Input data.")
    .set_support_level(1)
    .add_type_rel("CustomLayerNorm", CustomLayerNormRel);

} // namespace relay
} // namespace tvm
