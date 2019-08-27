#include <relay/op/op_common.h>
#include <relay/op/type_relations.h>
#include <relay/pass/alter_op_layout.h>
#include <tvm/expr.h>
#include <tvm/operation.h>
#include <tvm/relay/op.h>

#include "custom_dense.h"
#include "weight_pack_attrs.h"

namespace tvm {
namespace relay {

bool DenseWeightPackRel(
    const Array<Type>& types,
    int num_inputs, /* unused */
    const Attrs& attrs,
    const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);

  const auto* weight = types[0].as<TensorTypeNode>();
  if (weight == nullptr) {
    return false;
  }
  CHECK(weight->dtype == Float(32));
  CHECK_EQ(weight->shape.size(), 2);

  auto weight_pack_attrs = attrs.as<WeightPackAttrs>();
  int32_t pack_width = weight_pack_attrs->pack_width;
  int32_t out_dim = *as_const_int(weight->shape[0]);
  CHECK_EQ((out_dim % pack_width), 0);
  out_dim = out_dim / pack_width;

  Array<tvm::Expr> oshape = weight->shape;
  oshape.Set(0, out_dim);
  oshape.push_back(pack_width);
  reporter->Assign(types[1], TensorTypeNode::make(oshape, weight->dtype));
  return true;
}

bool CustomDenseRel(
    const Array<Type>& types,
    int num_inputs, /* unused */
    const Attrs& attrs,
    const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);

  const auto* data = types[0].as<TensorTypeNode>();
  const auto* weight = types[1].as<TensorTypeNode>();
  if (data == nullptr || weight == nullptr) {
    return false;
  }
  CHECK_EQ(data->shape.size(), 2);
  CHECK_EQ(weight->shape.size(), 3);
  CHECK_EQ(*as_const_int(data->shape[1]),
      *as_const_int(weight->shape[1]));
  int32_t out_dim = (*as_const_int(weight->shape[0])) *
    (*(as_const_int(weight->shape[2])));
  CHECK_GT(out_dim, 0);

  CHECK(data->dtype == Float(32));

  Array<tvm::Expr> oshape = data->shape;
  oshape.Set((oshape.size() - 1), out_dim);
  reporter->Assign(types[2], TensorTypeNode::make(oshape, data->dtype));
  return true;
}

} // namespace relay
} // namespace tvm
