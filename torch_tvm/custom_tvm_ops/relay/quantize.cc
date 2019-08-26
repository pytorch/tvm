#include <tvm/data_layout.h>
#include <tvm/relay/op.h>
#include <vector>

#include <tvm/expr.h>
#include <tvm/operation.h>
#include <relay/op/op_common.h>
#include <relay/op/type_relations.h>
#include <relay/pass/alter_op_layout.h>
#include "quantize_attrs.h"

#include <cstdint>
namespace tvm {
namespace relay {

Expr MakeDataInt8Quantization(Expr data, Expr zero_point, Expr scale, bool is_signed, int precision) {
  static const Op& op = Op::Get("nn.quantize_data_int8_quantize");
  auto attrs = make_node<QuantizeSchemeAttrs>();
  attrs->precision = precision;
  attrs->is_signed = is_signed;
  return CallNode::make(op, {data, zero_point, scale}, Attrs(attrs), {});
}

bool DataInt8QuantizationRel(const Array<Type>& types,
                             int num_inputs,
                             const Attrs& attrs,
                             const TypeReporter& reporter) {
  // todo: add axis to decide which dim to do the accumulation
  CHECK_EQ(types.size(), 4);
  const QuantizeSchemeAttrs* param = attrs.as<QuantizeSchemeAttrs>();
  const auto* data = types[0].as<TensorTypeNode>();
  // unchnaged shape
  Array<tvm::Expr> oshape = data->shape;
  Array<tvm::Expr> acc_oshape = {oshape[0]};

  DataType out_dtype;
  if(param->is_signed) {
    out_dtype = Int(param->precision);
  } else {
    out_dtype = UInt(param->precision);
  }
  std::vector<Type> fields;
  fields.push_back(TensorTypeNode::make(oshape, out_dtype));
  fields.push_back(TensorTypeNode::make(acc_oshape, Int(32)));
  reporter->Assign(types[3], TupleTypeNode::make(Array<Type>(fields)));
  return true;
}

Expr MakeFindMinMax(Expr data) {
  static const Op& op = Op::Get("nn.quantize_findminmax");
  return CallNode::make(op, {data}, Attrs(), {});
}

bool FindMinMaxRel(const Array<Type>& types,
                   int num_inputs,
                   const Attrs& attrs,
                   const TypeReporter& reporter) {
    CHECK_EQ(types.size(), 2);
    const auto* data = types[0].as<TensorTypeNode>();
    std::vector<IndexExpr> oshape({1});
    std::vector<Type> fields;
    fields.push_back(TensorTypeNode::make(oshape, data->dtype));
    fields.push_back(TensorTypeNode::make(oshape, data->dtype));
    reporter->Assign(types[1], TupleTypeNode::make(Array<Type>(fields)));
    return true;
}

Expr MakeDataMMDequantize(Expr data,
                          Expr weight,
                          Expr weight_acc,
                          Expr data_acc,
                          Expr data_scale,
                          Expr data_zero_point,
                          const double w_scale,
                          const int w_zp) {
  auto attrs = make_node<QuantizedParamsAttrs>();
  attrs->w_scale = w_scale;
  attrs->w_zp = w_zp;
  static const Op& op = Op::Get("nn.quantize_data_mm_dequantize");
  return CallNode::make(op, {data, weight, weight_acc, data_acc, data_scale, data_zero_point}, Attrs(attrs), {});
}

bool DataMMDequantizeRel(const Array<Type>& types,
                             int num_inputs,
                             const Attrs& attrs,
                             const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 7);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* weight = types[1].as<TensorTypeNode>();
  // TODO: check the acc shape
  // Assume acc32 input
  Array<tvm::Expr> wshape = weight->shape;
  Array<tvm::Expr> oshape = data->shape;
  oshape.Set((oshape.size() - 1), wshape[0]);
  reporter->Assign(types[6], TensorTypeNode::make(oshape, Float(32)));
  return true;
}

Expr MakeChooseQuantizeParams(Expr data_min, Expr data_max, bool is_signed, int precision) {
  auto attrs = make_node<QuantizeSchemeAttrs>();
  attrs->precision = precision;
  attrs->is_signed = is_signed;
  static const Op& op = Op::Get("nn.choose_quantize_params");
  return CallNode::make(op, {data_min, data_max}, Attrs(attrs), {});
}

bool ChooseQuantizeParamsRel(const Array<Type>& types,
                             int num_inputs,
                             const Attrs& attrs,
                             const TypeReporter& reporter) {
    CHECK_EQ(types.size(), 3);
    const auto* data = types[0].as<TensorTypeNode>();
    std::vector<IndexExpr> oshape({1});
    std::vector<Type> fields;
    fields.push_back(TensorTypeNode::make(oshape, Int(32)));
    fields.push_back(TensorTypeNode::make(oshape, data->dtype));
    reporter->Assign(types[2], TupleTypeNode::make(Array<Type>(fields)));
    return true;
}

}  // namespace relay
}  // namespace tvm
