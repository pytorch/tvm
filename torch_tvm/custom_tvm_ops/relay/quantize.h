#pragma once

#include <relay/op/op_common.h>
#include <relay/op/type_relations.h>
#include <relay/pass/alter_op_layout.h>
#include <tvm/expr.h>
#include <tvm/operation.h>
#include <tvm/relay/op.h>

namespace tvm {
namespace relay {

Expr MakeDataInt8Quantization(
    Expr data,
    Expr zero_point,
    Expr scale,
    bool is_signed,
    int precision);

bool DataInt8QuantizationRel(
    const Array<Type>& types,
    int num_inputs,
    const Attrs& attrs,
    const TypeReporter& reporter);

Expr MakeFindMinMax(Expr data);

bool FindMinMaxRel(
    const Array<Type>& types,
    int num_inputs,
    const Attrs& attrs,
    const TypeReporter& reporter);

Expr MakeDataMMDequantize(
    Expr data,
    Expr weight,
    Expr weight_acc,
    Expr data_acc,
    Expr data_scale,
    Expr data_zero_point,
    const double w_scale,
    const int w_zp);

bool DataMMDequantizeRel(
    const Array<Type>& types,
    int num_inputs,
    const Attrs& attrs,
    const TypeReporter& reporter);

Expr MakeChooseQuantizeParams(
    Expr data_min,
    Expr data_max,
    bool is_signed,
    int precision);

bool ChooseQuantizeParamsRel(
    const Array<Type>& types,
    int num_inputs,
    const Attrs& attrs,
    const TypeReporter& reporter);

} // namespace relay
} // namespace tvm
