#pragma once

#include <string>

#include <tvm/relay/expr.h>
#include <tvm/attrs.h>

namespace topi {

Array<Tensor> data_int8_quantize(
    const Tensor& data,
    const Tensor& zero_point,
    const Tensor& scale,
    bool is_signed,
    int precision);

Array<Tensor> data_int8_row_offset(const Tensor& quantized_data);

Array<Tensor> data_int8_mm_dequantize(
    const Tensor& data,
    const Tensor& weight,
    const Tensor& weight_acc,
    const Tensor& data_acc,
    const Tensor& data_scale,
    const Tensor& data_zero_point,
    const double weight_scale,
    const int weight_zero_point,
    const int N);

} // namespace topi
