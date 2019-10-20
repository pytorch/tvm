from __future__ import absolute_import

import topi
from tvm.relay.op import op as reg
from tvm import autotvm
import tvm

from torch_tvm.custom_tvm_ops.topi import quantized_linear_int8

@reg.register_compute("nn.quantize_data_mm_dequantize")
def compute_quantized_mm_dequantize(attrs, inputs, out_type, target):
    data = inputs[0]
    weight = inputs[1]
    weight_acc = inputs[2]
    data_acc = inputs[3]
    data_scale = inputs[4]
    data_zero_point = inputs[5]
    weight_scale = attrs.w_scale
    weight_zero_point = attrs.w_zp
    N = attrs.N
    out = quantized_linear_int8.quantized_mm_dequantize(data, weight, \
            weight_acc, data_acc, data_scale, data_zero_point, weight_scale, \
            weight_zero_point, N, out_type.dtype)
    return [out]


# switch to use C++ version schedule
# @reg.register_schedule("nn.quantize_data_mm_dequantize")
def schedule_quantized_mm_dequantize(attrs, outs, target):
    with target:
        return quantized_linear_int8.schedule_quantized_mm_dequantize(outs)


@reg.register_compute("nn.quantize_data_int8_quantize")
def compute_data_int8_quantize(attrs, inputs, out_type, target):
    data = inputs[0]
    zero_point = inputs[1]
    scale = inputs[2]
    precision = attrs.precision
    is_signed = attrs.is_signed
    out = quantized_linear_int8.data_int8_quantize(data, zero_point, \
            scale, is_signed, precision, out_type.dtype)
    return [out]


@reg.register_schedule("nn.quantize_data_int8_quantize")
def schedule_data_int8_quantize(attrs, outs, target):
    with target:
        return quantized_linear_int8.schedule_data_int8_quantize(outs)


@reg.register_compute("nn.quantize_data_int8_row_offset")
def compute_data_int8_row_offset(attrs, inputs, out_type, target):
    data = inputs[0]
    out = quantized_linear_int8.data_int8_row_offset(data, out_type.dtype)
    return [out]


@reg.register_schedule("nn.quantize_data_int8_row_offset")
def schedule_data_int8_row_offset(attrs, outs, target):
    with target:
        return quantized_linear_int8.schedule_data_int8_row_offset(outs)
