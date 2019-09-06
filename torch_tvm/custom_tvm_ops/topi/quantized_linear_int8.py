import tvm
from tvm import autotvm
from tvm.autotvm.task.space import SplitEntity

from topi.nn.util import get_pad_tuple
from topi.nn.pad import pad
from topi.util import simplify, get_const_tuple
from topi.generic import nn
from topi import tag

@tvm.target.generic_func
def quantized_mm_dequantize(data, weight, weight_acc, data_acc, data_scale, \
        data_zero_point, weight_scale, weight_zero_point, N, out_dtype=None):
    quantized_mm_dequant_fn = \
            tvm.get_global_func("nn.compute_quantized_mm_dequantize")
    return quantized_mm_dequant_fn(data, weight, weight_acc, data_acc, data_scale, \
            data_zero_point, weight_scale, weight_zero_point, N)[0]


# The reason to have to register_topi_compute at all is due to the fact that
# "workload" attr is annotated only during this decorator. We need "workload"
# attr for register_topi_schedule which is inturn needed to be able to autotune.
# "workload" can perhaps be annotated from the c++ counterpart but that is not the
# prevailing practice. Plus it also seems cleaner to do it here.
# Although admittedly this all looks like just a hack, but that is due to way
# autotvm machinary works.
@autotvm.register_topi_compute(quantized_mm_dequantize, 'cpu', ['direct'])
def _quantized_mm_dequantize_dummy(cfg, data, weight, weight_acc, data_acc, \
        data_scale, data_zero_point, weight_scale, weight_zero_point, N, \
        out_dtype):
    quantized_mm_dequant_fn = \
            tvm.get_global_func("nn.compute_quantized_mm_dequantize")
    return quantized_mm_dequant_fn(data, weight, weight_acc, data_acc, data_scale, \
            data_zero_point, weight_scale, weight_zero_point, N)[0]


@tvm.target.generic_func
def schedule_quantized_mm_dequantize(outs):
    return nn._default_schedule(outs, False)


@autotvm.register_topi_schedule(schedule_quantized_mm_dequantize, 'cpu', ['direct'])
def _schedule_quantized_mm_dequantize(cfg, outs):
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def traverse(op):
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.tensor.ComputeOp) and \
                        tensor.op not in scheduled_ops:
                    traverse(tensor.op)

        if 'mm_dequantize' in op.tag:
            output = op.output(0)
            _schedule_mm_dequantize(cfg, s, output)

        scheduled_ops.append(op)

    traverse(outs[0].op)
    return s


def _schedule_mm_dequantize(cfg, s, C):
    input_tensors = C.op.input_tensors
    for input_tensor in input_tensors:
        if "quantized_mm" in input_tensor.op.tag:
            _schedule_quantized_mm(cfg, s, input_tensor)


def _schedule_quantized_mm(cfg, s, QGEMM):
    x, y = s[QGEMM].op.axis
    k, = s[QGEMM].op.reduce_axis
    yo, yi = s[QGEMM].split(y, factor=16)
    ko, ki = s[QGEMM].split(k, factor=4)
    s[QGEMM].reorder(yo, ko, x, yi, ki)
    s[QGEMM].unroll(x)
