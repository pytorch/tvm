import tvm
from tvm import autotvm
import topi
from topi import nn
from topi.util import get_const_tuple
from topi.x86.dense import _declaration_dense

@autotvm.register_topi_compute(nn.dense, "cpu", "direct", override=True)
def declare_customized_dense(cfg, data, weight, bias=None, out_dtype=None):
    reshape_required = False
    reshaped_data = data
    if len(data.shape) > 2:
        assert (len(data.shape)==3 and len(weight.shape)==2), \
                "Required dims for data=3 and weight=2."
        reshape_required = True
        total_size = 1
        for s in data.shape:
            total_size *= s
        reshaped_data = topi.reshape(data, newshape=(total_size//data.shape[-1], data.shape[-1]))

    result = _declaration_dense(cfg, reshaped_data, weight, bias, out_dtype)
    if reshape_required:
        out_dim, _ = get_const_tuple(weight.shape)
        new_shape = get_const_tuple(data.shape[:-1])
        new_shape = new_shape + (out_dim,)
        result = topi.reshape(result, newshape=new_shape)
    return result

