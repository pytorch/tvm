from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from tvm import relay # This registers all the schedules

from ._torch_tvm import *
from ._torch_tvm import _push_relay_expr
from tvm._ffi.function import _init_api # This lets us use PackedFunc with torch_tvm
_init_api("torch_tvm")

def to_relay(pt_func, inputs):
    if type(pt_func) is not torch._C.Function:
        pt_func = torch.jit.trace(pt_func, inputs)
    handle = _push_relay_expr(pt_func.graph_for(*inputs), inputs)
    return _pop_relay_expr(handle)
