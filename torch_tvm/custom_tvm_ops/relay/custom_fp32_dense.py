from __future__ import absolute_import

import topi
from tvm.relay.op import op as reg
from tvm.relay.op.op import OpPattern, schedule_injective

from torch_tvm.custom_tvm_ops.topi import custom_fp32_dense

# dense
@reg.register_compute("nn.custom_dense")
def compute_custom_dense(attrs, inputs, out_type, target):
    out_dtype = inputs[0].dtype
    return [topi.nn.dense(inputs[0], inputs[1], None, out_dtype)]


@reg.register_schedule("nn.custom_dense")
def schedule_dense(attrs, outputs, target):
    with target:
        return topi.generic.schedule_dense(outputs)


reg.register_pattern("nn.custom_dense", reg.OpPattern.OUT_ELEMWISE_FUSABLE)

# weight pack
@reg.register_compute("nn.dense_weight_pack")
def compute_dense_weight_pack(attrs, inputs, out_type, target):
    out_dtype = inputs[0].dtype
    return [custom_fp32_dense.dense_weight_pack(inputs[0], attrs.pack_width)]


@reg.register_schedule("nn.dense_weight_pack")
def schedule_dense_weight_pack(attrs, outputs, target):
    with target:
        return custom_fp32_dense.schedule_dense_weight_pack(outputs)


reg.register_pattern("nn.dense_weight_pack", reg.OpPattern.OPAQUE)
