import numpy
import torch
import unittest
from collections import namedtuple
import itertools
import logging
import tvm
import os

source_location = os.path.abspath(__file__)
pytorch_tvm_location = os.path.join(source_location, "../../..")
pytorch_tvm_location = os.path.abspath(pytorch_tvm_location)
logger = logging.getLogger()
lib_path=os.path.join(pytorch_tvm_location, \
        "build/custom_tvm_ops/topi/libcustom_tvm_ops_topi.so")
torch.ops.load_library(lib_path)
lib_path=os.path.join(pytorch_tvm_location, \
        "build/custom_tvm_ops/relay/libcustom_tvm_ops_relay.so")
torch.ops.load_library(lib_path)

# Important to import relay after loading the library because
# importing python/tvm/relay/op/nn/_make.py happens during
# import relay and _make.py initializes internal dictionary of
# apis. If we import it before then the registration that happens
# during loading relay lib will not be visible on the python side.
# TODO: Fix this.
import tvm
from tvm import relay

BuildConfig = namedtuple('BuildConfig', 'ctx target')

class CustomLayerNormUtils(object):
    EPSILON_FLOAT = 1e-9
    @staticmethod
    def pt_layer_norm(a, shape, normalized_axis, weight, bias):
        a_np = a.asnumpy()
        weight_pt = bias_pt = None
        if weight:
            weight_pt = torch.from_numpy(weight.asnumpy())
        if bias:
            bias_pt = torch.from_numpy(bias.asnumpy())
        a_pt = torch.from_numpy(a_np)
        pt_normalized_axis = []
        for i in range(len(normalized_axis)):
            pt_normalized_axis.append(shape[normalized_axis[i]])
        a_out = torch.layer_norm(a_pt, pt_normalized_axis, weight_pt, \
                bias_pt, eps=CustomLayerNormUtils.EPSILON_FLOAT, cudnn_enable=False)
        return a_out.numpy()

    @staticmethod
    def print_schedule(schedule, input_ph, output_ph):
        print(tvm.lower(schedule, [input_ph, output_ph], simple_mode=True))

    @staticmethod
    def optimize_schedule(schedule, output_tensor):
        mean_var_sum = output_tensor.op.input_tensors[0]
        divide_1 = output_tensor.op.input_tensors[1]
        divide_2 = output_tensor.op.input_tensors[2]
        schedule[divide_1].compute_inline()
        schedule[divide_2].compute_inline()
        #schedule[mean_var_sum].compute_at(s[normalized_output_ph], normalized_output_ph.op.axis[0])

    @staticmethod
    def tvm_layer_norm_via_topi(a, a_out, shape, normalized_axis, \
            build_config, weight=None, bias=None):
        ctx = build_config.ctx
        target = build_config.target
        num_axis_to_normalize = len(normalized_axis)
        start_index = len(shape) - num_axis_to_normalize
        weight_shape = shape[start_index:]
        input_ph = tvm.placeholder(shape, name='input_placeholder')
        weights_ph = tvm.placeholder(weight_shape, name='weights_placeholder')
        bias_ph = tvm.placeholder(weight_shape, name='bias_placeholder')
        affine = False
        layer_norm = tvm.get_global_func("nn.custom_layer_norm")
        if weight is not None and bias is not None:
            affine = True
        normalized_output_ph = layer_norm(input_ph, weights_ph, bias_ph, \
                num_axis_to_normalize, affine, CustomLayerNormUtils.EPSILON_FLOAT)
        s = tvm.create_schedule([normalized_output_ph.op])
        #optimize_schedule(s, normalized_output_ph)
        if affine:
            layer_norm_func = tvm.build(s, [input_ph, weights_ph, \
                    bias_ph, normalized_output_ph], target=target)
            layer_norm_func(a, weight, bias, a_out)
        else:
            layer_norm_func = tvm.build(s, [input_ph, normalized_output_ph],
                    target=target)
            layer_norm_func(a, a_out)
        return a_out.asnumpy()

    @staticmethod
    def tvm_layer_norm_via_relay(a, shape, normalized_axis, build_config, \
            weight=None, bias=None):
        ctx = build_config.ctx
        target = build_config.target
        # Must clear the compile engine as it caches the functions compiled.
        # Have not looked thru code yet but it seems that it probably specializes
        # on shapes that are passed in the placeholders below. The issue is that
        # there is only one layer_norm relay op, but making this op will return
        # different Tensor expression depending on the affine parameter being true
        # or not. Now if we dont clear the cache the older function gets used
        # which may correspond to affine or may not.
        # This is a general problem. Caching function should really account for
        # parameters matching as well and not just shapes of input tensors.
        compile_engine = tvm.relay.backend.compile_engine.get()
        tvm.relay.backend.compile_engine.CompileEngine.clear(compile_engine)
        # Assume normalized axis is sorted.
        num_axis_to_normalize = len(normalized_axis)
        start_index = len(shape) - num_axis_to_normalize
        weight_shape = shape[start_index:]
        data_ph = relay.var("data", relay.TensorType(shape, "float32"))
        weight_ph = relay.var("weight_ph", relay.TensorType(weight_shape, "float32"))
        bias_ph = relay.var("bias_ph", relay.TensorType(weight_shape, "float32"))
        affine = False
        if weight is not None and bias is not None:
            affine = True
        # TODO: Should really fix this. Do a python wrapper in tvm style
        # to hide _make call.
        out_ph = relay.op.nn._make.custom_layer_norm(data_ph, weight_ph, bias_ph, \
                num_axis_to_normalize, affine, CustomLayerNormUtils.EPSILON_FLOAT)
        func = relay.Function([data_ph, weight_ph, bias_ph], out_ph)
        func_mod = relay.module.Module.from_expr(func)
        intrp = relay.create_executor("graph", mod=func_mod, ctx=ctx, target=target)
        if affine:
            tvm_out = intrp.evaluate()(a, weight, bias)
        else:
            tvm_out = intrp.evaluate()(a)
        return tvm_out.asnumpy()

    @staticmethod
    def test_custom_layer_norm(shape, normalized_axis, build_config, dtype, affine):
        ctx = build_config.ctx
        a = tvm.nd.array(numpy.random.rand(*shape).astype(dtype), ctx)
        a_out = tvm.nd.array(numpy.zeros(shape, dtype=dtype), ctx)
        weight = None
        bias = None
        if affine:
            weight_shape = shape[normalized_axis[0]:]
            weight = tvm.nd.array(numpy.random.rand(*weight_shape).astype(dtype), ctx)
            bias = tvm.nd.array(numpy.random.rand(*weight_shape).astype(dtype), ctx)

        pt_out = CustomLayerNormUtils.pt_layer_norm(a, shape, normalized_axis, weight, bias)
        tvm_out_via_topi = CustomLayerNormUtils.tvm_layer_norm_via_topi(a, \
                a_out, shape, normalized_axis, build_config, weight, bias)
        numpy.testing.assert_array_almost_equal(pt_out, \
                tvm_out_via_topi, decimal=5)
        tvm_out_via_relay = CustomLayerNormUtils.tvm_layer_norm_via_relay(a, \
                shape, normalized_axis, build_config, weight, bias)
        numpy.testing.assert_array_almost_equal(pt_out, \
                tvm_out_via_relay, decimal=5)

    @staticmethod
    def gen_shapes(shape_list, dims=1):
        shapes = []
        for _ in range(dims):
            shapes.append([s for s in shape_list])
        return itertools.product(*shapes)

class TestFBLayerNorm(unittest.TestCase):
    def setUp(self):
        super(TestFBLayerNorm, self).setUp()
        target="llvm -mcpu=broadwell"
        ctx = tvm.context(target, 0)
        self.build_config = BuildConfig(ctx, target)
        self.dtype = "float32"
        numpy.random.seed(42)

    def test_twodim_array(self):
        dims = [8, 16, 32, 64]
        shapes = CustomLayerNormUtils.gen_shapes(dims, 2)
        for shape in shapes:
            logger.info("Testing shape {},{}".format(*shape))
            normalized_axis = [1]

            CustomLayerNormUtils.test_custom_layer_norm(shape, normalized_axis, \
                    self.build_config, self.dtype, False)
            CustomLayerNormUtils.test_custom_layer_norm(shape, normalized_axis, \
                    self.build_config, self.dtype, True)

    def test_threedim_array(self):
        dims = [8, 16, 32, 64]
        shapes = CustomLayerNormUtils.gen_shapes(dims, 3)
        for shape in shapes:
            logger.info("Testing shape {},{}".format(*shape))
            normalized_axis = [2]

            CustomLayerNormUtils.test_custom_layer_norm(shape, normalized_axis, \
                    self.build_config, self.dtype, False)
            CustomLayerNormUtils.test_custom_layer_norm(shape, normalized_axis, \
                    self.build_config, self.dtype, True)

    def test_threedim_array_2(self):
        dims = [8, 16, 32, 64]
        shapes = CustomLayerNormUtils.gen_shapes(dims, 3)
        for shape in shapes:
            logger.info("Testing shape {},{}".format(*shape))
            normalized_axis = [1, 2]

            CustomLayerNormUtils.test_custom_layer_norm(shape, normalized_axis, \
                    self.build_config, self.dtype, False)
            CustomLayerNormUtils.test_custom_layer_norm(shape, normalized_axis, \
                    self.build_config, self.dtype, True)

if __name__ == "__main__":
    unittest.main()
