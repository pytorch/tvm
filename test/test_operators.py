import unittest
from test.util import TVMTest
from torch.testing import FileCheck

import torch
import torch.nn.functional as F
import torch

# test jit tvm operators


class TestOperators(TVMTest):
    @TVMTest.given(shape=TVMTest.rand_shape(rank=1))
    def test_add(self, shape):
        x = torch.rand(shape)
        y = torch.rand(shape)
        z = torch.rand(shape)

        def add(a, b, c):
            return a + b + c

        ref_out, tvm_out = self.runBoth(add, x, y, z)
        assert torch.allclose(ref_out, tvm_out)

    @TVMTest.given(shape=TVMTest.rand_shape(rank=1))
    def test_mul(self, shape):
        x = torch.rand(shape)
        y = torch.rand(shape)
        z = torch.rand(shape)

        def mul(a, b, c):
            return a * b * c

        ref_out, tvm_out = self.runBoth(mul, x, y, z)
        assert torch.allclose(ref_out, tvm_out)

    @TVMTest.given(
        shape=TVMTest.rand_shape(min_rank=3, max_rank=4, min_dim=4, max_dim=4),
        kernel_size=TVMTest.rand_int(3, 3),
        num_kernels=TVMTest.rand_int(5, 5),
        examples=10,
    )
    def test_conv_simple(self, shape, kernel_size, num_kernels):
        # NCHW
        X = torch.rand(shape)
        if (len(shape) == 4):
            W = torch.rand((num_kernels, shape[1], kernel_size, kernel_size))
            conv_fn = F.conv2d
        else:
            # Case of 1D conv
            W = torch.rand((num_kernels, shape[1], kernel_size))
            conv_fn = F.conv1d
        bias = torch.rand(num_kernels)

        def conv(a, b):
            return conv_fn(a + a, b)

        def conv_bias(a, b, c):
            return conv_fn(a + a, b, c)

        ref_out, tvm_out = self.runBoth(conv, X, W)
        assert torch.allclose(ref_out, tvm_out, rtol=0.01, atol=0.01)
        ref_out, tvm_out = self.runBoth(conv_bias, X, W, bias)
        assert torch.allclose(ref_out, tvm_out, rtol=0.01, atol=0.01)

    @TVMTest.given(
        shape=TVMTest.rand_shape(min_rank=3, max_rank=4, min_dim=15),
        kernel_size=TVMTest.rand_int(3, 6),
        num_kernels=TVMTest.rand_int(),
        stride=TVMTest.rand_list(TVMTest.rand_int(1, 2), 2),
        padding=TVMTest.rand_list(TVMTest.rand_int(0, 4), 2),
        dilation=TVMTest.rand_list(TVMTest.rand_int(
            1, 1), 2),  # TODO known broken in TVM
        examples=10,
    )
    def test_conv_complex(
        self, shape, kernel_size, num_kernels, stride, padding, dilation
    ):
        # NCHW
        X = torch.rand(shape)
        if (len(shape) == 4):
            W = torch.rand(num_kernels, shape[1], kernel_size, kernel_size)
            conv_fn = F.conv2d
        else:
            W = torch.rand(num_kernels, shape[1], kernel_size)
            conv_fn = F.conv1d
            stride = [stride[0]]
            padding = [padding[0]]
            dilation = [dilation[0]]

        def conv(a, b):
            return conv_fn(a + a, b, stride=stride, padding=padding, dilation=dilation)

        ref_out, tvm_out = self.runBoth(conv, X, W)
        assert torch.allclose(ref_out, tvm_out, rtol=0.01, atol=0.01)

    @TVMTest.given(
        shape=TVMTest.rand_shape(min_rank=3, max_rank=4, min_dim=15),
        kernel_size=TVMTest.rand_int(3, 8),
        stride=TVMTest.rand_list(TVMTest.rand_int(1, 2), 2),
        padding=TVMTest.rand_list(TVMTest.rand_int(0, 4), 2),
        dilation=TVMTest.rand_list(TVMTest.rand_int(1, 2), 2),
        groups=TVMTest.rand_int(4, 8),
        in_ch_per_group=TVMTest.rand_int(1, 4),
        out_ch_per_group=TVMTest.rand_int(1, 8),
        examples=10,
    )
    def test_group_conv(
        self, shape, kernel_size, stride, padding, dilation, groups, in_ch_per_group, out_ch_per_group
    ):
        # NCHW
        in_channels = in_ch_per_group * groups
        out_channels = out_ch_per_group * groups
        if (len(shape) == 4):
            X = torch.rand(shape[0], in_channels, shape[1], shape[2])
            W = torch.rand(out_channels, in_ch_per_group, kernel_size, kernel_size)
            conv_fn = F.conv2d
        else:
            X = torch.rand(shape[0], in_channels, shape[1])
            W = torch.rand(out_channels, in_ch_per_group, kernel_size)
            conv_fn = F.conv1d
            stride = [stride[0]]
            padding = [padding[0]]
            dilation = [dilation[0]]

        def conv(a, b):
            return conv_fn(a + a, b, stride=stride, padding=padding, dilation=dilation, groups=groups)

        ref_out, tvm_out = self.runBoth(conv, X, W)
        assert torch.allclose(ref_out, tvm_out, rtol=0.01, atol=0.01)

    @TVMTest.given(shape=TVMTest.rand_shape(rank=2, min_dim=5))
    def test_batch_norm(self, shape):
        a = torch.rand(shape)
        b = torch.rand(shape[1])
        c = torch.rand(shape[1])
        d = torch.rand(shape)

        def batch_norm(a, b, c, d):
            return F.batch_norm(a + d, b, c)

        ref_out, tvm_out = self.runBoth(batch_norm, a, b, c, d)
        assert torch.allclose(ref_out, tvm_out, rtol=0.05, atol=0.01)

    @TVMTest.given(shape=TVMTest.rand_shape(rank=2, min_dim=5))
    def test_batch_norm_weighted(self, shape):
        a = torch.rand(shape)
        b = torch.rand(shape[1])
        c = torch.rand(shape[1])
        d = torch.rand(shape)

        def batch_norm_weighted(a, b, c, d, weight, bias):
            return F.batch_norm(a + d, b, c, weight=weight, bias=bias)

        ref_out, tvm_out = self.runBoth(batch_norm_weighted, a, b, c, d, c, b)
        assert torch.allclose(ref_out, tvm_out, rtol=0.01, atol=0.01)

    @TVMTest.given(shape=TVMTest.rand_shape(min_rank=2, max_rank=4, min_dim=8),\
            examples=20)
    def test_layer_norm(self, shape):
        a = torch.rand(shape)
        axis = shape[1:]
        d = torch.rand(shape)

        def layer_norm(a, d):
            return F.layer_norm(a + d, axis)

        ref_out, tvm_out = self.runBoth(layer_norm, a, d)
        assert torch.allclose(ref_out, tvm_out, rtol=0.05, atol=0.01)

    @TVMTest.given(shape=TVMTest.rand_shape(min_rank=2, max_rank=4, min_dim=8),\
            examples=20)
    def test_layer_norm_weighted(self, shape):
        a = torch.rand(shape)
        b = torch.rand(shape[1:])
        c = torch.rand(shape[1:])
        axis = shape[1:]
        d = torch.rand(shape)

        def layer_norm(a, b, c, d):
            return F.layer_norm(a + d, axis, weight=b, bias=c)

        ref_out, tvm_out = self.runBoth(layer_norm, a, b, c, d)
        assert torch.allclose(ref_out, tvm_out, rtol=0.05, atol=0.01)

    @TVMTest.given(shape=TVMTest.rand_shape())
    def test_relu(self, shape):
        X = torch.rand(shape)

        def relu(a):
            return F.relu(F.relu(a))

        ref_out, tvm_out = self.runBoth(relu, X)
        assert torch.allclose(ref_out, tvm_out, rtol=0.01, atol=0.01)

    @TVMTest.given(shape=TVMTest.rand_shape(min_rank=2, max_rank=4, min_dim=8, max_dim=32),
        examples=20,
        )
    def test_max(self, shape):
        X = torch.rand(shape)
        axis = TVMTest.rand_int(-len(shape), len(shape)-1)()

        def max_fn(a):
            return torch.max(a + a, axis=axis)

        ref_out, tvm_out = self.runBoth(max_fn, X)
        ref_out_values, ref_out_indices = ref_out
        tvm_out_values, tvm_out_indices = tvm_out
        assert torch.allclose(ref_out_values, tvm_out_values, rtol=0.01, atol=0.01)
        assert torch.allclose(ref_out_values, tvm_out_values, rtol=0.01, atol=0.01)

    # Known bug -- stride > 2 has mismatched padding
    @TVMTest.given(
        shape=TVMTest.rand_shape(rank=4, min_dim=4),
        stride=TVMTest.rand_list(TVMTest.rand_int(2, 2), 2),
    )
    def test_avg_pool2d(self, shape, stride):
        X = torch.rand(shape)

        def avg_pool2d(a):
            return F.avg_pool2d(a, 2)

        def avg_pool2d_strides(a):
            return F.avg_pool2d(
                a, 2, stride=stride
            )

        ref_out, tvm_out = self.runBoth(avg_pool2d, X)
        assert torch.allclose(ref_out, tvm_out, rtol=0.01, atol=0.01)
        ref_out, tvm_out = self.runBoth(avg_pool2d_strides, X)
        assert torch.allclose(ref_out, tvm_out, rtol=0.01, atol=0.01)

    @TVMTest.given(
        shape=TVMTest.rand_shape(rank=4, min_dim=4),
    )
    def test_adaptive_avg_pool2d(self, shape):
        X = torch.rand(shape)

        def adaptive_avg_pool2d(a):
            return F.adaptive_avg_pool2d(a, 3)

        ref_out, tvm_out = self.runBoth(adaptive_avg_pool2d, X)
        assert torch.allclose(ref_out, tvm_out, rtol=0.01, atol=0.01)

    # Known bug -- ceil_mode=True sometimes has mismatched shapes
    @TVMTest.given(
        shape=TVMTest.rand_shape(rank=4, min_dim=4),
        stride=TVMTest.rand_list(TVMTest.rand_int(1, 2), 2),
    )
    def test_max_pool2d(self, shape, stride):
        X = torch.rand(shape)

        def max_pool2d(a):
            return F.max_pool2d(a, 3) + 2.0

        def max_pool2d_strides_padding_ceil_mode(a):
            return F.max_pool2d(
                a, 2, stride=stride, padding=1, ceil_mode=False
            )

        # TODO: fix the unstableness when ceil_mode=True case

        ref_out, tvm_out = self.runBoth(max_pool2d, X)
        assert torch.allclose(ref_out, tvm_out, rtol=0.01, atol=0.01)
        ref_out, tvm_out = self.runBoth(max_pool2d_strides_padding_ceil_mode, X)
        assert torch.allclose(ref_out, tvm_out, rtol=0.01, atol=0.01)


    @TVMTest.given(
        shape=TVMTest.rand_shape(rank=2, min_dim=4),
        out_features=TVMTest.rand_int(3, 6),
    )
    def test_fuse_linear_pattern_match(self, shape, out_features):
        input = torch.rand(shape)
        weight = torch.rand(out_features, shape[1])
        bias = torch.rand(out_features)

        def linear_addmm(input, weight, bias):
            return torch.addmm(bias, input, weight.t())

        def linear_matmul_add(input, weight, bias):
            output = input.matmul(weight.t())
            output += bias
            return output

        def linear_matmul(input, weight):
            return input.matmul(weight.t())

        import torch_tvm
        torch_tvm.enable()
        # test addmm
        scripted_addmm = torch.jit.script(linear_matmul_add)
        addmm_graph = scripted_addmm.graph_for(input, weight, bias)
        FileCheck().check("aten::linear").check_not("addmm").check_not("aten::t").run(str(addmm_graph))

        # test matmul + add
        scripted_matmul_add = torch.jit.script(linear_matmul_add)
        matmul_add_graph = scripted_matmul_add.graph_for(input, weight, bias)
        FileCheck().check("aten::linear").check_not("matmul").check_not("aten::t").run(str(matmul_add_graph))

        # test matmul
        scripted_matmul = torch.jit.script(linear_matmul)
        matmul_graph = scripted_matmul.graph_for(input, weight)
        FileCheck().check("aten::linear").check_not("matmul").check_not("aten::t").run(str(matmul_graph))
        torch_tvm.disable()


    @TVMTest.given(
        shape=TVMTest.rand_shape(min_rank=2, max_rank=4, min_dim=4),
        out_features=TVMTest.rand_int(3, 128),
        examples=20,
    )
    def test_linear(self, shape, out_features):
        input = torch.rand(shape)
        weight = torch.rand(out_features, shape[-1])
        bias = torch.rand(out_features)

        def linear(input, weight, bias):
            return F.linear(input + 3.0, weight, bias) + 2.0

        def linear_no_bias(input, weight):
            return F.linear(input, weight) + 2.0

        ref_out, tvm_out = self.runBoth(linear, input, weight, bias)
        assert torch.allclose(ref_out, tvm_out, rtol=0.01, atol=0.01)

        ref_out_no_bias, tvm_out_no_bias = self.runBoth(linear_no_bias, input, weight)
        assert torch.allclose(ref_out_no_bias, tvm_out_no_bias, rtol=0.01, atol=0.01)


    # Have to make min_dim large since we may compare non tensorized imlementation
    # against fbgemm and that requires K dim to be large enough.
    @TVMTest.given(
        shape=TVMTest.rand_shape(rank=2, min_dim=32, max_dim=64),
        out_features=TVMTest.rand_int(15, 64),
    )
    def test_quantized_linear(self, shape, out_features):
        # This is necessary since for N of size > 16 we enforce it to be
        # multiple of 16. Right this defines packing of weights and that
        # specifically makes this requirement necessary.
        # On the other hand loosing this requirement needs changes that require
        # some changes on relay side which otherwise complains during shape
        # propagation in shape inference.
        # Same holds for k.
        shape[1] = shape[1] * 4
        if out_features > 16:
            out_features = out_features * 16
        input = torch.normal(torch.rand(shape))
        weight = torch.normal(torch.rand(out_features, shape[1]))
        bias = torch.normal(torch.rand(out_features))
        q_weight, col_offsets, scale, zero_point = \
            torch.fbgemm_linear_quantize_weight(weight.clone().float())
        packed_weight = torch.fbgemm_pack_quantized_matrix(q_weight.clone())

        def fbgemm_quantized_linear(input, weight, bias, col_offsets):
            return torch.fbgemm_linear_int8_weight(
                input.float(), weight, packed_weight, col_offsets, scale, zero_point, bias.float())
        ref_out, tvm_out = self.runBoth(fbgemm_quantized_linear, input, q_weight, bias, col_offsets)
        # Too loose a bound. We can make this 0.05
        # however this works only when we can tensorize, i.e. use AVX instructions.
        # Since without that we cast int8 to int32 before mul we always get more precision
        # which can result in mismatch. Fix this test once we have AVX2 impl as well.
        assert torch.allclose(ref_out, tvm_out, rtol=0.5, atol=0.5)


    @TVMTest.given(
        shape=TVMTest.rand_shape(rank=2, min_dim=4),
    )
    def test_concat_fuse(self, shape):
        input1 = torch.rand(shape)
        input2 = torch.rand(shape)

        def concat(x1, x2):
            return torch.cat((x1, x2), 0)

        import torch_tvm
        torch_tvm.enable()
        # test concat
        scripted_concat = torch.jit.script(concat)
        concat_graph = scripted_concat.graph_for(input1, input2)
        FileCheck().check("prim::FusedConcat").check_not("prim::ListConstruct").check_not("aten::cat").run(str(concat_graph))

    @TVMTest.given(
        shape=TVMTest.rand_shape(rank=2, min_dim=4),
    )
    def test_concat_op(self, shape):
        input1 = torch.rand(shape)
        input2 = torch.rand(shape)

        # if we didn't use relu, single op will not be lowered
        def concat_relu(x1, x2):
            return F.relu(torch.cat((x1, x2), 0))

        ref_out, tvm_out = self.runBoth(concat_relu, input1, input2)
        assert torch.allclose(ref_out, tvm_out, rtol=0.01, atol=0.01)

    @TVMTest.given(
        shape=TVMTest.rand_shape(rank=3, min_dim=4),
        out_features=TVMTest.rand_int(3, 6),
    )
    def test_bmm(self, shape, out_features):
        input = torch.rand(shape)
        weight = torch.rand(shape[0], shape[-1], out_features)

        def bmm(input, weight):
            return torch.bmm(input, weight) + 2.0

        ref_out, tvm_out = self.runBoth(bmm, input, weight)
        assert torch.allclose(ref_out, tvm_out, rtol=0.01, atol=0.01)

    @TVMTest.given(
        shape=TVMTest.rand_shape(rank=2, min_dim=4),
    )
    def test_reshape(self, shape):
        input = torch.rand(shape)

        def reshape(input):
            return torch.reshape(input, (-1,))

        ref_out, tvm_out = self.runBoth(reshape, input)
        assert torch.allclose(ref_out, tvm_out, rtol=0.01, atol=0.01)

        def reshape(input):
            return torch.reshape(input, (1, 1, *shape))

        ref_out, tvm_out = self.runBoth(reshape, input)
        assert torch.allclose(ref_out, tvm_out, rtol=0.01, atol=0.01)

        def reshape(input):
            return torch.reshape(input, (1, -1))

        ref_out, tvm_out = self.runBoth(reshape, input)
        assert torch.allclose(ref_out, tvm_out, rtol=0.01, atol=0.01)

        def reshape(input):
            return torch.reshape(input, (shape[0], 1, 1, shape[1]))

        ref_out, tvm_out = self.runBoth(reshape, input)
        assert torch.allclose(ref_out, tvm_out, rtol=0.01, atol=0.01)

    @TVMTest.given(
        shape=TVMTest.rand_shape(rank=2, min_dim=4),
        axis=TVMTest.rand_int(0, 1),
    )
    def test_softmax(self, shape, axis):
        input = torch.rand(shape)

        def softmax(input):
            return torch.softmax(input, axis=axis)

        ref_out, tvm_out = self.runBoth(softmax, input)
        assert torch.allclose(ref_out, tvm_out, rtol=0.01, atol=0.01)

    @TVMTest.given(
        shape=TVMTest.rand_shape(rank=1),
    )
    def test_clamp(self, shape):
        input = torch.rand(shape)

        def clamp(input):
            return torch.clamp(input + 3.0, 0.0, 6.0)

        ref_out, tvm_out = self.runBoth(clamp, input)
        assert torch.allclose(ref_out, tvm_out, rtol=0.01, atol=0.01)

if __name__ == "__main__":
    unittest.main()
