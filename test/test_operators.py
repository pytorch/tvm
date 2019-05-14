import unittest
from test.util import TVMTest

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
        shape=TVMTest.rand_shape(rank=4, min_dim=4, max_dim=4),
        kernel_size=TVMTest.rand_int(3, 3),
        num_kernels=TVMTest.rand_int(5, 5),
    )
    def test_conv_simple(self, shape, kernel_size, num_kernels):
        # NCHW
        X = torch.rand(shape)
        W = torch.rand((num_kernels, shape[1], kernel_size, kernel_size))
        bias = torch.rand(num_kernels)

        def conv(a, b):
            return F.conv2d(a + a, b)

        def conv_bias(a, b, c):
            return F.conv2d(a + a, b, c)

        ref_out, tvm_out = self.runBoth(conv, X, W)
        assert torch.allclose(ref_out, tvm_out, rtol=0.01, atol=0.01)
        ref_out, tvm_out = self.runBoth(conv_bias, X, W, bias)
        assert torch.allclose(ref_out, tvm_out, rtol=0.01, atol=0.01)

    @TVMTest.given(
        shape=TVMTest.rand_shape(rank=4, min_dim=15),
        kernel_size=TVMTest.rand_int(3, 6),
        num_kernels=TVMTest.rand_int(),
        stride=TVMTest.rand_list(TVMTest.rand_int(1, 2), 2),
        padding=TVMTest.rand_list(TVMTest.rand_int(0, 4), 2),
        dilation=TVMTest.rand_list(TVMTest.rand_int(1, 1), 2), # TODO known broken in TVM
    )
    def test_conv_complex(
        self, shape, kernel_size, num_kernels, stride, padding, dilation
    ):
        # NCHW
        X = torch.rand(shape)
        W = torch.rand(num_kernels, shape[1], kernel_size, kernel_size)

        def conv(a, b):
            return F.conv2d(a + a, b, stride=stride, padding=padding, dilation=dilation)

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

    @TVMTest.given(shape=TVMTest.rand_shape())
    def test_relu(self, shape):
        X = torch.rand(shape)

        def relu(a):
            return F.relu(F.relu(a))

        ref_out, tvm_out = self.runBoth(relu, X)
        assert torch.allclose(ref_out, tvm_out, rtol=0.01, atol=0.01)

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
    def test_linear(self, shape, out_features):
        input = torch.rand(shape)
        weight = torch.rand(out_features, shape[1])
        bias = torch.rand(out_features)
        def linear(input, weight, bias):
            return F.linear(input, weight, bias) + 2.0

        ref_out, tvm_out = self.runBoth(linear, input, weight, bias)
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


if __name__ == "__main__":
    unittest.main()
