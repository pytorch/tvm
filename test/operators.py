import unittest
from util import TestCase

import torch
import torch.nn.functional as F
import torch

# test jit tvm operators
class TestOperators(TestCase):

    def test_add(self):
        def add(a, b, c):
          return a + b + c

        self.checkTraceTVM(add, verbose=True)

    def test_mul(self):
        def mul(a, b, c):
            return a * b * c

        self.checkTraceTVM(mul, verbose=True)

    def test_conv(self):
        def conv(a, b, c):
            return F.conv2d(a, b) + c

        def conv_stride_padding_dilation(a, b, c):
            return F.conv2d(a, b, stride=(2, 1), padding=(4, 2), dilation=(3,1)) + c

        self.checkTraceTVM(conv, input_shapes=[[20,16,10,10], [10,16,3,3], [20,10,8,8]], verbose=True)
        self.checkTraceTVM(conv_stride_padding_dilation, input_shapes=[[20,16,50,3], [33,16,3,5], [20,33,26,3]], verbose=True)


    def test_batch_norm(self):
        def batch_norm(a, b, c, d):
            return F.batch_norm(a + d, b, c)

        a = torch.rand(10,10)
        b = torch.rand(10)
        c = torch.rand(10)
        d = torch.rand(10,10)

        self.checkTraceTVM(batch_norm,
          input_tensors=[a,b,c,d], verbose=True)

        def batch_norm_weighted(a, b, c, d, weight, bias):
            return F.batch_norm(a + d, b, c, weight=weight, bias=bias)

        self.checkTraceTVM(batch_norm_weighted,
          input_tensors=[a,b,c,d,c,b], verbose=True)


if __name__ == '__main__':
    unittest.main()
