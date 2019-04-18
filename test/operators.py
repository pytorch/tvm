import unittest
from util import TestCase

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

        self.checkTraceTVM(conv, input_shapes=[[20,16,10,10], [10,16,3,3], [20,10,8,8]], verbose=True)

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

    def test_relu(self):
        def relu(a):
            return F.relu(F.relu(a))

        self.checkTraceTVM(relu, input_shapes=[(100,)], verbose=True)


if __name__ == '__main__':
    unittest.main()
