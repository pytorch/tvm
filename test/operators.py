import unittest
from util import TestCase

import torch.nn.functional as F

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

if __name__ == '__main__':
    unittest.main()
