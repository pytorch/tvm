import unittest
from test.util import TVMTest
import torch_tvm
import torch
import tvm

class IntegrationTest(TVMTest):
    def test_basic(self):
        def mm():
          # Algorithm
          K = 128
          M = 128
          N = 128
          k = tvm.reduce_axis((0, K), 'k')
          A = tvm.placeholder((M, K), name='A')
          B = tvm.placeholder((K, N), name='B')
          C = tvm.compute(
              (M, N),
              lambda x, y: tvm.sum(A[x, k] * B[k, y], axis=k),
              name='C')
          return C

        torch_tvm.register_operator("mm", mm, None)
        print(torch.ops.tvm.mm)

        torch_tvm.enable()
        @torch.jit.script
        def test(a, b, c):
          return torch.ops.tvm.mm(a + c, b)
        
        X = torch.randn(128,128)
        print(test.graph)
        print(test.graph_for(X,X,X))
        o = test(X, X, X)

if __name__ == "__main__":
    unittest.main()

