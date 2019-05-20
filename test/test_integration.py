import unittest
from test.util import TVMTest
import torch_tvm
import torch
import tvm
from tvm import relay

class IntegrationTest(TVMTest):
    def test_basic(self):
        torch_tvm.register_operator("woo", "nn.relu")

        torch_tvm.enable()

        @torch.jit.script
        def test(a, b):
          d = torch.ops.tvm.woo(a)
          e = torch.ops.tvm.woo(b)
          return d + e
        
        X = torch.randn(128,128)
        print(test.graph)
        print(test.graph_for(X,X))
        o = test(X, X)

        def test(a, b):
          d = torch.ops.tvm.woo(a)
          e = torch.ops.tvm.woo(b)
          return d + e
        o = test(X, X)

if __name__ == "__main__":
    unittest.main()
