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
        o1 = test(X, X)

        torch_tvm.disable()

        # Ensure eager mode works
        def test(a, b):
          d = torch.ops.tvm.woo(a)
          e = torch.ops.tvm.woo(b)
          return d + e

        o2 = test(X, X)
        assert torch.allclose(o1, o2, rtol=0.01, atol=0.01)

        # Ensure JIT and TVM JIT both work
        ref_out, tvm_out = self.runBoth(test, X, X)
        assert torch.allclose(ref_out, tvm_out, rtol=0.01, atol=0.01)


if __name__ == "__main__":
    unittest.main()
