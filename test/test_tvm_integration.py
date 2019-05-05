import unittest
from test.util import TVMTest
import torch_tvm
import torch
import tvm
from tvm import relay

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
              lambda x, y: tvm.sum(A[x, k].astype("float32") * B[k, y].astype("float32"), axis=k),
              name='C')
          return C
        # dense
        #@reg.register_compute("nn.dense")
        def compute_dense(attrs, inputs, out_type, target):
            """Compute definition of dense"""
            out_dtype = attrs.out_dtype
            out_dtype = inputs[0].dtype if out_dtype == "" else out_dtype
            return [topi.nn.dense(inputs[0], inputs[1], out_dtype=out_dtype)]


        #@reg.register_schedule("nn.dense")
        def schedule_dense(attrs, outputs, target):
            """Schedule definition of dense"""
            with target:
                return topi.generic.schedule_dense(outputs)
        relay.op.register("torch.mm", "FTVMCompute", value=mm)
        #relay.op.register("torch.mm", "FTVMSchedule", value=schedule_dense)
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

