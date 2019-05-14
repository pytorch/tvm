from test.util import TVMTest
import torch
import torch_tvm

class TestCore(TVMTest):
    @TVMTest.given(shape=TVMTest.rand_shape(rank=1))
    def test_core(self, shape):
        x = torch.rand(shape)
        y = torch.rand(shape)
        z = torch.rand(shape)

        def add(a, b, c):
            return a + b + c

        inputs = [x,y,z]

        trace_jit = torch.jit.trace(add, inputs)
        jit_out = trace_jit(*inputs)

        torch_tvm.enable()
        trace_tvm = torch.jit.trace(add, inputs)
        tvm_out = trace_tvm(*inputs)
        torch_tvm.disable()
        torch.testing.assert_allclose(jit_out, tvm_out, rtol=0.01, atol=0.01)

        torch_tvm.enable(opt_level=1)
        trace_tvm = torch.jit.trace(add, inputs)
        tvm_out = trace_tvm(*inputs)
        torch_tvm.disable()
        torch.testing.assert_allclose(jit_out, tvm_out, rtol=0.01, atol=0.01)

        torch_tvm.enable(opt_level=3)
        trace_tvm = torch.jit.trace(add, inputs)
        tvm_out = trace_tvm(*inputs)
        torch_tvm.disable()
        torch.testing.assert_allclose(jit_out, tvm_out, rtol=0.01, atol=0.01)

        torch_tvm.enable(device_type="cpu", device="llvm", host="llvm")
        trace_tvm = torch.jit.trace(add, inputs)
        tvm_out = trace_tvm(*inputs)
        torch_tvm.disable()
        torch.testing.assert_allclose(jit_out, tvm_out, rtol=0.01, atol=0.01)
