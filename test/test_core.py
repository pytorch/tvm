import unittest
from test.util import TVMTest
import torch
import torch_tvm


class TestCore(TVMTest):
    def test_get_handle(self):
        shape = 8
        x = torch.rand(shape)
        y = torch.rand(shape)
        z = torch.rand(shape)

        def add(a, b, c):
            return a + b + c

        @torch.jit.script
        def mul(a, b, c):
            return a * b * c

        inputs = [x, y, z]

        torch_tvm.enable()

        trace_tvm = torch.jit.trace(add, inputs)

        relay_graph = torch_tvm.to_relay(trace_tvm, inputs)
        relay_graph = torch_tvm.to_relay(add, inputs)
        relay_graph = torch_tvm.to_relay(mul, inputs)

        torch_tvm.disable()

    @TVMTest.given(shape=TVMTest.rand_shape(rank=1))
    @unittest.skip("causing segfaults, need to fix operator registration before enable it")
    def test_registry(self, shape):
        x = torch.rand(shape)
        y0 = torch.ops.tvm.relu(x)
        y1 = torch.relu(x)

        torch.testing.assert_allclose(y0, y1)

    @TVMTest.given(shape=TVMTest.rand_shape(rank=1))
    def test_core(self, shape):
        x = torch.rand(shape)
        y = torch.rand(shape)
        z = torch.rand(shape)

        def add(a, b, c):
            return a + b + c

        inputs = [x, y, z]

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

    @TVMTest.given(
        shape=TVMTest.rand_shape(rank=4, min_dim=4),
        examples=1
    )
    def test_fall_back(self, shape):
        inputs = torch.rand(shape)

        def add(input):
            return torch.add(input, 1, 2)

        jit_script_reshape = torch.jit.script(add)
        jit_out = jit_script_reshape(inputs)

        with self.assertRaises(RuntimeError):
            tvm_strict_script_reshape = torch.jit.script(add)
            torch_tvm.enable(strict=True)
            tvm_out = tvm_strict_script_reshape(inputs)
            torch_tvm.disable()

        torch_tvm.enable(strict=False)
        tvm_script_reshape = torch.jit.script(add)
        tvm_out = tvm_script_reshape(inputs)
        torch_tvm.disable()

        torch.testing.assert_allclose(jit_out, tvm_out, rtol=0.01, atol=0.01)

    @TVMTest.given(
        shape=TVMTest.rand_shape(rank=4, min_dim=4),
        examples=1
    )
    def test_dropout_removal(self, shape):
        input_a = torch.rand(shape)
        input_b = torch.rand(shape)
        input_c = torch.rand(shape)

        def dropout_training(a, b, c):
            t = a + b
            s = torch.dropout(t, 0.1, True)
            return s + c

        def dropout_inference(a, b, c):
            t = a + b
            s = torch.dropout(t, 0.1, False)
            return s + c

        torch_tvm.enable()
        tvm_graph_training = torch.jit.trace(dropout_training, \
                (input_a, input_b, input_c))
        tvm_graph_inference = torch.jit.trace(dropout_inference, \
                (input_a, input_b, input_c))
        torch_tvm.disable()
        assert "aten::dropout" in \
                str(tvm_graph_training.graph_for(input_a, input_b, input_c)), \
                "dropout must not be removed during training."
        assert "aten::dropout" not in \
                str(tvm_graph_inference.graph_for(input_a, input_b, input_c)), \
                "dropout must be removed during inference."

if __name__ == "__main__":
    unittest.main()
