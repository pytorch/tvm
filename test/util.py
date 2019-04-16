import time
import numpy
from numbers import Number
import unittest

import torch
from torch.autograd.profiler import profile

from tvm import relay # This registers all the schedules
import torch_tvm

# base TestCase class
class TestCase(unittest.TestCase):
    precision = 1e-5

    def __init__(self, method_name='runTest'):
        super(TestCase, self).__init__(method_name)

    def __assertTensorsEqual(self, a, b, prec=None, message="", allow_inf=False):
        super(TestCase, self).assertEqual(a.size(), b.size(), message)
        if a.numel() > 0:
            if a.device.type == 'cpu' and a.dtype == torch.float16:
                # CPU half tensors don't have the methods we need below
                a = a.to(torch.float32)
            b = b.to(a)

            if (a.dtype == torch.bool) != (b.dtype == torch.bool):
                raise TypeError("Was expecting both tensors to be bool type.")
            else:
                if a.dtype == torch.bool and b.dtype == torch.bool:
                    # we want to respect precision but as bool doesn't support substraction,
                    # boolean tensor has to be converted to int
                    a = a.to(torch.int)
                    b = b.to(torch.int)

                diff = a - b
                if a.is_floating_point():
                    # check that NaNs are in the same locations
                    nan_mask = torch.isnan(a)
                    self.assertTrue(torch.equal(nan_mask, torch.isnan(b)), message)
                    diff[nan_mask] = 0
                    # inf check if allow_inf=True
                    if allow_inf:
                        inf_mask = torch.isinf(a)
                        inf_sign = inf_mask.sign()
                        self.assertTrue(torch.equal(inf_sign, torch.isinf(b).sign()), message)
                        diff[inf_mask] = 0
                # TODO: implement abs on CharTensor (int8)
                if diff.is_signed() and diff.dtype != torch.int8:
                    diff = diff.abs()
                max_err = diff.max()
                self.assertLessEqual(max_err, prec, message)

    def assertEqual(self, x, y, prec=None, message="", allow_inf=False):
        if isinstance(prec, str) and message == '':
            message = prec
            prec = None
        if prec is None:
            prec = self.precision

        if isinstance(x, torch.Tensor) and isinstance(y, Number):
            self.assertEqual(x.item(), y, prec, message, allow_inf)
        elif isinstance(y, torch.Tensor) and isinstance(x, Number):
            self.assertEqual(x, y.item(), prec, message, allow_inf)
        elif isinstance(x, torch.Tensor) and isinstance(y, numpy.bool_):
            self.assertEqual(x.item(), y, prec, message, allow_inf)
        elif isinstance(y, torch.Tensor) and isinstance(x, numpy.bool_):
            self.assertEqual(x, y.item(), prec, message, allow_inf)
        elif isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            self.__assertTensorsEqual(x, y, prec, message, allow_inf)
        elif isinstance(x, dict) and isinstance(y, dict):
            if isinstance(x, OrderedDict) and isinstance(y, OrderedDict):
                self.assertEqual(x.items(), y.items())
            else:
                self.assertEqual(set(x.keys()), set(y.keys()))
                key_list = list(x.keys())
                self.assertEqual([x[k] for k in key_list], [y[k] for k in key_list])
        elif is_iterable(x) and is_iterable(y):
            super(TestCase, self).assertEqual(len(x), len(y), message)
            for x_, y_ in zip(x, y):
                self.assertEqual(x_, y_, prec, message)
        elif isinstance(x, Number) and isinstance(y, Number):
            if abs(x) == inf or abs(y) == inf:
                if allow_inf:
                    super(TestCase, self).assertEqual(x, y, message)
                else:
                    self.fail("Expected finite numeric values - x={}, y={}".format(x, y))
                return
            super(TestCase, self).assertLessEqual(abs(x - y), prec, message)
        else:
            super(TestCase, self).assertEqual(x, y, message)

    def checkTraceTVM(self, func, input_tensors=None, input_shapes=None, size=100000, runs=100, verbose=False):
        # prepare inputs
        if input_tensors is None:
            if input_shapes is None:
                seed = torch.rand(size) / runs / 2
                input_tensors = (seed, seed, seed)
            else:
                input_tensors = []
                for shape in input_shapes:
                    seed = torch.rand(*shape) / runs / 2
                    input_tensors.append(seed)

        # jit the function
        trace_jit = torch.jit.trace(func, input_tensors)
        # specialize the graph with the inputs
        _ = trace_jit(*input_tensors)
        # timeit the perf
        jit_start = time.time()
        for _ in range(runs):
          outputs_jit = trace_jit(*input_tensors)
        jit_time = time.time() - jit_start

        # jit the function and lower to TVM
        torch_tvm.enable()
        trace_tvm = torch.jit.trace(func, input_tensors)
        tvm_unused = "TVM was not able to optimize this trace."
        assert "tvm::CompilationGroup" in str(trace_tvm.graph_for(*input_tensors)), tvm_unused
        # tvm compile the graph and ensure TVM is used
        with profile() as p:
            _ = trace_tvm(*input_tensors)
        assert "TVM" in [_.name for _ in p.function_events], tvm_unused
        torch_tvm.disable()
        # timeit the perf
        tvm_start = time.time()
        for _ in range(runs):
          outputs_tvm = trace_tvm(*input_tensors)
        tvm_time = time.time() - tvm_start

        if verbose:
            print("\noperator " + func.__name__ + ":\t{} runs of size {}".format(runs, size)
                  + " \tjit time:{:.4f}s".format(jit_time)
                  + "\ttvm time:{:.4f}s".format(tvm_time))
        self.assertEqual(outputs_jit, outputs_tvm)
