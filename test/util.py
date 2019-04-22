import time
import numpy
from numbers import Number
import unittest

import torch
from torch.autograd.profiler import profile

import random

import torch_tvm

# base TVMTest class
class TVMTest(unittest.TestCase):
    precision = 1e-5

    def __init__(self, method_name="runTest"):
        super(TVMTest, self).__init__(method_name)

    @classmethod
    def rand_int(cls, min_=1, max_=4):
        def get():
            return random.randint(min_, max_)

        return get

    @classmethod
    def rand_list(cls, elem_fn, num_elem):
        def get():
            return [elem_fn() for _ in range(num_elem)]

        return get

    @classmethod
    def rand_shape(
        cls, min_rank=1, max_rank=4, min_dim=1, max_dim=2 ** 4, like=None, rank=None
    ):
        def get():
            rank_ = rank
            if not rank_:
                rank_ = cls.rand_int(min_rank, max_rank)()
            return cls.rand_list(cls.rand_int(min_dim, max_dim), rank_)()

        return get

    @classmethod
    def given(*args_, examples=2, **kwargs_):
        def f(fn):
            def f_impl(*args, **kwargs):
                for _ in range(examples):
                    for k in kwargs_:
                        kwargs[k] = kwargs_[k]()
                    try:
                        fn(*args, **kwargs)
                    except Exception as e:
                        print("Inputs:", kwargs)
                        raise (e)

            return f_impl

        return f

    def runBoth(self, func, *inputs, check_tvm=True):
        # jit the function
        trace_jit = torch.jit.trace(func, inputs)
        ref_out = trace_jit(*inputs)

        # jit the function and lower to TVM
        torch_tvm.enable()
        trace_tvm = torch.jit.trace(func, inputs)
        tvm_out = trace_tvm(*inputs)

        if check_tvm == True:
            tvm_unused = "TVM was not able to optimize this trace."
            assert "tvm::CompilationGroup" in str(
                trace_tvm.graph_for(*inputs)
            ), tvm_unused
            # tvm compile the graph and ensure TVM is used
            with profile() as p:
                _ = trace_tvm(*inputs)
            assert "TVM" in [_.name for _ in p.function_events], tvm_unused

        torch_tvm.disable()

        return ref_out, tvm_out
