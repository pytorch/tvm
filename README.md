# Pytorch TVM Extension
[![CircleCI](https://circleci.com/gh/pytorch/tvm.svg?style=svg)](https://circleci.com/gh/pytorch/tvm)
<img align="right" width="400" src="http://ec2-3-14-143-1.us-east-2.compute.amazonaws.com/benchmarks.png?">

Please note that this is a work in progress.


## Build

For improved performance, you'll need to build PyTorch on top of this PR: https://github.com/pytorch/pytorch/pull/20284
```
cd pytorch
git fetch origin pull/20284/head:tvm_dev
git checkout tvm_dev
python setup.py install
```

Otherwise, install the latest Nightly build of PyTorch.

Then, build this repo
```
# Make sure the right llvm-config is in your PATH
python setup.py install
```

## Test

```
python setup.py test 
```

## Usage

This package transparently hooks into PyTorch's JIT, so the same tooling is applicable (see `@torch.jit.script`, `torch.jit.trace` and `graph_for`).  See below for an example.

```
import torch_tvm

torch_tvm.enable()

# The following function will be compiled with TVM
@torch.jit.script
def my_func(a, b, c):
    return a * b + c
```

To disable the JIT hooks, use `torch_tvm.disable()`.

## Code Layout

- `register.cpp`: Sets up pybind bindings and invokes the registration of a TVM backend.
- `compiler.{h,cpp}`: Main logic to compile a PyTorch JIT graph with TVM.
- `operators.{h,cpp}`: Location of mapping from JIT IR to TVM operators.

![TVM Integration](https://github.com/pytorch/tvm/blob/master/pt_execution.png?raw=true)

## v0.1 Roadmap

Below, in order, is a prioritized list of tasks for this repository.

- [x] End to end build and runtime
- [ ] Operator translation
  - [x] Add
  - [x] Multiply
  - [x] Convolution
  - [x] BatchNorm
  - [x] Relu
  - [x] AveragePool
  - [x] MaxPool
  - [x] Linear
  - [x] Reshape
  - [ ] AdaptiveAveragePool
  - [ ] Views
- [x] Tooling
  - [x] Model coverage checks
  - [x] Benchmarks for master
- [x] User exposed configurations
  - [x] Backend selection (CPU/Cuda/OpenCL)
  - [x] Optimization level
- [ ] Custom TVM operator registration
  - Enable Python/C++ mechanism to use custom TVM operators and schedules
- [ ] Bail-out mechanism
  - When TVM cannot compile a subgraph, invoke PyTorch JIT fallback
- [ ] Zero copy `set_input`
- [ ] Subsystem integration
  - [ ] Threadpool integration
  - [ ] Allocator integration
  - [ ] Distributed communication
- [ ] Advanced IR integration
  - [ ] Control flow
  - [ ] Aliasing

