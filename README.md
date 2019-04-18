# Pytorch TVM Extension

## Build

You'll need to build PyTorch on top of this PR: https://github.com/pytorch/pytorch/pull/18588
```
cd pytorch
git fetch origin pull/18588/head:tvm_dev
git checkout tvm_dev
python setup.py install
```

Then, you'll need to build this repo seperately
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
from tvm import relay # This imports all the topi operators
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

## TODO

- Add ability to register translations from opaque op names to TVM (as is done in `operator.cpp`) from Python.
- Zero copy `set_input`
- Bail-out mechanism (invoke PyTorch JIT fallback)
- Threadpool integration
- Allocator integration
- Operator translation
  - [x] Add
  - [x] Multiply
  - [x] Convolution
  - [x] BatchNorm
  - [x] Relu
  - [x] AveragePool
  - [ ] MaxPool
  - [ ] Linear
- Tensor manipulation
  - [ ] Reshape
  - [ ] Views

