# Pytorch TVM Extension

## Build

You'll need to build PyTorch on top of this PR: https://github.com/pytorch/pytorch/pull/18846
```
cd pytorch
git fetch origin pull/18846/head:tvm_dev
git checkout tvm_dev
python setup.py install
```

Then, you'll need to build this repo seperately
```
cd pytorch_tvm
mkdir build && cd build
USE_LLVM=/home/bwasti/llvm_build/install/bin/llvm-config LLVM_DIR=/home/bwasti/llvm_build/install/lib/cmake/llvm PYTHON_EXTENSION=`python-config --extension-suffix` PYTORCH_DIR=/data/users/bwasti/pytorch/torch/ cmake .. -GNinja
PYTORCH_DIR=/data/users/bwasti/pytorch/torch/ LLVM_DIR=/home/bwasti/llvm_build/install/lib/cmake/llvm ninja
cd ..
```

## Test

```
PYTHONPATH=build python test.py
```

## Code Layout

- `register.cpp`: Sets up pybind bindings and invokes the registration of a TVM backend.
- `compiler.{h,cpp}`: Main logic to compile a PyTorch JIT graph with TVM.
- `operators.{h,cpp}`: Location of mapping from JIT IR to TVM operators.


## TODO

- Add ability to register translations from opaque op names to TVM (as is done in `operator.cpp`) from Python.
- Zero copy `set_input`
