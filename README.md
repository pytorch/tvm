# Pytorch TVM Extension

## Build

You'll need to build PyTorch on top of this PR: https://github.com/pytorch/pytorch/pull/18846
```
cd pytorch
git fetch origin pull/18846/head:tvm_dev
git checkout tvm_dev
python setup.py install
```

You'll also need the facebookexperimental copy of TVM:
```
git clone https://github.com/facebookexperimental/tvm.git --recursive
cd tvm
mkdir build && cd build
# Be sure to build with -DINSTALL_DEV=ON
LLVM_DIR=/home/bwasti/llvm_build/install/lib/cmake/llvm cmake .. -DCMAKE_INSTALL_PREFIX:PATH=. -DINSTALL_DEV=ON -DUSE_LLVM=/home/bwasti/llvm_build/install/bin/llvm-config -GNinja
ninja install
```

Then, you'll need to build this repo seperately
```
cd pytorch_tvm
mkdir build && cd build
PYTORCH_DIR=/data/users/bwasti/pytorch/torch TVM_DIR=/home/bwasti/local/tvm/build cmake .. -DPYTHON_EXECUTABLE=$(which python) -GNinja && ninja
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
- Zero copy `set_input`.
- Threadpool
- Allocator
- Operator translation
  - [x] Add
  - [x] Multiply
  - [ ] Convolution
  - [ ] BatchNorm
  - [ ] Relu
  - [ ] AveragePool
  - [ ] MaxPool
  - [ ] Linear
- Tensor manipulation
  - [ ] Reshape
  - [ ] Views

