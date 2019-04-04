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
