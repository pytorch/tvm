# Pytorch TVM Extension

## Build

```
mkdir build && cd build
USE_LLVM=/home/bwasti/llvm_build/install/bin/llvm-config LLVM_DIR=/home/bwasti/llvm_build/install/lib/cmake/llvm PYTHON_EXTENSION=`python-config --extension-suffix` PYTORCH_DIR=/data/users/bwasti/pytorch/torch/ cmake .. -GNinja
PYTORCH_DIR=/data/users/bwasti/pytorch/torch/ LLVM_DIR=/home/bwasti/llvm_build/install/lib/cmake/llvm ninja
cd ..
```

## Test

```
PYTHONPATH=build python test.py
```
