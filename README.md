# Pytorch TVM Extension

## Build

```
mkdir build && cd build
PYTHON_EXTENSION=`python-config --extension-suffix` PYTORCH_DIR=/path/to/pytorch/torch/ cmake ..
make
```

## Test

```
PYTHONPATH=build python test.py
```
