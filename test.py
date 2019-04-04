import time

import torch

@torch.jit.script
def foo(x, y, z):
  return x * y * z

size = 1000

x = torch.rand(size)
t = time.time()
for _ in range(1000):
  x = foo(x, x, x)
print("regular", time.time() - t)

import torch_tvm

@torch.jit.script
def foo(x, y, z):
  return x * y * z

x = torch.rand(size)
t = time.time()
for _ in range(1000):
  x = foo(x, x, x)
print("TVM", time.time() - t)

