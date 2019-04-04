import time

import torch

@torch.jit.script
def foo(a, b, c):
  return a * b * c

size = 100000
runs = 100
seed = torch.rand(size) * 10

x = seed
t = time.time()
for _ in range(runs):
  x = foo(x, x, x)
print("Default", time.time() - t)

import torch_tvm

@torch.jit.script
def foo(a, b, c):
  return a * b * c

y = seed

# precompile the foo on first run with inputs
_ = foo(y, y, y)

t = time.time()
for _ in range(runs):
  y = foo(y, y, y)
print("TVM", time.time() - t)

assert torch.allclose(y, x)
