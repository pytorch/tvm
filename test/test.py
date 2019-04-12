import time

import torch

@torch.jit.script
def foo(a, b, c):
  return a * b + c

size = 100000
runs = 1000
print("{} runs of size {}".format(runs, size))
seed = torch.rand(size) / runs

x = seed
t = time.time()
for _ in range(runs):
  x = foo(x, x, x)
print("Default run \t\t{:.4f}s".format(time.time() - t))

import torch_tvm

@torch.jit.script
def foo(a, b, c):
  return a * b + c

y = seed

# precompile the foo on first run with inputs
t = time.time()
_ = foo(y, y, y)
print("TVM compilation \t{:.4f}s".format(time.time() - t))

t = time.time()
for _ in range(runs):
  y = foo(y, y, y)
print("TVM run \t\t{:.4f}s".format(time.time() - t))

print()
print("Sample points:")
print(x[:10])
print(y[:10])
rtol = 0.01
assert torch.allclose(y, x, rtol=rtol)
print("Passed correctness check (rtol = {}).".format(rtol))
