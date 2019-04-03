import torch, torch_tvm

@torch.jit.script
def foo(x, y):
  return x + y

if __name__ == "__main__":
  print(foo(torch.rand(3), torch.rand(3)))
