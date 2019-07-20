import torch
import torch_tvm

torch_tvm.enable()

# The following function will be compiled with TVM
@torch.jit.script
def my_func(a):
    return torch.softmax(a, dim=0)

inputs = [torch.randn(10, 1)]
relay_graph = torch_tvm.to_relay(my_func, inputs)
print(relay_graph)
