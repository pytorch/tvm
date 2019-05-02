from test.test_models import resnet18, resnet101
from skimage import io
import torch
from torch.autograd.profiler import profile
import torch_tvm
import time
from tvm import autotvm

def genImage():
    image = io.imread('test/cat.png')[:,:,:3].transpose(2,0,1)
    image = torch.unsqueeze(torch.Tensor(image), 0)
    return [image]

def benchmark(model, input_fn=genImage, iters=100, warmup=10):
    with torch.no_grad():
      inputs = input_fn()
      print("Tracing model with JIT")
      trace_jit = torch.jit.trace(model, inputs)
      print("Warming JIT up with {} runs".format(warmup))
      for _ in range(warmup):
        _ = trace_jit(*inputs)

      print("Running JIT {} times".format(iters))
      start = time.time()
      for _ in range(iters):
        _ = trace_jit(*inputs)
      jit_time = time.time() - start
      print("Done benchmarking JIT")

      with autotvm.apply_history_best("test/autotvm_tuning.log"):
        torch_tvm.enable()
        print("Tracing model with TVM")
        trace_tvm = torch.jit.trace(model, inputs)
        print("Warming TVM up with {} iters".format(warmup))
        for _ in range(warmup):
          _ = trace_tvm(*inputs)

        print("Running TVM {} times".format(iters))
        start = time.time()
        for _ in range(iters):
          _ = trace_tvm(*inputs)
        tvm_time = time.time() - start
        print("Done benchmarking TVM")
      print("JIT: {} iter/s\nTVM: {} iter/s".format(iters/jit_time, iters/tvm_time))

def run_benchmark():
    model = resnet18(True)
    model.eval()
    benchmark(model)

if __name__ == "__main__":
    run_benchmark()
