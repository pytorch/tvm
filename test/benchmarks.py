from test.test_models import resnet18, resnext101_32x8d
from skimage import io
import torch
from torch.autograd.profiler import profile
import torch_tvm
import time
from tvm import autotvm
import sys
import os

def genImage():
    image = io.imread('test/cat.png')[:,:,:3].transpose(2,0,1)
    image = torch.unsqueeze(torch.Tensor(image), 0)
    return [image]

def benchmark(model, csv_file, input_fn=genImage, iters=100, warmup=10):
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
        with torch.autograd.profiler.profile() as prof:
          _ = trace_tvm(*inputs)
        tvm_profiled_time = 0
        total_profiled_time = 0
        for p in prof.key_averages():
          total_profiled_time += int(p.cpu_time)
          if p.key == "TVM":
            tvm_profiled_time += int(p.cpu_time)
        print("Done benchmarking TVM, which compiled {:.2f}% of compute".format(100 * tvm_profiled_time / total_profiled_time))
        if csv_file:
          exists = os.path.isfile(csv_file)
          with open(csv_file, 'a' if exists else 'w') as f:
            if not exists:
              f.write("timestamp,iter_per_sec\n")
            f.write("{},{}\n".format(int(time.time()), iters/tvm_time))
      print("JIT: {} iter/s\nTVM: {} iter/s".format(iters/jit_time, iters/tvm_time))

def run_benchmark(csv_file):
    model = resnet18(True)
    model.eval()
    benchmark(model, csv_file)

if __name__ == "__main__":
    csv_file = None
    if len(sys.argv) == 3 and sys.argv[1] == "--csv":
        csv_file = sys.argv[2]
    run_benchmark(csv_file)
