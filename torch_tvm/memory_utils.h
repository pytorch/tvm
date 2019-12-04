#pragma  once

#include <ATen/DLConvertor.h>
#include <ATen/Tensor.h>
#include <torch/csrc/jit/ir.h>

#include <dlpack/dlpack.h>

#include <memory>

namespace torch_tvm {
namespace utils {

struct DLManagedTensorDeleter {
  void operator()(DLManagedTensor* manager_ctx) {
    if (manager_ctx == nullptr) {
      return;
    }

    auto dl_tensor = manager_ctx->dl_tensor;
    TORCH_CHECK(dl_tensor.ctx.device_type == kDLCPU);
    if (dl_tensor.data) {
      TORCH_CHECK((dl_tensor.shape && dl_tensor.strides), "If DLTensor's data"
          " pointer is valid then shape and strides must be as well.")
      std::free(dl_tensor.data);
      delete[] dl_tensor.shape;
      delete[] dl_tensor.strides;
    }
    delete manager_ctx;
  }
};

bool isAligned(void* data_ptr, std::uintptr_t alignment_in_bytes);

DLManagedTensor* allocAndCopyData(const at::Tensor& tensor);
using DLManagedTensorPtr = std::unique_ptr<DLManagedTensor,
      DLManagedTensorDeleter>;

} // utils
} // torch_tvm
