#include <tvm/runtime/device_api.h>

#include "memory_utils.h"

namespace torch_tvm {
namespace utils {

bool isAligned(void* data_ptr, std::uintptr_t alignment_in_bytes) {
  auto mask = alignment_in_bytes - 1;
  TORCH_CHECK((alignment_in_bytes & mask) == 0);
  return (reinterpret_cast<std::uintptr_t>(data_ptr) & mask) == 0;
}

DLManagedTensor* allocAndCopyData(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.device().is_cpu());
  DLManagedTensor* dl_managed_tensor = new DLManagedTensor();
  auto contig_tensor = tensor;
  if (!tensor.is_contiguous()) {
    auto contig_tensor = tensor.contiguous();
  }
  // managed_tensor_deleter is supplied to unique_ptr as a deleter
  // of this managed memory. Thus setting deleter to nullptr;
  dl_managed_tensor->deleter = nullptr;
  dl_managed_tensor->manager_ctx = dl_managed_tensor;
  auto& dl_tensor = dl_managed_tensor->dl_tensor;

  auto num_dims = contig_tensor.dim();
  dl_tensor.ndim = num_dims;
  dl_tensor.dtype = at::getDLDataType(contig_tensor);
  int64_t device_id = 0;
  dl_tensor.ctx = getDLContext(contig_tensor, device_id);
  dl_tensor.shape = dl_tensor.strides = nullptr;
  dl_tensor.data = nullptr;
  dl_tensor.shape = new int64_t[num_dims];
  dl_tensor.strides = new int64_t[num_dims];
  TORCH_CHECK(dl_tensor.shape != nullptr && dl_tensor.strides != nullptr,
      "Memory allocation failed for DLTensor shape and strides"
      "by ManagedTensors.");

  auto tensor_sizes = contig_tensor.sizes();
  auto tensor_strides = contig_tensor.strides();
  for (int64_t i = 0; i < num_dims; ++i) {
    dl_tensor.shape[i] = tensor_sizes[i];
    dl_tensor.strides[i] = tensor_strides[i];
  }

  // make sure the allocated size is a multiple of alignment
  auto nbytes_alloc = contig_tensor.nbytes();
  auto rem = nbytes_alloc % tvm::runtime::kAllocAlignment;
  if (rem > 0) {
    nbytes_alloc += tvm::runtime::kAllocAlignment - rem;
  }
  dl_tensor.data = aligned_alloc(tvm::runtime::kAllocAlignment, nbytes_alloc);
  TORCH_CHECK(dl_tensor.data != nullptr,
      "Memory allocation failed for DLTensor data by ManagedTensors.");

  std::memcpy(dl_tensor.data, contig_tensor.data_ptr(), contig_tensor.nbytes());
  dl_tensor.byte_offset = 0;

  return dl_managed_tensor;
}

} // utils
} // torch_tvm
