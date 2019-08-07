#pragma once

#include <tvm/node/container.h>
#include <tvm/schedule.h>
#include <tvm/tensor.h>

namespace topi {
namespace generic {
tvm::Schedule schedule_custom_layer_norm(const tvm::Array<tvm::Tensor>& outs);
} // namespace generic
} // namespace topi
