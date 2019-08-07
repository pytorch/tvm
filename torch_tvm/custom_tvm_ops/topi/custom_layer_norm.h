#pragma once

#include <tvm/attrs.h>
#include <tvm/relay/expr.h>

namespace topi {
Tensor custom_layer_norm(
    const Tensor& data,
    const Tensor& gamma,
    const Tensor& beta,
    const int num_axis_to_normalize,
    const bool affine,
    const float eps);
} // namespace topi
