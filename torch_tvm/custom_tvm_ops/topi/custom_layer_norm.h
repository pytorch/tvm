#pragma once

#include <tvm/relay/expr.h>
#include <tvm/attrs.h>

namespace topi {
Tensor custom_layer_norm(const Tensor& data, const Tensor& gamma,
    const Tensor& beta, const int num_axis_to_normalize,
    const bool affine, const float eps);
} // topi
