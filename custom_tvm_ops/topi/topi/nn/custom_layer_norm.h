#pragma once

#include <string>

#include <tvm/relay/expr.h>
#include <tvm/attrs.h>

#include "topi/tags.h"
#include "topi/reduction.h"

namespace topi {
using namespace tvm;

inline Array<Tensor> calculate_mean_and_variance(const Tensor& data,
    const Array<Integer>& normalized_axis) {
  auto ndim = data->shape.size();
  auto real_axis = GetRealAxis(static_cast<int>(ndim), normalized_axis);
  auto reduce_axes = MakeReduceAxes(real_axis, data);
  // Gets the shape of result after reduction.
  auto target_shape = MakeReduceTargetShape(real_axis, data, false, false);

  auto fidentity = [](std::vector<Type> types) {
    Array<Expr> result;
    CHECK(types.size() == 2);
    CHECK(types[0] == types[1]);
    result.push_back(tvm::make_const(types[0], 0)); // Mean sum
    result.push_back(tvm::make_const(types[0], 0)); // Variance sum
    return result;
  };
  auto fcombine = [](Array<Var> lhs, Array<Var> rhs) {
    Array<Expr> result;
    result.push_back(lhs[0] + rhs[0]); // mean
    result.push_back(lhs[1] + rhs[1] * rhs[1]); // variance
    return result;
  };
  auto reducer = MakeCommReducer(fcombine, fidentity, "mean_variance_sum");
  auto compute = [ndim, &real_axis, &reduce_axes, &reducer, &data]
    (const Array<Var>& indices) {
      Array<Expr> eval_range;
      int arg_counter = 0;
      int red_counter = 0;

      // eval_range takes index value from indices for thenon reduction axis.
      // And for the reduction axis adds reduce_axes which is a Range axis.
      // Thus for some 2 dim tensor [5, 5] with dim 1 to reduce, with dim 0
      // index of 3 eval_range would (3, ReduceAxis(0, 4))
      for (size_t i = 0; i < ndim; ++i) {
        if (std::find(real_axis.begin(), real_axis.end(), i) != real_axis.end()) {
          // real_axis contains i
          eval_range.push_back(reduce_axes[red_counter]);
          red_counter++;
        } else {
          eval_range.push_back(indices[i]);
        }
      }
      return reducer({ data(eval_range), data(eval_range) }, reduce_axes, nullptr);
    };

  tvm::relay::IndexExpr num_elements = make_const(data->dtype, 1);
  for (int64_t i = 0; i < normalized_axis.size(); ++i) {
    num_elements *= data->shape[normalized_axis[i]->value];
  }
  auto reduce_outputs = tvm::compute(target_shape, compute, data->op->name + "mean_var_sum", kCommReduce);
  return {reduce_outputs[0]/num_elements, reduce_outputs[1]/num_elements};
}

inline Tensor custom_layer_norm_impl(const Tensor& data,
  const Array<Integer>& normalized_axis) {
  auto ndim = data->shape.size();
  auto normalized_axis_dim = normalized_axis.size();
  CHECK(ndim > normalized_axis_dim);
  auto mean_variance = calculate_mean_and_variance(data, normalized_axis);
  auto layer_norm_compute = [&mean_variance, &data, ndim, normalized_axis_dim]
    (const Array<Var>& indices) {
      Array<Var> mean_variance_indices(indices.begin(), indices.begin() + (ndim - normalized_axis_dim));
      auto mean = mean_variance[0];
      auto variance = mean_variance[1];
      auto epsilon = tvm::make_const(Float(32), 1e-9);
      auto var_0 = tvm::max((variance(mean_variance_indices) -
            mean(mean_variance_indices) * mean(mean_variance_indices)), epsilon);
      Expr one = make_const(data->dtype, 1);
      auto var_rsqrt = one/tvm::sqrt(var_0);
      return (data(indices) * var_rsqrt - var_rsqrt * mean(mean_variance_indices));
    };
  return tvm::compute(data->shape, layer_norm_compute, data->op->name, data->op->name);
}

inline Tensor custom_layer_norm_impl_affine(const Tensor& data, const Tensor& gamma,
  const Tensor& beta, const Array<Integer>& normalized_axis) {
  auto ndim = data->shape.size();
  auto normalized_axis_dim = normalized_axis.size();
  CHECK(ndim > normalized_axis_dim);
  auto mean_variance = calculate_mean_and_variance(data, normalized_axis);
  auto layer_norm_compute = [&mean_variance, &data, ndim, normalized_axis_dim, &gamma, &beta]
    (const Array<Var>& indices) {
      Array<Var> mean_variance_indices(indices.begin(), indices.begin() + (ndim - normalized_axis_dim));
      Array<Var> affine_indices(indices.begin() + (ndim - normalized_axis_dim), indices.end());
      auto mean = mean_variance[0];
      auto variance = mean_variance[1];
      auto epsilon = tvm::make_const(Float(32), 1e-9);
      auto var_0 = tvm::max((variance(mean_variance_indices) -
            mean(mean_variance_indices) * mean(mean_variance_indices)), epsilon);
      Expr one = make_const(data->dtype, 1);
      auto var_rsqrt = one/tvm::sqrt(var_0);
      return ((data(indices) * var_rsqrt - var_rsqrt *
            mean(mean_variance_indices)) * gamma(affine_indices) + beta(affine_indices));
    };
  return tvm::compute(data->shape, layer_norm_compute, data->op->name, data->op->name);
}

inline Tensor custom_layer_norm(const Tensor& data, const Tensor& gamma,
    const Tensor& beta, const Array<Integer>& normalized_axis,
    const bool affine) {
  if (affine) {
    return custom_layer_norm_impl_affine(data, gamma, beta, normalized_axis);
  }
  else {
    return custom_layer_norm_impl(data, normalized_axis);
  }
}

} // namespace topi
