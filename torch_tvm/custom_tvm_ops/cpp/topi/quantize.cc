#include <string>
#include <limits>

#include <tvm/attrs.h>
#include <tvm/relay/expr.h>

#include "topi/reduction.h"
#include "topi/tags.h"

#include "quantize.h"

namespace topi {
using namespace tvm;

Array<Tensor> data_int8_quantize(
    const Tensor& data,
    const Tensor& zero_point,
    const Tensor& scale,
    bool is_signed,
    int precision) {
  auto q_min = is_signed ? -(1 << (precision - 1)) : 0;
  auto q_max = is_signed ? ((1 << (precision - 1)) - 1) : (1 << precision) - 1;
  auto target_type = is_signed ? Int(8) : UInt(8);
  auto inverse_scale = 1 /scale(0);

  auto clamp_output = tvm::compute(
      data->shape,
      [&](Var i, Var j) {
         return tvm::cast(target_type, tvm::nearbyint(
            tvm::min(
               tvm::max(tvm::cast(Float(32), zero_point(0)) + data(i, j)*inverse_scale, q_min),
               q_max
            )
         ));
      },
      "tensor",
      "int8_quantize_data"
      );

  return {clamp_output};
}

Array<Tensor> data_int8_row_offset(const Tensor& quantized_data) {

  auto k = tvm::reduce_axis(Range(0, quantized_data->shape[1]), "k");
  auto data_acc = tvm::compute(
      {quantized_data->shape[0]},
      [&](Var i) {
          return tvm::sum(tvm::cast(Int(32), quantized_data(i, k)), {k});
      },
      "tensor",
      "int8_quantize_row_offset"
      );

  return {data_acc};
}

Array<Tensor> data_int8_mm_dequantize(
    const Tensor& data,
    const Tensor& weight,
    const Tensor& weight_acc,
    const Tensor& data_acc,
    const Tensor& data_scale,
    const Tensor& data_zero_point,
    const double weight_scale,
    const int weight_zero_point,
    const int32_t N) {
  // assume M, K and N, K on input shape
  CHECK(weight->shape.size() == 4);
  auto k = tvm::reduce_axis(Range(0, data->shape[1]), "k");
  auto scale_mul = make_const(Float(32), weight_scale) * data_scale(0);
  auto out_shape = {data->shape[0], weight->shape[0] * weight->shape[2]};

  auto quantized_mm = tvm::compute(
        out_shape,
        [&](Var i, Var j) {
          return tvm::sum(tvm::cast(Int(32), data(i, k)) * tvm::cast(Int(32), weight(j / 16, k / 4, j % 16, k % 4)), {k});
        },
        "tensor",
        "quantized_mm"
      );

  auto result = tvm::compute(
        {data->shape[0], Expr(N)},
        [&](Var i, Var j) {
          return scale_mul*(tvm::cast(Float(32), (quantized_mm(i, j)-data_acc(i)*weight_zero_point-
                            weight_acc(j)*data_zero_point(0))));
        },
        "tensor",
        "mm_dequantize"
      );

  return {result};
}
} // namespace topi
