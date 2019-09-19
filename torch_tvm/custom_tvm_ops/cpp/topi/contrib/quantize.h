#pragma once

#include "topi/detail/extern.h"
#include "topi/tags.h"
#include "tvm/operation.h"

namespace topi {
namespace contrib {

using namespace tvm;
using namespace topi::detail;

inline Array<Tensor> quantize_findminmax(
    const Tensor& data) {

  return make_extern(
        {{1}, {1}},
        {data->dtype, data->dtype},
        {data},
        [&](Array<Buffer> ins, Array<Buffer> outs) {
          return call_packed({
            Expr("tvm.contrib.find_minmax"),
            pack_buffer(ins[0]),
            pack_buffer(outs[0]),
            pack_buffer(outs[1])});
        },
        "C",
        "findminmax",
        {});
}

inline Array<Tensor> choose_quantize_params(
    const Tensor& data_min,
    const Tensor& data_max,
    bool is_signed,
    int precision) {
  auto q_min = is_signed? -(1 << (precision - 1)) : 0;
  auto q_max = is_signed? ((1 << (precision - 1)) - 1) : (1 << precision) - 1;

  return make_extern(
        {{1}, {1}},
        {Int(32), Float(32)},
        {data_min, data_max},
        [&](Array<Buffer> ins, Array<Buffer> outs) {
          return call_packed({
            Expr("tvm.contrib.choose_quantize_params"),
            pack_buffer(ins[0]),
            pack_buffer(ins[1]),
            pack_buffer(outs[0]),
            pack_buffer(outs[1]),
            q_min,
            q_max});
        },
        "C",
        "chooseQuantizeParams",
        {});
}

} // namespace contrib
} // namespace topi
