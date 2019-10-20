#pragma once

#include "tvm/operation.h"

namespace topi {
using namespace tvm;

namespace generic {
/*!
 * \brief Create an x86 schedule for the given quantize ops.
 *
 * \param target The target to generate a schedule for.
 * \param outs The output tensors.
 *
 * \return A schedule for the given ops.
 */
inline Schedule schedule_choose_quantize_params(
    const Array<Tensor>& outs) {
  Array<Operation> out_ops;
  for (auto out : outs) {
    out_ops.push_back(out->op);
  }
  auto s = create_schedule(out_ops);
  return s;
}

inline Schedule schedule_quantize_findminmax(
    const Array<Tensor>& outs) {
  Array<Operation> out_ops;
  for (auto out : outs) {
    out_ops.push_back(out->op);
  }
  auto s = create_schedule(out_ops);
  return s;
}

inline Schedule schedule_quantize_data_int8_quantize(
    const Array<Tensor>& outs) {
  Array<Operation> out_ops;
  for (auto out : outs) {
    out_ops.push_back(out->op);
  }
  auto s = create_schedule(out_ops);
  return s;
}

inline Schedule schedule_quantize_data_int8_row_offset(
    const Array<Tensor>& outs) {
  Array<Operation> out_ops;
  for (auto out : outs) {
    out_ops.push_back(out->op);
  }
  auto s = create_schedule(out_ops);
  return s;
}

inline Schedule schedule_quantized_mm_dequantize(
    const Target& target,
    const Array<Tensor>& outs) {
  Array<Operation> out_ops;
  for (auto out : outs) {
    out_ops.push_back(out->op);
  }
  auto s = create_schedule(out_ops);
  return s;
}

} // namespace generic
} // namespace topi
