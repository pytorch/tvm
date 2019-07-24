#pragma once

namespace topi {
using namespace tvm;

namespace generic {
/*!
 * \brief Create an x86 schedule for the given sparse ops.
 *
 * \param target The target to generate a schedule for.
 * \param outs The output tensors.
 *
 * \return A schedule for the given ops.
 */
inline Schedule schedule_custom_layer_norm(
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
