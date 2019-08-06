#pragma once

namespace topi {

namespace generic {
/*!
 * \brief Create an x86 schedule for the given sparse ops.
 *
 * \param target The target to generate a schedule for.
 * \param outs The output tensors.
 *
 * \return A schedule for the given ops.
 */
inline tvm::Schedule schedule_custom_layer_norm(
    const tvm::Array<tvm::Tensor>& outs) {
  tvm::Array<tvm::Operation> out_ops;
  for (auto out : outs) {
    out_ops.push_back(out->op);
  }
  auto s = create_schedule(out_ops);
  return s;
}
} // namespace generic
} // namespace topi
