#include "custom_layer_norm_generic_sched.h"

namespace topi {
namespace generic {
tvm::Schedule schedule_custom_layer_norm(
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
