#include <tvm/expr.h>
#include <tvm/operation.h>
#include <tvm/tensor.h>
#include <topi/tags.h>
#include <topi/detail/array_utils.h>
#include "custom_layer_norm_generic_sched.h"

namespace topi {
namespace generic {

namespace {
void layer_norm_sched(tvm::Schedule& s, const tvm::Tensor& out) {
  auto divide_1 = out->op.as<tvm::ComputeOpNode>()->InputTensors()[1];
  auto divide_2 = out->op.as<tvm::ComputeOpNode>()->InputTensors()[2];
  auto mean_var_sum = divide_1->op.as<tvm::ComputeOpNode>()->InputTensors()[0];
  s[divide_1].compute_inline();
  s[divide_2].compute_inline();
  auto k = s[mean_var_sum]->op.as<tvm::ComputeOpNode>()->reduce_axis;
  tvm::IterVar ko, ki;
  s[mean_var_sum].split(k[0], 16, &ko, &ki);
  auto factored_tensors = s.rfactor(mean_var_sum, ki, -1);
  s[mean_var_sum].compute_at(s[out], out->op.as<tvm::ComputeOpNode>()->axis[0]);
  s[factored_tensors[0]].compute_at(s[out],
      out->op.as<tvm::ComputeOpNode>()->axis[0]);
}
} // namespace

tvm::Schedule schedule_custom_layer_norm(const tvm::Array<tvm::Tensor>& outs) {
  tvm::Array<tvm::Operation> out_ops;
  for (auto out : outs) {
    out_ops.push_back(out->op);
  }
  auto s = create_schedule(out_ops);
  //Copy paste traverse logic from elsewhere. This is also the traverse logic
  //in dense schedule in dense.py
  std::function<void(tvm::Operation)> traverse;
  traverse = [&](const tvm::Operation& op) {
    // Inline all one-to-one-mapping operators except the last stage (output)
    if (is_injective(op->tag)) {
      if (!detail::contains(s->outputs, op)) {
        s[op].compute_inline();
      }
      for (auto tensor : op->InputTensors()) {
        if (tensor->op->InputTensors().size() > 0) {
          traverse(tensor->op);
        }
      }
    }
    else if (op->tag == "custom_layer_norm_tag") {
      auto layer_norm = op.output(0);
      layer_norm_sched(s, layer_norm);
    } else {
      LOG(ERROR) << "Unsupported operator " << op->tag;
    }
  };
  traverse(outs[0]->op);
  return s;
}
} // namespace generic
} // namespace topi
