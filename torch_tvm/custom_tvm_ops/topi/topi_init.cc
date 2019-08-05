#include <topi/x86/default.h>
#include <tvm/build_module.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/expr.h>
#include <tvm/runtime/packed_func.h>
#include <topi/generic/injective.h>

#include <custom_tvm_ops/relay/layer_norm_attrs.h>

#include "topi_init.h"
#include "custom_layer_norm_generic_sched.h"
#include "custom_layer_norm.h"

#include <mutex>

namespace tvm {

using tvm::relay::OpPatternKind;

void registerCustomTopiSchedules() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    constexpr int kOutEWiseFusable =
        static_cast<int>(OpPatternKind::kOutEWiseFusable);

    auto reg_ptr = tvm::runtime::Registry::Get("relay.op._Register");
    CHECK(reg_ptr) << "Cannot find function for relay.op._Register";
    auto reg = *reg_ptr;

    reg("nn.custom_layer_norm", "TOpPattern", kOutEWiseFusable, 10);
    reg("nn.custom_layer_norm",
        "FTVMCompute",
        tvm::relay::FTVMCompute(
            [](const tvm::Attrs& attrs,
               const tvm::Array<tvm::Tensor>& inputs,
               const tvm::relay::Type& out_type,
               const tvm::Target& target) -> tvm::Array<tvm::Tensor> {
              const relay::CustomLayerNormAttrs* param = attrs.as<relay::CustomLayerNormAttrs>();
              auto num_axis_to_normalize = param->num_axis_to_normalize;
              auto affine = param->affine;
              auto eps = param->eps;
              return tvm::Array<tvm::Tensor>{topi::custom_layer_norm(
                    inputs[0], inputs[1], inputs[2], num_axis_to_normalize, affine, eps)};
            }),
        10);
    reg("nn.custom_layer_norm",
        "FTVMSchedule",
        tvm::relay::FTVMSchedule(
            [](const tvm::Attrs& attrs,
               const tvm::Array<tvm::Tensor>& outs,
               const tvm::Target& target) -> tvm::Schedule {
              // dispatches x86 schedule when the target is "llvm" and the
              // default schedule for other targets
              auto schedule_custom_layer_norm =
                  tvm::GenericFunc::Get("schedule_custom_layer_norm");
              return schedule_custom_layer_norm(outs);
            }),
        10);
  });
}
} // namespace tvm
