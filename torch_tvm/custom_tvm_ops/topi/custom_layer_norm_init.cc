#include <topi/generic/injective.h>
#include <topi/x86/default.h>
#include <tvm/build_module.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/runtime/packed_func.h>

#include <custom_tvm_ops/relay/custom_layer_norm_attrs.h>

#include "custom_layer_norm.h"
#include "custom_layer_norm_generic_sched.h"

namespace tvm {

using tvm::relay::OpPatternKind;

class CustomTOPIOpRegisterer {
 public:
  CustomTOPIOpRegisterer() {
    auto reg_ptr = runtime::Registry::Get("relay.op._Register");
    CHECK(reg_ptr) << "Cannot find function for relay.op._Register";
    (*reg_ptr)(
        "nn.custom_layer_norm",
        "TOpPattern",
        static_cast<int>(OpPatternKind::kOutEWiseFusable),
        10);
    (*reg_ptr)(
        "nn.custom_layer_norm",
        "FTVMCompute",
        tvm::relay::FTVMCompute(
            [](const tvm::Attrs& attrs,
               const tvm::Array<tvm::Tensor>& inputs,
               const tvm::relay::Type& out_type,
               const tvm::Target& target) -> tvm::Array<tvm::Tensor> {
              const relay::CustomLayerNormAttrs* param =
                  attrs.as<relay::CustomLayerNormAttrs>();
              auto num_axis_to_normalize = param->num_axis_to_normalize;
              auto affine = param->affine;
              auto eps = param->eps;
              return tvm::Array<tvm::Tensor>{topi::custom_layer_norm(
                  inputs[0],
                  inputs[1],
                  inputs[2],
                  num_axis_to_normalize,
                  affine,
                  eps)};
            }),
        10);
    (*reg_ptr)(
        "nn.custom_layer_norm",
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
  }
};

static CustomTOPIOpRegisterer custom_top_op_registerer;

} // namespace tvm

namespace topi {

using namespace tvm;
using namespace tvm::runtime;

TVM_REGISTER_GLOBAL("nn.custom_layer_norm")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      CHECK(args.size() == 6);
      *rv = custom_layer_norm(
          args[0],
          args[1],
          args[2],
          args[3],
          args[4],
          static_cast<double>(args[5]));
    });

TVM_REGISTER_GLOBAL("topi.generic.schedule_custom_layer_norm")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      *rv = topi::generic::schedule_custom_layer_norm(args[0]);
    });

TVM_REGISTER_GENERIC_FUNC(schedule_custom_layer_norm)
    .set_default(PackedFunc([](TVMArgs args, TVMRetValue* ret) {
      Array<Tensor> outs;
      NodeRef argNodeRef = args[0];
      if (argNodeRef->type_index() == outs->type_index()) {
        outs = args[0];
      } else {
        outs = Array<Tensor>{args[0]};
      }
      *ret = topi::generic::schedule_custom_layer_norm(outs);
    }));

} // namespace topi
