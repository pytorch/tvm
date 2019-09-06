#include <topi/generic/injective.h>
#include <topi/x86/default.h>
#include <tvm/build_module.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/runtime/packed_func.h>

#include <custom_tvm_ops/cpp/relay/custom_layer_norm_attrs.h>
#include <custom_tvm_ops/cpp/relay/quantize_attrs.h>

#include "custom_layer_norm.h"
#include "custom_layer_norm_generic_sched.h"
#include "quantize.h"
#include "contrib/quantize.h"
#include "generic/quantize_generic_sched.h"


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
              return topi::generic::schedule_custom_layer_norm(outs);
            }),
        10);
    (*reg_ptr)(
        "nn.quantize_findminmax",
        "TOpPattern",
        static_cast<int>(OpPatternKind::kCommReduce),
        10);
    (*reg_ptr)(
        "nn.quantize_findminmax",
        "FTVMCompute",
        tvm::relay::FTVMCompute(
            [](const tvm::Attrs& attrs,
               const tvm::Array<tvm::Tensor>& inputs,
               const tvm::relay::Type& out_type,
               const tvm::Target& target) -> tvm::Array<tvm::Tensor> {
              return topi::contrib::quantize_findminmax(inputs[0]);
            }),
        10);
    (*reg_ptr)(
        "nn.quantize_findminmax",
        "FTVMSchedule",
        tvm::relay::FTVMSchedule(
            [](const tvm::Attrs& attrs,
               const tvm::Array<tvm::Tensor>& outs,
               const tvm::Target& target) -> tvm::Schedule {
              return  topi::generic::schedule_quantize_findminmax(outs);
            }),
        10);
    (*reg_ptr)(
        "nn.choose_quantize_params",
        "TOpPattern",
        static_cast<int>(OpPatternKind::kOpaque),
        10);
    (*reg_ptr)(
        "nn.choose_quantize_params",
        "FTVMCompute",
        tvm::relay::FTVMCompute(
            [](const tvm::Attrs& attrs,
               const tvm::Array<tvm::Tensor>& inputs,
               const tvm::relay::Type& out_type,
               const tvm::Target& target) -> tvm::Array<tvm::Tensor> {
              const auto* param = attrs.as<relay::QuantizeSchemeAttrs>();
              auto precision = param->precision;
              auto is_signed = param->is_signed;
              return topi::contrib::choose_quantize_params(inputs[0], inputs[1], is_signed, precision);
            }),
        10);
    (*reg_ptr)(
        "nn.choose_quantize_params",
        "FTVMSchedule",
        tvm::relay::FTVMSchedule(
            [](const tvm::Attrs& attrs,
               const tvm::Array<tvm::Tensor>& outs,
               const tvm::Target& target) -> tvm::Schedule {
              return topi::generic::schedule_choose_quantize_params(outs);
            }),
        10);
    (*reg_ptr)(
        "nn.quantize_data_int8_quantize",
        "TOpPattern",
        static_cast<int>(OpPatternKind::kOutEWiseFusable),
        10);
    (*reg_ptr)(
        "nn.quantize_data_int8_quantize",
        "FTVMCompute",
        tvm::relay::FTVMCompute(
            [](const tvm::Attrs& attrs,
               const tvm::Array<tvm::Tensor>& inputs,
               const tvm::relay::Type& out_type,
               const tvm::Target& target) -> tvm::Array<tvm::Tensor> {
              const auto* param = attrs.as<relay::QuantizeSchemeAttrs>();
              auto precision = param->precision;
              auto is_signed = param->is_signed;
              return topi::data_int8_quantize(inputs[0], inputs[1], inputs[2], is_signed, precision);
            }),
        10);
    (*reg_ptr)(
        "nn.quantize_data_int8_quantize",
        "FTVMSchedule",
        tvm::relay::FTVMSchedule(
            [](const tvm::Attrs& attrs,
               const tvm::Array<tvm::Tensor>& outs,
               const tvm::Target& target) -> tvm::Schedule {
              return topi::generic::schedule_quantize_data_int8_quantize(outs);
            }),
        10);
    (*reg_ptr)(
        "nn.quantize_data_mm_dequantize",
        "TOpPattern",
        static_cast<int>(OpPatternKind::kOutEWiseFusable),
        10);
  }
};

static CustomTOPIOpRegisterer custom_top_op_registerer;

} // namespace tvm

namespace topi {

using namespace tvm;
using namespace tvm::runtime;
// For python API
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

TVM_REGISTER_GLOBAL("nn.compute_quantized_mm_dequantize")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      CHECK(args.size() == 9);
      *rv = data_int8_mm_dequantize(
        args[0], args[1], args[2], args[3],
        args[4], args[5], args[6], args[7], args[8]);
    });

} // namespace topi
