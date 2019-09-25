#include <tvm/data_layout.h>
#include <tvm/relay/op.h>
#include <vector>

#include <tvm/expr.h>
#include <tvm/operation.h>
#include <relay/op/op_common.h>
#include <relay/op/type_relations.h>
#include <relay/pass/alter_op_layout.h>
#include "quantize_attrs.h"
#include "quantize.h"

#include <cstdint>
namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(QuantizedParamsAttrs);

TVM_REGISTER_NODE_TYPE(QuantizeSchemeAttrs);

TVM_REGISTER_API("relay.op.nn._make.quantize_data_int8_quantize")
  .set_body_typed(MakeDataInt8Quantization);

TVM_REGISTER_API("relay.op.nn._make.quantize_data_int8_row_offset")
  .set_body_typed(MakeDataInt8RowOffset);


RELAY_REGISTER_OP("nn.quantize_data_int8_quantize")
.describe(R"code(dynamic quantization of activation.
- **data**: (M, N)
)code" TVM_ADD_FILELINE)
  .set_num_inputs(3)
  .add_argument("data", "Tensor", "The input tensor.")
  .add_argument("zero_point", "Tensor", "The zero_point parameter for quantization")
  .add_argument("scale", "Tensor", "the scale parameter for quantization")
  .set_attrs_type_key("relay.attrs.QuantizeSchemeAttrs")
  .set_support_level(10)
  .add_type_rel("DataInt8Quantization", DataInt8QuantizationRel);


RELAY_REGISTER_OP("nn.quantize_data_int8_row_offset")
.describe(R"code(dynamic row offset calculation of quantized data.
- **data**: (M, N)
)code" TVM_ADD_FILELINE)
  .set_num_inputs(1)
  .add_argument("data", "Tensor", "Quantized input tensor.")
  .set_support_level(10)
  .add_type_rel("DataInt8RowOffset", DataInt8RowOffsetRel);


TVM_REGISTER_API("relay.op.nn._make.quantize_findminmax")
  .set_body_typed(MakeFindMinMax);

RELAY_REGISTER_OP("nn.quantize_findminmax")
.describe(R"code(find min and max of the input data.
- **data**: (M, N)
)code" TVM_ADD_FILELINE)
  .set_num_inputs(1)
  .add_argument("data", "Tensor", "The input data tensor.")
  .set_support_level(5)
  .add_type_rel("FindMinMax", FindMinMaxRel);


TVM_REGISTER_API("relay.op.nn._make.quantize_data_mm_dequantize")
  .set_body_typed(MakeDataMMDequantize);

RELAY_REGISTER_OP("nn.quantize_data_mm_dequantize")
.describe(R"code(multiply the weight and data, then dequantize the data into floating point.
- **data**: (M, N)
)code" TVM_ADD_FILELINE)
  .set_num_inputs(6)
  .add_argument("data", "Tensor", "The input data tensor.")
  .add_argument("weight", "Tensor", "The input weight tensor.")
  .add_argument("weight_acc", "Tensor", "The accumulation of each column")
  .add_argument("data_acc", "Tensor", "The accumulation of each row")
  .add_argument("data_scale", "Tensor", "The activation scale")
  .add_argument("data_zero_point", "Tensor", "The activation zero_point")
  .set_attrs_type_key("relay.attrs.QuantizedParamsAttrs")
  .set_support_level(5)
  .add_type_rel("DataMMDequantize", DataMMDequantizeRel);


TVM_REGISTER_API("relay.op.nn._make.choose_quantize_params")
.set_body_typed(MakeChooseQuantizeParams);

RELAY_REGISTER_OP("nn.choose_quantize_params")
.describe(R"code(calculate the zero_point and scale.
)code" TVM_ADD_FILELINE)
  .set_num_inputs(2)
  .set_attrs_type_key("relay.attrs.QuantizeSchemeAttrs")
  .add_argument("data_min", "Tensor", "The min of input data.")
  .add_argument("data_max", "Tensor", "The max of input data.")
  .set_support_level(4)
  .add_type_rel("ChooseQuantizeParams", ChooseQuantizeParamsRel);

}  // namespace relay
}  // namespace tvm
