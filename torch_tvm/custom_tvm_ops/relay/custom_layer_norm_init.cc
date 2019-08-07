#include <relay/op/op_common.h>
#include <relay/op/type_relations.h>
#include <relay/pass/alter_op_layout.h>
#include <tvm/expr.h>
#include <tvm/operation.h>
#include <tvm/relay/op.h>
#include "custom_layer_norm.h"
#include "custom_layer_norm_attrs.h"

namespace tvm {
namespace relay {

TVM_REGISTER_API("relay.op.nn._make.custom_layer_norm")
    .set_body_typed(MakeCustomLayerNorm);

RELAY_REGISTER_OP("nn.custom_layer_norm")
    .describe(R"code(Applies the layer norm transformation with per element
    affine transform applied after normalization.

- **data**: `Tensor with N dims`
- **out**: `Tensor with N dims`

)code" TVM_ADD_FILELINE)
    .set_attrs_type_key("relay.attrs.CustomLayerNormAttrs")
    .set_num_inputs(3)
    .add_argument("data", "ND Tensor", "Input data.")
    .add_argument("gamma", "ND Tensor", "Input data.")
    .add_argument("beta", "ND Tensor", "Input data.")
    .set_support_level(1)
    .add_type_rel("CustomLayerNorm", CustomLayerNormRel);

} // namespace relay
} // namespace tvm
