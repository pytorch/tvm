#include <relay/op/op_common.h>
#include <relay/op/type_relations.h>
#include <relay/pass/alter_op_layout.h>
#include <tvm/expr.h>
#include <tvm/operation.h>
#include <tvm/relay/op.h>
#include "custom_dense.h"

namespace tvm {
namespace relay {

RELAY_REGISTER_OP("nn.custom_dense")
    .describe(R"code(Applies GEMM op on data with weights which are
prepacked for cache friendly vectorization.
- **data**: `Tensor with 2 dims`
- **weight**: `Tensor with 3 dims`
- **out**: `Tensor with 2 dims`

)code" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .add_argument("data", "ND Tensor", "Input data.")
    .add_argument("weight", "ND Tensor", "Input data.")
    .set_support_level(1)
    .add_type_rel("CustomDense", CustomDenseRel);

RELAY_REGISTER_OP("nn.dense_weight_pack")
    .describe(R"code(Packs weight data for cache friendly vectorization.
- **weight**: `Tensor with 2 dims`
- **out**: `Tensor with 3 dims`

)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("weight", "ND Tensor", "Input data.")
    .set_support_level(1)
    .add_type_rel("DenseWeightPack", DenseWeightPackRel);
} // namespace relay
} // namespace tvm
