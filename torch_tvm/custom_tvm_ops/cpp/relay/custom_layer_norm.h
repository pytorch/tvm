#include <relay/op/op_common.h>
#include <relay/op/type_relations.h>
#include <relay/pass/alter_op_layout.h>
#include <tvm/expr.h>
#include <tvm/operation.h>
#include <tvm/relay/op.h>

namespace tvm {
namespace relay {

Expr MakeCustomLayerNorm(
    Expr data,
    Expr gamma,
    Expr beta,
    const int num_axis_to_normalize,
    const bool affine,
    const double eps);

bool CustomLayerNormRel(
    const Array<Type>& types,
    int num_inputs, /* unused */
    const Attrs& attrs,
    const TypeReporter& reporter);
} // namespace relay
} // namespace tvm
