#include <relay/op/op_common.h>
#include <relay/op/type_relations.h>
#include <relay/pass/alter_op_layout.h>
#include <tvm/expr.h>
#include <tvm/operation.h>
#include <tvm/relay/op.h>

namespace tvm {
namespace relay {

bool CustomDenseRel(
    const Array<Type>& types,
    int num_inputs, /* unused */
    const Attrs& attrs,
    const TypeReporter& reporter);

bool DenseWeightPackRel(
    const Array<Type>& types,
    int num_inputs, /* unused */
    const Attrs& attrs,
    const TypeReporter& reporter);
} // namespace relay
} // namespace tvm
