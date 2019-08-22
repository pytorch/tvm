#include "none_var_expr.h"

namespace tvm {
namespace relay {

NoneVar NoneVarNode::make() {
  NodePtr<NoneVarNode> n = make_node<NoneVarNode>();
  return NoneVar(n);
}

TVM_REGISTER_NODE_TYPE(NoneVarNode);

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<NoneVarNode>([](const NoneVarNode* node, tvm::IRPrinter* p) {
    p->stream << "NoneVar()";
  });

} // relay
} // tvm
