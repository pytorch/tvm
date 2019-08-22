#pragma once

#include <tvm/attrs.h>
#include <tvm/relay/expr.h>

namespace tvm {
namespace relay {

class NoneVar;
class NoneVarNode : public ExprNode {
 public:

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  TVM_DLL static NoneVar make();

  static constexpr const char* _type_key = "relay.NoneVar";
  TVM_DECLARE_NODE_TYPE_INFO(NoneVarNode, ExprNode);
};
RELAY_DEFINE_NODE_REF(NoneVar, NoneVarNode, Expr);
} // relay
} // tvm
