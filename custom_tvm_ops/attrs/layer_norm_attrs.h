#pragma once

#include <tvm/relay/expr.h>
#include <tvm/attrs.h>

namespace tvm {
namespace relay {
struct CustomLayerNormAttrs : public tvm::AttrsNode<CustomLayerNormAttrs> {
  Array<Integer> axis;
  bool affine;
  TVM_DECLARE_ATTRS(CustomLayerNormAttrs,
      "relay.attrs.CustomLayerNormAttrs") {
    TVM_ATTR_FIELD(axis).set_default(NullValue<Array<Integer>>());
    TVM_ATTR_FIELD(affine).set_default(false);
  }
};
}
}

