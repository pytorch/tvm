#pragma once

#include <tvm/relay/expr.h>
#include <tvm/attrs.h>

namespace tvm {
namespace relay {
struct CustomLayerNormAttrs : public tvm::AttrsNode<CustomLayerNormAttrs> {
  int num_axis_to_normalize;
  bool affine;
  TVM_DECLARE_ATTRS(CustomLayerNormAttrs,
      "relay.attrs.CustomLayerNormAttrs") {
    TVM_ATTR_FIELD(num_axis_to_normalize).set_default(-1);
    TVM_ATTR_FIELD(affine).set_default(false);
  }
};
}
}

