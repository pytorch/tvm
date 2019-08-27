#pragma once

#include <tvm/attrs.h>
#include <tvm/relay/expr.h>

namespace tvm {
namespace relay {
struct CustomLayerNormAttrs : public tvm::AttrsNode<CustomLayerNormAttrs> {
  int num_axis_to_normalize;
  bool affine;
  double eps;
  TVM_DECLARE_ATTRS(CustomLayerNormAttrs, "relay.attrs.CustomLayerNormAttrs") {
    TVM_ATTR_FIELD(num_axis_to_normalize).set_default(-1);
    TVM_ATTR_FIELD(affine).set_default(false);
    TVM_ATTR_FIELD(eps).set_default(1e-5);
  }
};
} // namespace relay
} // namespace tvm
