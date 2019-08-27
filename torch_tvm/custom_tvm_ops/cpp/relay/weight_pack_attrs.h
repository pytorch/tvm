#pragma once

#include <tvm/attrs.h>
#include <tvm/relay/expr.h>

namespace tvm {
namespace relay {

struct WeightPackAttrs : public tvm::AttrsNode<WeightPackAttrs> {
  int32_t pack_width;
  TVM_DECLARE_ATTRS(WeightPackAttrs, "relay.attrs.WeightPackAttrs") {
    TVM_ATTR_FIELD(pack_width).set_default(1);
  }
};
} // namespace relay
} // namespace tvm
