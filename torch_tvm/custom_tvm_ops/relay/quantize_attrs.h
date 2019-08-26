#pragma once

#include <tvm/relay/expr.h>
#include <tvm/attrs.h>

namespace tvm {
namespace relay {
struct QuantizeSchemeAttrs : public tvm::AttrsNode<QuantizeSchemeAttrs> {
  int precision;
  bool is_signed;

   TVM_DECLARE_ATTRS(QuantizeSchemeAttrs, "relay.attrs.QuantizedParamsAttrs") {
    TVM_ATTR_FIELD(precision).set_default(8)
      .describe("The integer precision we want to quantize to.");
    TVM_ATTR_FIELD(is_signed).set_default(false)
      .describe("Signed or unsigned integer we want to quantize to.");
  }
};

struct QuantizedParamsAttrs : public tvm::AttrsNode<QuantizedParamsAttrs> {
  double w_scale;
  int w_zp;

   TVM_DECLARE_ATTRS(QuantizedParamsAttrs, "relay.attrs.QuantizedParamsAttrs") {
    TVM_ATTR_FIELD(w_scale).set_default(1.0)
      .describe("weight scale.");
    TVM_ATTR_FIELD(w_zp).set_default(0)
      .describe("weight zero point.");
  }
};
}
}
