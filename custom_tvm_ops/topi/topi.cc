#include <tvm/build_module.h>
#include <tvm/packed_func_ext.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include "topi/generic/custom_layer_norm.h"
#include "topi/nn/custom_layer_norm.h"

#include "topi_init.h"

int temp_to_be_removed = (tvm::registerCustomTopiSchedules(), 0);

namespace topi {

using namespace tvm;
using namespace tvm::runtime;

TVM_REGISTER_GLOBAL("nn.custom_layer_norm")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      CHECK(args.size() == 6);
      *rv = custom_layer_norm(args[0], args[1], args[2],
          args[3], args[4],static_cast<double>(args[5]));
    });

TVM_REGISTER_GLOBAL("topi.generic.schedule_custom_layer_norm")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      *rv = topi::generic::schedule_custom_layer_norm(args[0]);
    });

TVM_REGISTER_GENERIC_FUNC(schedule_custom_layer_norm)
    .set_default(
        PackedFunc([](TVMArgs args, TVMRetValue* ret) {
          Array<Tensor> outs;
          NodeRef argNodeRef = args[0];
          if (argNodeRef->type_index() == outs->type_index()) {
            outs = args[0];
          } else {
            outs = Array<Tensor>{args[0]};
          }
          *ret = topi::generic::schedule_custom_layer_norm(outs);
        }
      )
    );

} // namespace topi
