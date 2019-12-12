#include "replace_single_ops.h"
#include "fusion_pass.h"
#include "operators.h"

#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

using namespace torch::jit;

namespace {
std::unordered_set<std::string> whitelisted_ops {
  "aten::matmul",
};
}

// Selectively replace some of the ops with a TVM block
void ReplaceSingleOps(std::shared_ptr<Graph>& graph) {
  for (auto it = graph->nodes().begin(); it != graph->nodes().end();) {
    if (whitelisted_ops.find(it->kind().toQualString()) !=
        whitelisted_ops.end() && isSupported(*it)) {
      auto tvm_node = SubgraphUtils::createSingletonSubgraph(
          *it, getTVMSymbol());
      it = ++tvm_node->iterator();
    } else {
      ++it;
    }
  }
}
