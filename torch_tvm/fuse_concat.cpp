#include "fusion_pass.h"
#include "operators.h"

using namespace torch::jit;

const size_t subgraph_arg_limit_ = 128;

bool isFusableCatNode(Node* node) {
  if (node->kind() != aten::cat) {
    return false;
  }
  if (!node->is_constant(attr::dim)) {
    return false;
  }

  auto tensors_node = node->namedInput(attr::tensors)->node();
  if ((tensors_node->inputs().size() + node->outputs().size()) >
      subgraph_arg_limit_) {
    return false;
  }
  if (tensors_node->kind() != prim::ListConstruct) {
    return false;
  }

  if (tensors_node->output()->uses().size() > 1) {
    return false;
  }

  return true;
}

Node* createFusedConcat(Node* node) {
  AT_ASSERT(node->kind() == aten::cat);
  Graph* graph = node->owningGraph();
  Node* list_construct = node->namedInput(attr::tensors)->node();
  int64_t dim = node->get<int64_t>(attr::dim).value();

  Node* fused_cat = graph->create(prim::FusedConcat, list_construct->inputs())
                        ->i_(attr::dim, dim);
  fused_cat->insertBefore(list_construct);
  fused_cat->output()->copyMetadata(node->output());

  node->output()->replaceAllUsesWith(fused_cat->output());
  node->destroyCurrent();
  if (list_construct->output()->uses().empty()) {
    list_construct->destroy();
  }
  return fused_cat;
}

void fuseConcats(Block* block_) {
  for (auto it = block_->nodes().rbegin(); it != block_->nodes().rend();
       ++it) {
    Node* cat = *it;
    if (!isFusableCatNode(cat)) {
      continue;
    }
    createFusedConcat(cat);
  }
}

void FuseConcat(std::shared_ptr<Graph> graph) {
  auto block = graph->block();
  fuseConcats(block);
}
