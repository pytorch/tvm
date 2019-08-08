#include "fusion_pass.h"
#include "operators.h"

using namespace torch::jit;

bool isFusableCatNode(Node* node) {
  if (node->kind() != aten::cat)
    return false;
  if (!node->is_constant(attr::dim))
    return false;

  auto tensors_node = node->namedInput(attr::tensors)->node();
  if ((tensors_node->inputs().size() + node->outputs().size()) >
      128) {
    return false;
  }
  if (tensors_node->kind() != prim::ListConstruct)
    return false;

  if (tensors_node->output()->uses().size() > 1)
    return false;
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
  if (list_construct->output()->uses().empty()) {
    list_construct->destroy();
  }
  return fused_cat;
}

void fuseConcats(Block* block_, AliasDb& aliasDb) {
  for (auto it = block_->nodes().rbegin(); it != block_->nodes().rend();
       ++it) {
    Node* cat = *it;
    if (!isFusableCatNode(cat)) {
      continue;
    }

    Node* list_construct = cat->namedInput(attr::tensors)->node();
    createFusedConcat(cat);
  }
}

void FuseConcat(std::shared_ptr<Graph> graph) {
  AliasDb aliasDb(graph);
  auto block = graph->block();
  fuseConcats(block, aliasDb);
}
