#include "fusion_pass.h"
#include "operators.h"

using namespace torch::jit;

value_list sortReverseTopological(ArrayRef<Value*> inputs, Block* block) {
  value_list result;
  for (auto i : inputs) {
    if (i->node()->owningBlock() == block) {
      result.push_back(i);
    }
  }
  // Sort in reverse topological order
  std::sort(result.begin(), result.end(), [&](Value* a, Value* b) {
    return a->node()->isAfter(b->node());
  });
  return result;
}

bool canHandle(Block* block, AliasDb& aliasDb);
bool canHandle(Node* node, AliasDb& aliasDb) {
  if (node->kind() == prim::Constant) {
    return true;
  }
  if (node->kind() == prim::Loop) {
    return false; // TODO
    Block* body = node->blocks().at(0);
    return canHandle(body, aliasDb);
  }
  return isSupported(node);
}

bool canHandle(Block* block, AliasDb& aliasDb) {
  for (Node* node : block->nodes()) {
    if (!canHandle(node, aliasDb)) {
      return false;
    }
  }
  return true;
}

c10::optional<Node*> tryMerge(
    Node* consumer,
    Node* producer,
    AliasDb& aliasDb) {
  bool canMerge = canHandle(producer, aliasDb) &&
      canHandle(consumer, aliasDb) &&
      aliasDb.moveBeforeTopologicallyValid(producer, consumer);

  if (!canMerge) {
    return c10::nullopt;
  }

  // Consumer is only allowed to have writers
  if (aliasDb.hasInputWriters(consumer)) {
    if (!aliasDb.isInPlace(producer)) {
      return c10::nullopt;
    }
  }

  if (aliasDb.hasOutputWriters(consumer)) {
    return c10::nullopt;
  }

  if (!consumer->hasAttribute(attr::Subgraph) &&
      consumer->kind() != getTVMSymbol()) {
    consumer = SubgraphUtils::createSingletonSubgraph(consumer, getTVMSymbol());
  }
  SubgraphUtils::mergeNodeIntoSubgraph(producer, consumer);

  return consumer;
}

graph_node_list::iterator scanNode(
    Node* consumer,
    AliasDb& aliasDb,
    Block* block) {
  auto inputs = sortReverseTopological(consumer->inputs(), block);
  for (auto input : inputs) {
    if (auto group = tryMerge(consumer, input->node(), aliasDb)) {
      // we successfully merged, so the new group's `inputs` may have
      // changed. So rescan the new group for more merging opportunities.
      return group.value()->reverseIterator();
    }
  }
  return ++consumer->reverseIterator();
}

void FuseSupportedOps(std::shared_ptr<Graph> graph) {
  AliasDb aliasDb(graph);
  auto block = graph->block();

  for (auto it = block->nodes().rbegin(); it != block->nodes().rend();) {
    it = scanNode(*it, aliasDb, block);
  }
}

const torch::jit::Symbol& getTVMSymbol() {
  static torch::jit::Symbol tvm_sym =
      torch::jit::Symbol::fromQualString("tvm::CompilationGroup");
  return tvm_sym;
}
