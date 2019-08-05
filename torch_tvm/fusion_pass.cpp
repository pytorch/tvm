#include "fusion_pass.h"
#include "operators.h"

#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>

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

#define REQ(cond)                           \
  if (!(cond)) {                            \
    GRAPH_DEBUG("Failed cond " #cond "\n"); \
    return c10::nullopt;                    \
  }
c10::optional<Node*> tryMerge(
    Node* consumer,
    Node* producer,
    AliasDb& aliasDb) {
  GRAPH_DEBUG(
      "Trying producer ",
      producer->kind().toQualString(),
      " and consumer ",
      consumer->kind().toQualString(),
      ":\n");

  // Symbolic checks
  REQ(canHandle(producer, aliasDb));
  REQ((canHandle(consumer, aliasDb) || consumer->kind() == getTVMSymbol()));

  // Alias checks
  // Requirement:
  // - moveAfterTopologicallyValid(consumer, producer)
  // - One of:
  //   1) Both are in-place ops
  //   2) Consumer is in-place, producer !hasInputWriters
  //   3) Producer is in-place, consumer !hasOutputWriters
  REQ(aliasDb.moveAfterTopologicallyValid(consumer, producer));

  // 1)
  if (!(aliasDb.isMutable(consumer) && aliasDb.isMutable(producer))) {
    // 2)
    if (aliasDb.isMutable(consumer)) {
      REQ(!aliasDb.hasInputWriters(producer));
      // 3)
    } else if (aliasDb.isMutable(producer)) {
      REQ(!aliasDb.hasOutputWriters(consumer));
    }
  }

  if (!consumer->hasAttribute(attr::Subgraph) &&
      consumer->kind() != getTVMSymbol()) {
    consumer = SubgraphUtils::createSingletonSubgraph(consumer, getTVMSymbol());
  }
  if (producer->kind() == prim::Constant) {
    auto& subgraph = consumer->g(attr::Subgraph);
    Node* in_const = subgraph->createClone(producer, [](Value*) -> Value* {
      throw std::runtime_error("unexpected input");
    });
    subgraph->insertNode(in_const);
  } else {
    SubgraphUtils::mergeNodeIntoSubgraph(producer, consumer);
  }

  return consumer;
}
#undef REQ

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
  EliminateCommonSubexpression(graph);
  EliminateDeadCode(graph);
}

const torch::jit::Symbol& getTVMSymbol() {
  static torch::jit::Symbol tvm_sym =
      torch::jit::Symbol::fromQualString("tvm::CompilationGroup");
  return tvm_sym;
}
