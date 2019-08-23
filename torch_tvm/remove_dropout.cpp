#include "remove_dropout.h"

using namespace torch::jit;

bool isDropoutRemovable(const Node* node) {
  const auto inputs = node->inputs();
  TORCH_INTERNAL_ASSERT(inputs.size() == 3);
  const Value* training_input = inputs[2];
  auto optional_ivalue = toIValue(training_input);
  TORCH_INTERNAL_ASSERT(optional_ivalue.has_value());
  const IValue& val = optional_ivalue.value();
  TORCH_INTERNAL_ASSERT(val.isBool());
  const bool is_training = val.toBool();
  return !is_training;
}

void RemoveDropout(std::shared_ptr<Graph>& graph) {
  auto block = graph->block();
  std::vector<Node *> deleted_nodes;

  for (auto it = block->nodes().rbegin(); it != block->nodes().rend(); it++) {
    Node* node = *it;
    if (node->kind() == aten::dropout && isDropoutRemovable(*it)) {
      // Input tensor of dropout.
      Value* input_value = node->inputs()[0];
      // Output tensor.
      Value* output_value = node->outputs()[0];
      output_value->replaceAllUsesWith(input_value);
      deleted_nodes.push_back(node);
    }
  }
  for(auto del_node : deleted_nodes) {
    del_node->destroy();
  }
}
