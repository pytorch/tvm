#include <torch/csrc/jit/passes/alias_analysis.h>
#include <torch/csrc/jit/passes/graph_fuser.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

void FuseSupportedOps(std::shared_ptr<torch::jit::Graph> graph);

extern const torch::jit::Symbol tvm_sym;
