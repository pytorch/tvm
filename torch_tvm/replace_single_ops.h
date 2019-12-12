#pragma once

#include <torch/csrc/jit/ir.h>

TORCH_API void ReplaceSingleOps(std::shared_ptr<torch::jit::Graph>& graph);
