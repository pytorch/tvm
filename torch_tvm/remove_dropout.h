#pragma once

#include <torch/csrc/jit/ir.h>

TORCH_API void RemoveDropout(std::shared_ptr<torch::jit::Graph>& graph);
