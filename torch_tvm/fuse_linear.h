#pragma once

#include <torch/csrc/jit/ir.h>

using namespace torch::jit;

TORCH_API void FuseLinear(std::shared_ptr<Graph>& graph);
