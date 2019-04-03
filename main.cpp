#include <torch/csrc/jit/pass_manager.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using namespace torch::jit;

void tvmPass(std::shared_ptr<Graph>&g) {
  std::cerr << "This pass works!\n";
  return;
}

PYBIND11_MODULE(torch_tvm, m) {
  RegisterPass p(tvmPass);
  m.doc() = "This module does nothing but register a TVM backend.";
}
