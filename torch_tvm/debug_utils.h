#pragma once

#include <fstream>
#include <thread>
#include <sstream>

#include <torch/csrc/jit/ir.h>

#include <tvm/build_module.h>
#include <tvm/lowered_func.h>

class DebugLogger {
 public:
  // Delete copy constructor and assign.
  DebugLogger(const DebugLogger&) = delete;
  DebugLogger operator=(const DebugLogger&) = delete;
  DebugLogger();

  void printGraph(const std::shared_ptr<torch::jit::Graph>& subgraph); 

  void printLoweredFuncs(tvm::runtime::Module& build_mod); 

  void printASM(tvm::runtime::Module& mod); 

 private:
  std::ofstream debug_file_;
};

DebugLogger& getDebugLogger(); 
