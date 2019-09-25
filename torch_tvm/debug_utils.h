#pragma once

#include <fstream>

#include <torch/csrc/jit/ir.h>

#include <tvm/build_module.h>
#include <tvm/lowered_func.h>

class DebugLogger {
 public:
  DebugLogger() = default;
  DebugLogger(const std::string& file_name) {
    debug_file_ = std::ofstream(file_name);
    if (!debug_file_.is_open()) {
      LOG(WARNING)
	  << "Could not open file:" << file_name
	  << ", will dump debug info to stdout\n";
    }
  }

  void printGraph(const std::shared_ptr<torch::jit::Graph>& subgraph) {
    if (debug_file_.is_open()) {
      debug_file_ <<"subgraph \n";
      debug_file_ << *subgraph << std::endl;
      debug_file_ <<"END OF Input subgraph\n";
    } else {
      std::cout <<"subgraph \n";
      std::cout << *subgraph << std::endl;
      std::cout <<"END OF Input subgraph\n";
    }
  }

  void printLoweredFuncs(tvm::runtime::Module& build_mod) {
    auto lowered_f = build_mod.GetFunction("get_lowered_funcs", false);
    tvm::Map<std::string, tvm::Array<tvm::LoweredFunc> > lowered_funcs = lowered_f();
    for (auto funcs : lowered_funcs) {
      for (auto f : funcs.second) {
	if (debug_file_.is_open()) {
	  debug_file_ << "===== lowered func=====\n";
	  debug_file_ << f->body << std::endl;
	  debug_file_ << "===== end of lowered func=====\n";
	} else {
	  std::cout << "===== lowered func=====\n";
	  std::cout << f->body << std::endl;
	  std::cout << "===== end of lowered func=====\n";
	}
      }
    }
  }

  void printASM(tvm::runtime::Module& mod) {
    if (debug_file_.is_open()) {
      debug_file_ << "======= ASM ========\n";
      debug_file_ << mod->GetSource("asm") << std::endl;
      debug_file_ << "======= END OF ASM========\n";
    } else {
      std::cout << "======= ASM ========\n";
      std::cout << mod->GetSource("asm") << std::endl;
      std::cout << "======= END OF ASM========\n";
    }
  }

 private:
  std::ofstream debug_file_;
};
