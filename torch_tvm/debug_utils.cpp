#include "debug_utils.h"

DebugLogger& getDebugLogger() {
  static thread_local DebugLogger debug_logger;
  return debug_logger;
}

DebugLogger::DebugLogger() {
  std::ostringstream ss;
  ss << std::this_thread::get_id();
  std::string file_name = "/tmp/debug_output_" +
			  ss.str() +
			  ".txt";
  debug_file_ = std::ofstream(file_name, std::ios::out);
  if (!debug_file_.is_open()) {
    LOG(WARNING)
	<< "Could not open file:" << file_name
	<< ", will dump debug info to stdout\n";
  }
}

void DebugLogger::printGraph(
    const std::shared_ptr<torch::jit::Graph>& subgraph) {
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

void DebugLogger::printLoweredFuncs(tvm::runtime::Module& build_mod) {
  tvm::runtime::PackedFunc lowered_f;
  try {
    lowered_f = build_mod.GetFunction("get_lowered_funcs", false);
  } catch (const std::exception& e) {
    LOG(WARNING) << "TVM runtime is not exposed lowered_funcs:"
                 << e.what() << std::endl;
    return;
  }
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

void DebugLogger::printASM(tvm::runtime::Module& mod) {
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

