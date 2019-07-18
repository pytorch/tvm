#include "compiler.h"
#include "operators.h"

#include <ATen/DLConvertor.h>
#include <torch/csrc/jit/constants.h>
#include <torch/csrc/jit/interpreter.h>
#include <limits>

using namespace torch::jit;

tvm::relay::Var TVMCompiler::convertToRelay(Value* val, TVMContext ctx) {
  auto optional_ivalue = toIValue(val);

  tvm::Array<HalideIR::Expr> sizes;
  if (auto ptt = val->type()->cast<ProfiledTensorType>())
  {

    auto csizes = ptt->sizes().concrete_sizes();
    TORCH_INTERNAL_ASSERT(csizes.has_value());
    for (const auto& size : *csizes)
    {
      sizes.push_back(HalideIR::Expr(static_cast<int32_t>(size)));
    }
  } else if (optional_ivalue.has_value()) {
    // TODO: inferTypeFrom should eventually create ProfiledTensorTypes
    val->inferTypeFrom(optional_ivalue.value().toTensor());
    auto pt_t = val->type()->expect<CompleteTensorType>();
    for (const auto& size : pt_t->sizes()) {
      sizes.push_back(HalideIR::Expr(static_cast<int32_t>(size)));
    }
  }
  else {
    TORCH_INTERNAL_ASSERT(0);
  }

  // TODO: support non-float tensors
  auto t = tvm::relay::TensorTypeNode::make(sizes, ::tvm::Float(32));
  auto v = tvm::relay::VarNode::make(
      val->debugName() +
          std::to_string(reinterpret_cast<std::uintptr_t>(val)),
      t);
  return v;
}

tvm::relay::Expr TVMCompiler::convertToRelay(
    const IValue& val,
    TVMContext ctx) {
  // All doubles are converted to floats
  if (val.isDouble()) {
    auto x = tvm::runtime::NDArray::Empty(
        {}, tvm::runtime::String2TVMType("float32"), ctx);
    auto d = val.toDouble();
    AT_CHECK(d <= std::numeric_limits<float>::max());
    AT_CHECK(d >= std::numeric_limits<float>::lowest());
    auto f = static_cast<float>(d);
    reinterpret_cast<float*>(x->data)[0] = f;
    auto v = tvm::relay::ConstantNode::make(x);
    return v;
  }
  // All Ints are converted to int32, which may overflow
  if (val.isInt()) {
    auto x = tvm::runtime::NDArray::Empty(
        {}, tvm::runtime::String2TVMType("int32"), ctx);
    auto l = val.toInt();
    AT_CHECK(l <= std::numeric_limits<int32_t>::max());
    AT_CHECK(l >= std::numeric_limits<int32_t>::lowest());
    reinterpret_cast<int32_t*>(x->data)[0] = l;
    auto v = tvm::relay::ConstantNode::make(x);
    return v;
  }
  if (val.isBool()) {
    auto x = tvm::runtime::NDArray::Empty(
        {}, tvm::runtime::String2TVMType("bool"), ctx);
    reinterpret_cast<bool*>(x->data)[0] = val.toBool();
    auto v = tvm::relay::ConstantNode::make(x);
    return v;
  }
  // TODO Add None type to Relay
  // HACK sentinel value used for None type
  if (val.isNone()) {
    auto x = tvm::runtime::NDArray::Empty(
        {}, tvm::runtime::String2TVMType("uint64"), ctx);
    reinterpret_cast<uint64_t*>(x->data)[0] = getNoneSentinel();
    auto v = tvm::relay::ConstantNode::make(x);
    return v;
  }
  if (val.isIntList()) {
    tvm::Array<tvm::relay::Expr> tuple_elems;
    for (const auto& elem : val.toIntList()) {
      auto x = tvm::runtime::NDArray::Empty(
          {}, tvm::runtime::String2TVMType("int32"), ctx);
      AT_CHECK(elem <= std::numeric_limits<int32_t>::max());
      AT_CHECK(elem >= std::numeric_limits<int32_t>::lowest());
      reinterpret_cast<int32_t*>(x->data)[0] = elem;
      auto v = tvm::relay::ConstantNode::make(x);
      tuple_elems.push_back(v);
    }
    return tvm::relay::TupleNode::make(tuple_elems);
  }
  AT_CHECK(
      0, "Cannot convert value ", val, " to Relay yet.  Please file a bug.\n");
}

tvm::relay::Function TVMCompiler::convertToRelay(
    std::shared_ptr<Graph> subgraph,
    TVMContext ctx,
    std::vector<Value*>* input_values) {
  std::unordered_map<Value*, tvm::relay::Expr> value_map;
  tvm::Array<tvm::relay::Var> input_vars;
  for (const auto& input : subgraph->inputs()) {
    TORCH_INTERNAL_ASSERT(input->type()->cast<ProfiledTensorType>());
    auto v = convertToRelay(input, ctx);
    input_vars.push_back(v);
    if (input_values) {
      input_values->emplace_back(input);
    }
    value_map[input] = v;
  }

  auto frontier = subgraph->inputs().vec();
  // TODO error handle incorrectly formed graphs (not dominated by frontier)
  while (frontier.size()) {
    std::vector<Value*> new_frontier = {};
    for (const auto& value : frontier) {
      auto uses = value->uses();
      for (const auto& use : uses) {
        tvm::Array<tvm::relay::Expr> relay_inputs;
        auto skip_user = false;
        for (const auto& input : use.user->inputs()) {
          if (value_map.find(input) == value_map.end()) {
            // We may be dealing with a constant, handle that here
            auto optional_ivalue = toIValue(input);
            if (!optional_ivalue.has_value()) {
              skip_user = true;
              break;
            } else {
              if (optional_ivalue.value().isTensor()) {
                if (input_values) {
                  input_values->emplace_back(input);
                }
                auto input_var = convertToRelay(input, ctx);
                input_vars.push_back(input_var);
                value_map[input] = input_var;
              } else {
                value_map[input] = convertToRelay(optional_ivalue.value(), ctx);
              }
            }
          }
          relay_inputs.push_back(value_map[input]);
        }
        if (skip_user) {
          continue;
        }
        // Things like prim::Return
        if (use.user->outputs().size() < 1) {
          continue;
        }
        // if there are 2+ outputs, getOperator returns a tuple
        if (use.user->outputs().size() == 1) {
          value_map[use.user->output()] = getOperator(use.user, relay_inputs);
          new_frontier.emplace_back(use.user->output());
        } else {
          auto tuple = getOperator(use.user, relay_inputs);
          int index = 0;
          for (const auto& output : use.user->outputs()) {
            auto n = tvm::make_node<tvm::relay::TupleGetItemNode>();
            n->tuple = tuple;
            n->index = index;
            value_map[output] = tvm::relay::TupleGetItem(n);
            index++;
            new_frontier.emplace_back(output);
          }
        }
      }
    }
    frontier = new_frontier;
  }

  tvm::NodePtr<tvm::relay::TupleNode> n =
      tvm::make_node<tvm::relay::TupleNode>();
  tvm::Array<tvm::relay::Expr> fields;
  for (const auto& sg_output : subgraph->outputs()) {
    AT_ASSERT(value_map.find(sg_output) != value_map.end());
    fields.push_back(value_map[sg_output]);
  }
  n->fields = std::move(fields);
  auto output = tvm::relay::Tuple(n);

  tvm::Array<tvm::relay::Var> free_vars = tvm::relay::FreeVars(output);
  AT_CHECK(
      free_vars.size() <= input_vars.size(),
      "Determined ",
      free_vars.size(),
      " free vars but only ",
      input_vars.size(),
      " inputs");

  return tvm::relay::FunctionNode::make(
      input_vars, output, tvm::relay::Type(), {});
}

TVMCompiler::TVMCompiler(
    const Node* node,
    int opt_level,
    bool strict,
    std::string device_type,
    std::string device,
    std::string host)
    : opt_level_(opt_level),
      strict_(strict),
      device_type_(device_type),
      device_(device),
      host_(host) {
  if (device_type_ == "gpu") {
    ctx_.device_type = kDLGPU;
  } else {
    ctx_.device_type = kDLCPU;
  }
  ctx_.device_id = 0;
  subgraph_ = node->g(attr::Subgraph);
  auto pfb = tvm::runtime::Registry::Get("relay.build_module._BuildModule");
  AT_ASSERT(pfb);
  build_mod_ = (*pfb)();
}

void TVMCompiler::run(Stack& stack) {
  std::unordered_map<Value*, IValue> value_to_ivalue;
  int num_inputs = subgraph_->inputs().size();
  at::ArrayRef<IValue> inputs = last(stack, num_inputs);

  for (auto i = 0; i < inputs.size(); ++i) {
    auto value_input = subgraph_->inputs()[i];
    value_to_ivalue[value_input] = inputs[i];
  }

  if (!cache_ || (cache_ && (*cache_).invalid)) {
    for (auto& kv : value_to_ivalue) {
      // TODO: convince Fuser to NOT create TVMCompilationGroups
      // if ANY of subgraph inputs weren't profiled
      TORCH_INTERNAL_ASSERT(kv.first->type()->cast<ProfiledTensorType>());
    }
    // bail out mechanism: try to convert to Relay, if it fails to convert the
    // graph by any reason(i.e. op difference), depend on the user preference,
    // either throw or fall back to the JIT interpreter for execution
    cache_ = TVMObject {};
    tvm::relay::Function tvm_func;
    try {
      tvm_func = convertToRelay(subgraph_, ctx_, &(*cache_).input_values);
      // we compiled the subgraph successfully
      (*cache_).invalid = false;
    } catch (const std::exception& e) {
      (*cache_).invalid = true;
      if (strict_) {
        AT_ERROR(
            "Pytorch TVM: fail to convert to relay, exception: ", e.what());
      }

      LOG(WARNING)
          << "Pytorch TVM: fail to convert to relay, exception: "
          << e.what() << "\n";
    }

    if ((*cache_).invalid)
    {
      LOG(WARNING) << "Falling back to JIT";
      InterpreterState(Code(subgraph_)).run(stack);
      return;
    }

    auto build_f = build_mod_.GetFunction("build", false);
    auto json_f = build_mod_.GetFunction("get_graph_json", false);
    auto mod_f = build_mod_.GetFunction("get_module", false);
    tvm::Map<tvm::Integer, tvm::Target> target_map = {
        {ctx_.device_type, tvm::Target::Create(device_)}};
    build_f(tvm_func, target_map, tvm::Target::Create(host_));
    std::string json = json_f();
    tvm::runtime::Module mod = mod_f();
    auto pfr = tvm::runtime::Registry::Get("tvm.graph_runtime.create");
    AT_ASSERT(pfr);

    tvm::runtime::Module run_mod =
        (*pfr)(json, mod, (int)ctx_.device_type, (int)ctx_.device_id);
    (*cache_).set_input = run_mod.GetFunction("set_input_zero_copy", false);
    (*cache_).kernel = run_mod.GetFunction("run", false);
    (*cache_).get_output = run_mod.GetFunction("get_output", false);
    auto get_num_outputs = run_mod.GetFunction("get_num_outputs", false);
    int n = get_num_outputs();
    AT_CHECK(
        subgraph_->outputs().size() == n,
        "Compiled subgraph with mismatching num outputs");

  }

  // setting arguments
  for (auto i = 0; i < (*cache_).input_values.size(); ++i) {
    auto* value = (*cache_).input_values[i];
    if (!value_to_ivalue.count(value)) {
      auto optional_ivalue = toIValue(value);
      AT_ASSERT(optional_ivalue.has_value());
      value_to_ivalue[value] = optional_ivalue.value();
    }
    auto ivalue = value_to_ivalue.at((*cache_).input_values[i]);
    auto tensor = ivalue.toTensor().to(at::kFloat);
    auto dl_tensor = at::toDLPack(tensor);
    (*cache_).set_input(i, tvm::runtime::NDArray::FromDLPack(dl_tensor));
  }

  (*cache_).kernel();

  // clean the stack and add outputs to the stack
  drop(stack, num_inputs);
  int i = 0;
  for (const auto& output : subgraph_->outputs()) {
    tvm::runtime::NDArray ret_val = (*cache_).get_output(i);
    auto dl_tensor = ret_val.ToDLPack();
    auto tensor = at::fromDLPack(dl_tensor);
    auto var = torch::autograd::make_variable(tensor);
    stack.push_back(IValue(var));
    i++;
  }
}
