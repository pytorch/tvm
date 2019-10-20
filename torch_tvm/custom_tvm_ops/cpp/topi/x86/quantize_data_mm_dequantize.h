#pragma once

#include <llvm/IR/Function.h>
#include "topi/detail/array_utils.h"
#include "topi/detail/fuse.h"
#include "topi/tags.h"
#include "tvm/buffer.h"
#include "tvm/operation.h"
#include "tvm/tensor_intrin.h"

namespace topi {
using namespace tvm;

namespace x86 {

// for avx2
TensorIntrin dot_1x4x16_int8_int8_int32_avx2() {
  tvm::Tensor data = tvm::placeholder({4}, UInt(8), "data");
  tvm::Tensor kernel = tvm::placeholder({16, 4}, Int(8), "kernel");
  auto k = tvm::reduce_axis(tvm::Range{0, 4}, "k");
  auto C = tvm::compute(
      {16},
      [&](Var i) {
        return tvm::sum(
            tvm::cast(Int(32), data(k)) * tvm::cast(Int(32), kernel(i, k)),
            {k});
      },
      "tensor",
      "dense");
  auto a_buf = BufferNode::make(
      /*Var ptr*/ Var("a_buf", Handle()),
      /*Type dtype*/ UInt(8),
      /*Array<Expr> shape*/ {4},
      /*Array<Expr> strides*/ {1},
      /*Expr elem_offset*/ Var("a_buf_elem_offset"),
      /*std::string name*/ "a_buf",
      /*std::string scope*/ "",
      /*int data_alignment*/ -1,
      /*int offset_factor*/ 1,
      /*BufferType buffer_type*/ kDefault);
  auto b_buf = BufferNode::make(
      Var("b_buf", Handle()),
      Int(8),
      {16, 4},
      {Var("ldw"), 1},
      Var("b_buf_elem_offset"),
      "b_buf",
      "",
      -1,
      1,
      kDefault);
  auto c_buf = BufferNode::make(
      Var("c_buf", Handle()),
      Int(32),
      {16},
      {1},
      Var("c_buf_elem_offset"),
      "c_buf",
      "",
      -1,
      1,
      kDefault);
  Expr a_int8 = a_buf.vload({0}, UInt(8, 4));
  Expr re_int32 = tvm::reinterpret(Int(32), a_int8);
  Expr vec_ai32 = tvm::cast(Int(32, 8), re_int32);
  Expr vec_a = tvm::reinterpret(Int(8, 32), vec_ai32);
  Expr vec_b_0 = b_buf.vload({0, 0}, Int(8, 32));
  Expr vec_b_1 = b_buf.vload({8, 0}, Int(8, 32));
  Expr vec_one = make_const(Int(16, 16), 1);
  Expr vec_zero = make_const(UInt(32), 0);

  constexpr auto pair = "llvm.x86.avx2.pmadd.ub.sw";
  constexpr auto quad = "llvm.x86.avx2.pmadd.wd";

  Expr llvm_pair =
      make_const(UInt(32), llvm::Function::lookupIntrinsicID(pair));
  Expr llvm_quad =
      make_const(UInt(32), llvm::Function::lookupIntrinsicID(quad));

  Expr pair_reduction_0 = ir::Call::make(
      /*Type type*/ Int(16, 16),
      /*std::string name*/ "llvm_intrin",
      /*Array<Expr> args*/ {llvm_pair, vec_zero, vec_a, vec_b_0},
      /*CallType call_type*/ ir::Call::PureIntrinsic
      /*FunctionRef func = FunctionRef()*/
      /*int value_index = 0*/
  );
  Expr quad_reduction_0 = ir::Call::make(
      Int(32, 8),
      "llvm_intrin",
      {llvm_quad, vec_zero, pair_reduction_0, vec_one},
      ir::Call::PureIntrinsic);
  Expr pair_reduction_1 = ir::Call::make(
      Int(16, 16),
      "llvm_intrin",
      {llvm_pair, vec_zero, vec_a, vec_b_1},
      ir::Call::PureIntrinsic);
  Expr quad_reduction_1 = ir::Call::make(
      Int(32, 8),
      "llvm_intrin",
      {llvm_quad, vec_zero, pair_reduction_1, vec_one},
      ir::Call::PureIntrinsic);

  Stmt reduce_init = c_buf.vstore({0}, make_const(Int(32, 16), 0));
  Stmt body_0 = c_buf.vstore({0}, quad_reduction_0);
  Stmt body_1 = c_buf.vstore({8}, quad_reduction_1);
  Stmt body = ir::Block::make(body_0, body_1);
  Stmt reduce_update_0 =
      c_buf.vstore({0}, quad_reduction_0 + c_buf.vload({0}, Int(32, 8)));
  Stmt reduce_update_1 =
      c_buf.vstore({8}, quad_reduction_1 + c_buf.vload({8}, Int(32, 8)));
  Stmt reduce_update = ir::Block::make(reduce_update_0, reduce_update_1);

  return TensorIntrinNode::make(
      /*std::string name*/ "tensor_intrin",
      /*Operation op*/ C->op,
      /*Array<Tensor> inputs*/ C->op->InputTensors(),
      /*Array<Buffer> buffers*/ {a_buf, b_buf, c_buf},
      /*Array<Var> scalar_params*/ {},
      /*Stmt body*/ body,
      /*Stmt body*/ reduce_init,
      /*Stmt body*/ reduce_update);
}

// for avx512
TensorIntrin dot_16x1x16_int8_int8_int32_avx512() {
  tvm::Tensor data = tvm::placeholder({4}, UInt(8), "data");
  tvm::Tensor kernel = tvm::placeholder({16, 4}, Int(8), "kernel");
  auto k = tvm::reduce_axis(tvm::Range{0, 4}, "k");
  auto C = tvm::compute(
      {16},
      [&](Var i) {
        return tvm::sum(
            tvm::cast(Int(32), data(k)) * tvm::cast(Int(32), kernel(i, k)),
            {k});
      },
      "tensor",
      "dense");
  auto a_buf = BufferNode::make(
      Var("a_buf", Handle()),
      UInt(8),
      {4},
      {1},
      Var("a_buf_elem_offset"),
      "a_buf",
      "",
      -1,
      1,
      kDefault);
  auto b_buf = BufferNode::make(
      Var("b_buf", Handle()),
      Int(8),
      {16, 4},
      {Var("ldw"), 1},
      Var("b_buf_elem_offset"),
      "b_buf",
      "",
      -1,
      1,
      kDefault);
  auto c_buf = BufferNode::make(
      Var("c_buf", Handle()),
      Int(32),
      {16},
      {1},
      Var("c_buf_elem_offset"),
      "c_buf",
      "",
      -1,
      1,
      kDefault);

  Expr a_int8 = a_buf.vload({0}, UInt(8, 4));
  Expr vec_b = b_buf.vload({0, 0}, Int(8, 64));
  Expr vec_one = make_const(Int(16, 32), 1);
  Expr re_int32 = tvm::reinterpret(Int(32), a_int8);
  Expr vec_ai32 = tvm::cast(Int(32, 16), re_int32);
  Expr vec_a = tvm::reinterpret(Int(8, 64), vec_ai32);
  Expr vec_zero = make_const(UInt(32), 0);
  constexpr auto pair = "llvm.x86.avx512.pmaddubs.w.512";
  constexpr auto quad = "llvm.x86.avx512.pmaddw.d.512";
  Expr llvm_pair =
      make_const(UInt(32), llvm::Function::lookupIntrinsicID(pair));
  Expr llvm_quad =
      make_const(UInt(32), llvm::Function::lookupIntrinsicID(quad));
  Expr pair_reduction = ir::Call::make(
      Int(16, 32),
      "llvm_intrin",
      {llvm_pair, vec_zero, vec_a, vec_b},
      ir::Call::PureIntrinsic);
  Expr quad_reduction = ir::Call::make(
      Int(32, 16),
      "llvm_intrin",
      {llvm_quad, vec_zero, pair_reduction, vec_one},
      ir::Call::PureIntrinsic);

  Stmt reduce_init = c_buf.vstore({0}, make_const(Int(32, 16), 0));
  Stmt body = c_buf.vstore({0}, quad_reduction);
  Stmt reduce_update =
      c_buf.vstore({0}, quad_reduction + c_buf.vload({0}, Int(32, 16)));

  return TensorIntrinNode::make(
      "tensor_intrin",
      C->op,
      C->op->InputTensors(),
      {a_buf, b_buf, c_buf},
      {},
      body,
      reduce_init,
      reduce_update);
}

inline Schedule schedule_quantized_mm_dequantize(
    const Target& target,
    const Array<Tensor>& outs) {
  Array<Operation> out_ops;
  for (auto t : outs) {
    out_ops.push_back(t->op);
  }
  auto s = create_schedule(out_ops);

  auto _schedule_quantized_mm = [&](const Tensor& input) {
    auto axis = s[input]->op.as<ComputeOpNode>()->axis;
    CHECK_EQ(axis.size(), 2);
    auto y = axis[0];
    auto x = axis[1];
    auto reduce_axis = s[input]->op.as<ComputeOpNode>()->reduce_axis;
    CHECK_EQ(reduce_axis.size(), 1);
    auto k = reduce_axis[0];
    auto x_dim_size = input->shape[1];
    if (*as_const_int(x_dim_size) >= 16) {
      IterVar xo, xi;
      IterVar ko, ki;
      s[input].split(x, 16, &xo, &xi);
      s[input].split(k, 4, &ko, &ki);
      s[input].reorder({xo, ko, y, xi, ki});
      s[input].unroll(y);
      if (target->options_array[0].as<tvm::ir::StringImm>()->value ==
          "-mcpu=skylake-avx512") {
        auto pc = dot_16x1x16_int8_int8_int32_avx512();
        s[input].tensorize(xi, pc);
      } else {
        auto pc = dot_1x4x16_int8_int8_int32_avx2();
        s[input].tensorize(xi, pc);
      }
    } else {
      s[input].reorder({y, x});
      s[input].unroll(y);
      s[input].vectorize(x);
    }
  };

  auto _schedule_mm_dequantize = [&](const Tensor& output) {
    for (auto tensor : output->op->InputTensors()) {
      if (tensor->op->tag.rfind("quantized_mm", 0) == 0) {
        _schedule_quantized_mm(tensor);
      }
    }
  };

  std::function<void(Operation)> traverse;
  traverse = [&](const Operation& op) {
    // Inline all one-to-one-mapping operators except the last stage (output)
    if (is_broadcast(op->tag)) {
      if (!detail::contains(s->outputs, op)) {
        s[op].compute_inline();
      }
      for (auto tensor : op->InputTensors()) {
        if (tensor->op->InputTensors().size() > 0) {
          traverse(tensor->op);
        }
      }
    }
    if (op->tag.rfind("mm_dequantize", 0) == 0) {
      auto output = op.output(0);
      _schedule_mm_dequantize(output);
    }
  };

  traverse(outs[0]->op);
  return s;
}
} // namespace x86
} // namespace topi
