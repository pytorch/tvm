import tvm
from tvm import autotvm
import topi
from topi import nn
from topi.util import get_const_tuple
from topi.x86.dense import _declaration_dense_reshape, \
        _declaration_dense_nopack_reshape, \
        _declaration_dense_nopack, \
        _declaration_dense_pack, \
        _default_dense_nopack_config

@autotvm.register_topi_compute(nn.dense, "cpu", "direct", override=True)
def declare_customized_dense(cfg, data, weight, bias=None, out_dtype=None):
    if len(data.shape) == 2:
        batch, _ = get_const_tuple(data.shape)
    elif len(data.shape) > 2:
        batch = get_const_tuple(data.shape)[0]
        pack_weights = (batch > 16)
        return _declaration_dense_reshape(cfg, data, weight, \
                bias, out_dtype, pack_weights)

    # For small batch sizes, don't pack weight into cache-friendly layout
    # because of overhead in packing and limited reuse from batch dimension
    # TODO(icemelon9): use a more systematic way to determine which schedule to use
    if batch <= 16:
        return _declaration_dense_nopack(cfg, data, weight, bias, out_dtype)
    return _declaration_dense_pack(cfg, data, weight, bias, out_dtype)

@autotvm.register_topi_compute(nn.dense, "cpu", "custom_direct_reshape")
def _declaration_dense_reshape(cfg, data, weight, bias=None, out_dtype=None, pack_weights=True):
    if out_dtype is None:
        out_dtype = data.dtype
    total_size = 1
    for s in data.shape:
        total_size *= s
    new_data = topi.reshape(data, newshape=(total_size//data.shape[-1], data.shape[-1]))
    batch, in_dim = get_const_tuple(new_data.shape)
    out_dim, _ = get_const_tuple(weight.shape)
    # create tuning space
    cfg.define_split("tile_y", batch, num_outputs=3)
    cfg.define_split("tile_x", out_dim, num_outputs=3)
    cfg.define_split("tile_k", in_dim, num_outputs=2)

    if pack_weights:
        if cfg.is_fallback:
            _default_dense_pack_config(cfg, batch, out_dim, in_dim)
        packw_bn = cfg["tile_x"].size[-1]
        k = tvm.reduce_axis((0, in_dim), name="k")
        packw_shape = (out_dim // packw_bn, in_dim, packw_bn)
        packw = tvm.compute(packw_shape,
                            lambda z, y, x: weight[z * packw_bn + x, y], name="packed_weight")

        C = tvm.compute((batch, out_dim),
                        lambda y, x: tvm.sum(
                            new_data[y, k].astype(out_dtype) *
                            packw[x // packw_bn, k, x % packw_bn].astype(out_dtype),
                            axis=k),
                        tag="dense_pack")
    else:
        if cfg.is_fallback:
            _default_dense_nopack_config(cfg, batch, out_dim, in_dim)
        vec = cfg["tile_k"].size[-1]
        k = tvm.reduce_axis((0, in_dim // vec), "k")
        CC = tvm.compute((batch, out_dim, vec),
                         lambda z, y, x: tvm.sum(
                             new_data[z, k * vec + x].astype(out_dtype) *
                             weight[y, k * vec + x].astype(out_dtype), axis=k))
        kk = tvm.reduce_axis((0, vec), "kk")
        C = tvm.compute((batch, out_dim),
                        lambda y, x: tvm.sum(CC[y, x, kk], axis=kk),
                        tag="dense_nopack")
    if bias is not None:
        C = tvm.compute((batch, out_dim), lambda i, j: C[i, j] + bias[j].astype(out_dtype),
                        tag=tag.BROADCAST)
    new_shape = get_const_tuple(data.shape[:-1])
    new_shape = new_shape + (out_dim,)
    C = topi.reshape(C, newshape=new_shape)
    return C

