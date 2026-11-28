"""内置算子的数学核。

每个算子提供两份实现：

* ``single_*`` —— 单样本组合的数学（逐样本回退与参考解释器共用）；
* ``batch_*``  —— 向量化实现（一次处理所有组合的有效行）。

两者数学必须一致，测试里用参考解释器对照向量化路径（绝对/相对 1e-7）。
所有位置类错误只在“有效行”上检查；掩码位在进入本模块前已被物理剔除。
"""

from __future__ import annotations

import numpy as np

from .errors import InvalidIndexError, UnavailableResult


# ===========================================================================
# gather：用位置整数读指定轴并保留发生次数
# ===========================================================================
def single_gather(table, indices, *, combo: int, axis: int, axis_name: str):
    limit = table.shape[axis]
    for pos, idx in enumerate(indices):
        if not 0 <= int(idx) < limit:
            raise InvalidIndexError(
                index=int(idx), limit=limit, combo=combo,
                position=pos, axis=axis_name,
            )
    return np.take(table, indices.astype(np.int64), axis=axis)


def batch_gather(table, flat_indices, *, axis: int, axis_name: str,
                 positions, combo_ids):
    """向量化 gather。flat_indices/positions/combo_ids 只含有效行。"""
    limit = table.shape[axis]
    bad = (flat_indices < 0) | (flat_indices >= limit)
    if np.any(bad):
        row = int(np.flatnonzero(bad)[0])
        raise InvalidIndexError(
            index=int(flat_indices[row]), limit=limit,
            combo=int(combo_ids[row]), position=int(positions[row]),
            axis=axis_name,
        )
    return np.take(table, flat_indices.astype(np.int64), axis=axis)


def gather_vjp(gout, indices, table_shape, *, axis: int):
    grad = np.zeros(table_shape, dtype=gout.dtype)
    np.add.at(grad, indices.astype(np.int64), gout)
    return grad


# ===========================================================================
# add：位置广播遵循 NumPy；名称用于跨映射层跟踪位置
# ===========================================================================
def single_add(a, b):
    return a + b


def batch_add(a, b):
    return a + b


def _unbroadcast(g, shape):
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and g.shape[axis] != 1:
            g = g.sum(axis=axis, keepdims=True)
    return g.reshape(shape)


def add_vjp(gout, a, b):
    return _unbroadcast(gout, a.shape), _unbroadcast(gout, b.shape)


# ===========================================================================
# matmul：标准矩阵乘，可广播共享轴
# ===========================================================================
def single_matmul(a, b):
    return a @ b


def matmul_vjp(gout, a, b):
    ga = gout @ np.swapaxes(b, -1, -2)
    gb = np.swapaxes(a, -1, -2) @ gout
    return ga, gb


# ===========================================================================
# reduce：sum / mean / max；空 sum 为零，其余无定义
# ===========================================================================
def single_reduce(x, mode, *, axis: int, combo: int):
    if x.shape[axis] == 0:
        if mode == "sum":
            return np.zeros(
                tuple(s for i, s in enumerate(x.shape) if i != axis), x.dtype
            )
        raise UnavailableResult(f"组合 {combo} 空段上的 reduce {mode!r} 无定义")
    if mode == "sum":
        return x.sum(axis=axis)
    if mode == "mean":
        return x.mean(axis=axis)
    if mode == "max":
        return x.max(axis=axis)
    raise ValueError(f"未知归约 {mode!r}")


def batch_segment_reduce(flat, offsets, mode):
    """对扁平有效行按分段归约。返回 (结果数组, 不适用组合 id 集合)。

    空 sum 为零且适用；空 mean/max 不适用，对应位置填零占位并在集合中报告，
    绝不让 NaN/除零流入有效结果。
    """
    nseg = offsets.size - 1
    rest = flat.shape[1:]
    counts = np.diff(offsets)
    if flat.shape[0] == 0:
        out = np.zeros((nseg,) + rest, dtype=flat.dtype)
        return out, set() if mode == "sum" else set(range(nseg))
    if mode == "sum":
        out = np.zeros((nseg,) + rest, dtype=flat.dtype)
        np.add.at(out, np.repeat(np.arange(nseg), counts), flat)
        return out, set()
    out = np.zeros((nseg,) + rest, dtype=flat.dtype)
    empty = set(np.flatnonzero(counts == 0).tolist())
    for c in range(nseg):
        s, e = int(offsets[c]), int(offsets[c + 1])
        if s == e:
            continue
        seg = flat[s:e]
        if mode == "max":
            out[c] = seg.max(axis=0)
        elif mode == "mean":
            out[c] = seg.mean(axis=0)
        else:
            raise ValueError(f"未知归约 {mode!r}")
    return out, empty


def reduce_vjp_flat(gout_per_combo, flat, offsets, mode):
    """分段归约的伴随，按扁平有效行返回；空段没有行因此无梯度可写。

    max 的并列项在实际最大项之间均分梯度（《交换约定》）。
    """
    grad = np.zeros_like(flat)
    counts = np.diff(offsets)
    for c in range(offsets.size - 1):
        s, e = int(offsets[c]), int(offsets[c + 1])
        if s == e:
            continue
        g = gout_per_combo[c]
        seg = flat[s:e]
        if mode == "sum":
            grad[s:e] = g
        elif mode == "mean":
            grad[s:e] = g / counts[c]
        elif mode == "max":
            mx = seg.max(axis=0)
            mask = (seg == mx)
            grad[s:e] = g * mask / mask.sum(axis=0)
        else:
            raise ValueError(f"未知归约 {mode!r}")
    return grad


# ===========================================================================
# scatter_add：贡献相加而非覆盖；重复位置按各次发生计算
# ===========================================================================
def single_scatter_add(rows, indices, size, *, combo: int):
    limit = size
    for pos, idx in enumerate(indices):
        if not 0 <= int(idx) < limit:
            raise InvalidIndexError(
                index=int(idx), limit=limit, combo=combo,
                position=pos, axis="lookup",
            )
    out_shape = list(rows.shape)
    out_shape[0] = size
    out = np.zeros(out_shape, dtype=rows.dtype)
    np.add.at(out, indices.astype(np.int64), rows)
    return out


def batch_scatter_add(flat_rows, flat_indices, combo_ids, segments,
                      size, *, axis: int):
    """向量化 scatter_add。输出带完整（多维）映射网格，空组合贡献为零。

    输入行不含 lookup 维；输出在逻辑位置 ``axis`` 开长度 ``size`` 的 lookup
    轴，其余逻辑轴逐行保留。重复位置的各次发生相加；组合 id 解码成多维
    网格下标，因此嵌套顺序不改变身份。
    """
    limit = size
    bad = (flat_indices < 0) | (flat_indices >= limit)
    if np.any(bad):
        row = int(np.flatnonzero(bad)[0])
        raise InvalidIndexError(
            index=int(flat_indices[row]), limit=limit,
            combo=int(combo_ids[row]), position=-1, axis="lookup",
        )
    in_trailing = list(flat_rows.shape[1:])
    out_trailing = list(in_trailing)
    out_trailing.insert(axis, size)
    ncombos = int(np.prod(segments.grid))
    combo_span = int(np.prod(out_trailing))
    in_span = int(np.prod(in_trailing))
    out = np.zeros(
        tuple(segments.grid) + tuple(out_trailing), dtype=flat_rows.dtype
    )
    flat_out = out.reshape(ncombos, combo_span)

    # 行优先步长
    in_strides = _row_strides(in_trailing)
    out_strides = _row_strides(out_trailing)
    # 输入列坐标 -> 输出块内列坐标：在输出的 axis 处插入 lookup 槽，
    # 输入第 ax 轴映射到输出第 (ax if ax<axis else ax+1) 轴。
    col_in = np.arange(in_span)
    coords_col = np.unravel_index(col_in, in_trailing)
    out_cols = np.zeros(in_span, dtype=np.int64)
    for ax in range(len(in_trailing)):
        out_ax = ax if ax < axis else ax + 1
        out_cols += coords_col[ax] * out_strides[out_ax]
    slot = flat_indices.astype(np.int64) * out_strides[axis]
    target_cols = slot[:, None] + out_cols[None, :]

    combo_coords = np.array(
        [segments.decode(int(cid)) for cid in combo_ids], dtype=np.int64
    )
    combo_flat = np.ravel_multi_index(
        [combo_coords[:, j] for j in range(len(segments.grid))],
        segments.grid,
    )
    rows2d = flat_rows.reshape(flat_rows.shape[0], in_span)
    np.add.at(
        flat_out,
        (np.repeat(combo_flat, in_span), target_cols.reshape(-1)),
        rows2d.reshape(-1),
    )
    return out


def _row_strides(shape):
    strides = [1] * len(shape)
    for ax in range(len(shape) - 2, -1, -1):
        strides[ax] = strides[ax + 1] * shape[ax + 1]
    return strides


def scatter_vjp(gout, idx_t, segments, *, axis: int):
    """scatter_add 的伴随：按组合网格坐标 + 位置整数取回每行梯度（保留重复）。"""
    grid_ndim = len(segments.grid)
    coords = np.array(
        [segments.decode(int(cid)) for cid in segments.combo_ids], dtype=np.int64
    )
    indexer = tuple(coords[:, j] for j in range(grid_ndim)) + (
        idx_t.data.astype(np.int64),
    )
    return gout[indexer]
