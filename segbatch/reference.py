"""逐样本参考解释器（oracle）。

完全按“每个映射组合单独跑一张单样本算子图”的语义实现，刻意不使用引擎的
向量化路径，只共用 :mod:`segbatch.ops` 里的单样本数学核与插件 ``fn/vjp``。
引擎的向量化结果必须与本解释器在 1e-7（绝对/相对）内一致。

参考解释器同时给出前向与反向，因此能独立验证：
* 向量化前向 == 逐样本前向；
* 引擎反传的共享/副本参数梯度 == 逐样本梯度按有效域合并。
"""

from __future__ import annotations

import numpy as np

from . import ops
from .segments import RaggedSegments

_RAGGED_OPS = {"gather", "add", "matmul", "plugin"}


def _neg_axis(axes, axis):
    i = int(axis)
    if i < 0:
        i += len(axes)
    return i


class ReferenceInterpreter:
    def __init__(self, graph, plugins, inputs, segments: RaggedSegments):
        self.graph = graph
        self.plugins = plugins
        self.inputs = inputs
        self.seg = segments
        self._tables = []
        self._tapes = []
        # 中间张量的逻辑轴声明（节点名 -> 轴元组），从引擎输出元数据复用
        self._axes = {name: t.axes for name, t in inputs.items()}

    # ------------------------------------------------------------------
    def _resolve_axis(self, tname, axis):
        if tname in self.inputs:
            return self.inputs[tname].resolve_axis(axis)
        return self._axes[tname].index(axis) if isinstance(axis, str) \
            else _neg_axis(self._axes[tname], axis)

    def _combo_inputs(self, cid):
        return {name: t.combo_view(cid) for name, t in self.inputs.items()}

    # ------------------------------------------------------------------
    def run_forward(self):
        seg = self.seg
        dense_vals: dict[str, list] = {}
        ragged_vals: dict[str, list] = {}
        for cid in range(seg.combos):
            table = self._combo_inputs(cid)
            tape = []
            for node in self.graph.nodes:
                out = self._run_node(node, table, cid, tape)
                table[node.name] = out
            self._tables.append(table)
            self._tapes.append(tape)
            for node in self.graph.nodes:
                val = table[node.name]
                if node.op in _RAGGED_OPS:
                    ragged_vals.setdefault(node.name, []).append(val)
                else:
                    dense_vals.setdefault(node.name, []).append(val)

        result = {}
        for name, vals in ragged_vals.items():
            chunks = [v for cid, v in enumerate(vals)
                      if seg.segment_length(cid) > 0]
            sample = next((v for v in vals if v.size), vals[0])
            flat = (np.concatenate(chunks, axis=0) if chunks
                    else np.empty((0,) + sample.shape[1:]))
            result[name] = ("ragged", flat)
        for name, vals in dense_vals.items():
            sample = vals[0]
            result[name] = (
                "dense",
                np.stack(vals, axis=0).reshape(seg.grid + sample.shape),
            )
        return result

    def _run_node(self, node, table, cid, tape):
        ins = [table[i] for i in node.inputs]
        if node.op == "gather":
            axis = self._resolve_axis(node.inputs[0], node.axis)
            out = ops.single_gather(
                ins[0], ins[1], combo=cid, axis=axis, axis_name=node.axis
            )
            new_axes = list(self._axes[node.inputs[0]])
            new_axes[axis] = "sequence"
            self._axes[node.name] = tuple(new_axes)
            tape.append((node, ins, out, {"axis": axis}))
            return out
        if node.op == "add":
            out = ops.single_add(ins[0], ins[1])
            self._axes[node.name] = self._merge_axes(node.inputs)
            tape.append((node, ins, out, {}))
            return out
        if node.op == "matmul":
            out = ops.single_matmul(ins[0], ins[1])
            self._axes[node.name] = (
                self._axes[node.inputs[0]][:-1]
                + (self._axes[node.inputs[1]][-1],)
            )
            tape.append((node, ins, out, {}))
            return out
        if node.op == "plugin":
            spec = self.plugins.get(node.plugin)
            q = ins[0]
            if q.shape[0] == 0:
                # 与引擎同一条硬约定：空段/全掩码轨迹不调用只对合法轨迹
                # 有定义的插件，更不靠掩码事后吞错。产出空行占位。
                out = np.empty((0,) + q.shape[1:], dtype=q.dtype)
                tape.append((node, ins, out,
                             {"plugin": spec.name, "skipped": True}))
            else:
                out = spec.fn(ins[0], ins[1])
                tape.append((node, ins, out,
                             {"plugin": spec.name, "skipped": False}))
            self._axes[node.name] = (
                ("sequence",) + tuple(a for a in self._axes[node.inputs[0]]
                                      if a != "sequence")
            )
            return out
        if node.op == "reduce":
            axis = self._resolve_axis(node.inputs[0], node.axis)
            x = ins[0]
            if x.shape[axis] == 0 and node.reduce != "sum":
                # 空 mean/max 无定义：填零占位，由调用方按 unavailable 处理
                out = np.zeros(
                    tuple(s for i, s in enumerate(x.shape) if i != axis),
                    dtype=x.dtype,
                )
            else:
                out = ops.single_reduce(
                    x, node.reduce, axis=axis, combo=cid
                )
            self._axes[node.name] = tuple(
                a for i, a in enumerate(self._axes[node.inputs[0]])
                if i != axis
            )
            tape.append((node, ins, out,
                         {"axis": axis, "mode": node.reduce}))
            return out
        if node.op == "scatter_add":
            rows, idx = ins
            if rows.shape[0] == 0:
                # 空段没有发生次数：输出全零槽，不触碰任何（可能非法的）索引
                out = np.zeros(
                    (node.size,) + rows.shape[1:], dtype=rows.dtype
                )
            else:
                out = ops.single_scatter_add(
                    rows, idx, node.size, combo=cid
                )
            base = list(self._axes[node.inputs[0]])
            base = [a for a in base if a != "sequence"]
            base.insert(0, node.axis)
            self._axes[node.name] = tuple(base)
            tape.append((node, ins, out, {"size": node.size}))
            return out
        raise ValueError(f"参考解释器不支持算子 {node.op}")

    def _merge_axes(self, names):
        axes = list(self._axes[names[0]])
        for n in names[1:]:
            for a in self._axes[n]:
                if a not in axes:
                    axes.append(a)
        return tuple(axes)

    # ------------------------------------------------------------------
    def run_backward(self, seeds_dense: dict, ragged_seeds: dict | None = None):
        """逐组合反向，再把局部梯度按有效域合并回共享/副本/逐样本参数。

        ``seeds_dense[name]`` 形状为网格 + 逻辑形状；空段不适用位置给 0。
        ``ragged_seeds[name]`` 为扁平有效行上的种子（用于 ragged 最终输出，
        当前图没有这种需求，保留接口）。
        """
        seg = self.seg
        ragged_seeds = ragged_seeds or {}
        acc_dense = {}
        ragged_parts = {}
        for cid in range(seg.combos):
            table = self._tables[cid]
            tape = self._tapes[cid]
            coords = seg.decode(cid)
            seed = {
                name: np.asarray(g)[coords] for name, g in seeds_dense.items()
            }
            local = self._backward_combo(cid, table, tape, seed)
            self._merge(cid, local, acc_dense, ragged_parts)

        result = dict(acc_dense)
        # ragged 输入梯度按分段顺序拼回扁平有效行
        for name, parts in ragged_parts.items():
            chunks = [g for cid, g in sorted(parts)
                      if seg.segment_length(cid) > 0]
            t = self.inputs[name]
            result[name] = (
                np.concatenate(chunks, axis=0) if chunks
                else np.empty((0,) + t.data.shape[1:])
            )
        return result

    def _backward_combo(self, cid, table, tape, seed):
        grads = dict(seed)

        def accum(name, g):
            grads[name] = g if name not in grads else grads[name] + g

        for node, ins, out, ctx in reversed(tape):
            gout = grads.get(node.name)
            if gout is None:
                continue
            if node.op == "gather":
                accum(node.inputs[0],
                      ops.gather_vjp(gout, ins[1], ins[0].shape,
                                     axis=ctx["axis"]))
            elif node.op == "add":
                ga, gb = ops.add_vjp(gout, ins[0], ins[1])
                accum(node.inputs[0], ga)
                accum(node.inputs[1], gb)
            elif node.op == "matmul":
                ga, gb = ops.matmul_vjp(gout, ins[0], ins[1])
                accum(node.inputs[0], ga)
                accum(node.inputs[1], gb)
            elif node.op == "reduce":
                if ins[0].shape[ctx["axis"]] == 0:
                    # 空段没有有效行，无局部梯度可写（不适用位置种子为 0）
                    continue
                # 归约输出可能同时收到种子与下游（如 scatter）反传，
                # 先把多出的行维求和回归约输出的形状。
                gout_r = gout
                while gout_r.ndim > ins[0].ndim - 1:
                    gout_r = gout_r.sum(axis=0)
                accum(node.inputs[0],
                      self._reduce_vjp_local(gout_r, ins[0],
                                             ctx["mode"], ctx["axis"]))
            elif node.op == "scatter_add":
                accum(node.inputs[0],
                      np.take(gout, ins[1].astype(np.int64), axis=0))
            elif node.op == "plugin":
                if ctx.get("skipped"):
                    # 空段未调用插件，也没有伴随可算
                    continue
                spec = self.plugins.get(ctx["plugin"])
                gq, gp = spec.vjp(ins[0], ins[1], out, gout)
                accum(node.inputs[0], gq)
                accum(node.inputs[1], gp)
        return {n: grads[n] for n in self.inputs if n in grads}

    def _reduce_vjp_local(self, gout, x, mode, axis):
        n = x.shape[axis]
        if mode == "sum":
            return np.broadcast_to(np.expand_dims(gout, axis), x.shape).copy()
        if mode == "mean":
            return np.broadcast_to(
                np.expand_dims(gout / n, axis), x.shape
            ).copy()
        mx = np.expand_dims(x.max(axis=axis), axis)
        mask = (x == mx)
        cnt = mask.sum(axis=axis, keepdims=True)
        return np.broadcast_to(
            np.expand_dims(gout, axis) * mask / cnt, x.shape
        ).copy()

    def _merge(self, cid, local_grads, acc_dense, ragged_parts):
        seg = self.seg
        coords = seg.decode(cid)
        for name, g in local_grads.items():
            t = self.inputs[name]
            if t.is_ragged:
                ragged_parts.setdefault(name, []).append((cid, g))
            elif not t.mapped:
                # 共享权重：所有有效组合的局部梯度相加，先回收序列行维
                g = self._collapse_rows(g, t.data.shape)
                acc_dense[name] = (
                    g if name not in acc_dense else acc_dense[name] + g
                )
            elif t.mapped == ("replica",):
                if name not in acc_dense:
                    acc_dense[name] = np.zeros_like(t.data)
                g = self._collapse_rows(g, t.data.shape[1:])
                r = coords[seg.layers.index("replica")]
                acc_dense[name][r] += g
            else:
                raise ValueError(f"未覆盖的输入映射形态 {name}: {t.mapped}")

    @staticmethod
    def _collapse_rows(g, target_shape):
        """把单样本局部梯度里多余的 sequence 行维求和掉，再对齐目标形状。"""
        while g.ndim > len(target_shape):
            g = g.sum(axis=0)
        for ax, size in enumerate(target_shape):
            if size == 1 and g.shape[ax] != 1:
                g = g.sum(axis=ax, keepdims=True)
        return g.reshape(target_shape)
