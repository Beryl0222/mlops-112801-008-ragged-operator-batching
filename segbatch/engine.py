"""批化执行引擎：把单样本算子组合成可声明不同批维的前向与反向。

设计要点
========
* 映射身份 ``replica`` / ``sample`` 与物理维序解耦；请求用 ``layers`` 声明
  嵌套顺序，规则按逻辑轴名工作。
* 每个节点先做规则推断（哪些输出轴是映射得到、哪些保持共享），规则违约
  在任何数值执行前抛出，保证不会留下半次参数梯度。
* 插件没有批化规则时，只对**有有效行的组合**逐样本回退，空段/全掩码组合
  不调用、不计预算；回退总量受显式预算约束。
* 前向写入磁带，反向按磁带回卷；映射到同一共享权重的各项梯度合并，
  且必须覆盖完整有效域才作为完整梯度返回。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import ops
from .budget import BudgetMeter
from .errors import GraphError, UndefinedTrajectory
from .plugins import PluginRegistry
from .rules import RuleRegistry
from .segments import RaggedSegments
from .tensor import LogicalTensor


# ===========================================================================
# 图与追踪记录
# ===========================================================================
@dataclass
class Node:
    name: str
    op: str
    inputs: tuple
    axis: object = None
    reduce: str | None = None
    size: int | None = None
    plugin: str | None = None


@dataclass
class RuleTrace:
    node: str
    op: str
    rule: str
    path: str                  # vector / fallback / sharded
    covered: tuple
    missing: tuple
    fallback_combos: list
    skipped_empty_combos: list
    output_mapped: tuple
    shard_keys: list = field(default_factory=list)
    note: str = ""


@dataclass
class TapeRecord:
    node: Node
    inputs: tuple
    output: LogicalTensor
    ctx: dict


@dataclass
class _PluginDecision:
    """插件节点的映射决策：覆盖能力由插件登记的批化回调决定。"""

    output_mapped: tuple
    output_ragged: bool
    covered_layers: tuple
    missing_layers: tuple
    rule_name: str
    shared_preserved: list = field(default_factory=list)
    violations: list = field(default_factory=list)


class Graph:
    def __init__(self, nodes: list[Node]):
        self.nodes = list(nodes)
        names = [n.name for n in self.nodes]
        if len(set(names)) != len(names):
            raise GraphError("图中节点名必须唯一")

    @classmethod
    def from_program(cls, programs: list[dict]) -> "Graph":
        nodes = []
        for p in programs:
            nodes.append(
                Node(
                    name=p["name"],
                    op=p["op"],
                    inputs=tuple(p["in"]),
                    axis=p.get("axis"),
                    reduce=p.get("reduce"),
                    size=p.get("size"),
                    plugin=p.get("plugin"),
                )
            )
        return cls(nodes)


# ===========================================================================
# 引擎
# ===========================================================================
class Engine:
    def __init__(
        self,
        graph: Graph,
        *,
        plugins: PluginRegistry | None = None,
        rules: RuleRegistry | None = None,
        fallback_budget: int | None = None,
        workspace_limit: int | None = None,
    ):
        self.graph = graph
        self.plugins = plugins or PluginRegistry()
        self.rules = rules or RuleRegistry()
        self.meter = BudgetMeter(fallback_budget, workspace_limit)
        self.traces: list[RuleTrace] = []

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------
    def run(
        self,
        layers: tuple,
        grid: tuple,
        inputs: dict[str, LogicalTensor],
        outputs: list[str],
    ) -> dict:
        layers = tuple(layers)
        grid = tuple(int(g) for g in grid)
        seg_grid = self._make_grid(layers, grid, inputs)
        table: dict[str, LogicalTensor] = dict(inputs)
        tape: list[TapeRecord] = []
        for tname, t in table.items():
            t.grid = seg_grid

        for node in self.graph.nodes:
            in_ts = [table[i] for i in node.inputs]
            out, rec = self._dispatch(node, in_ts, seg_grid)
            table[node.name] = out
            tape.append(rec)
            self.traces.append(self._trace_of(node, rec, seg_grid))

        self._tape = tape
        self._grid = seg_grid
        self._inputs = inputs
        result = {name: table[name] for name in outputs}
        return result

    def _make_grid(self, layers, grid, inputs):
        # 以任一 ragged 输入的分段为准；没有 ragged 输入时构造全稠密网格
        for t in inputs.values():
            if t.is_ragged:
                if t.segments.layers != layers or t.segments.grid != grid:
                    raise GraphError(
                        f"输入 {t.name} 的映射层 {t.segments.layers}/{t.segments.grid} "
                        f"与请求 {layers}/{grid} 不一致"
                    )
                return t.segments
        # 无 ragged 输入时的稠密网格兜底：每个组合都“有效”，
        # offsets 仅承载组合边界，不携带序列语义。
        offsets = np.arange(int(np.prod(grid)) + 1, dtype=np.int64)
        return RaggedSegments.from_offsets(layers, grid, offsets)

    # ------------------------------------------------------------------
    # 节点分派
    # ------------------------------------------------------------------
    def _dispatch(self, node: Node, in_ts: list[LogicalTensor], seg: RaggedSegments):
        if node.op == "gather":
            return self._do_gather(node, in_ts, seg)
        if node.op == "add":
            return self._do_add(node, in_ts, seg)
        if node.op == "matmul":
            return self._do_matmul(node, in_ts, seg)
        if node.op == "reduce":
            return self._do_reduce(node, in_ts, seg)
        if node.op == "scatter_add":
            return self._do_scatter(node, in_ts, seg)
        if node.op == "plugin":
            return self._do_plugin(node, in_ts, seg)
        raise GraphError(f"未知算子 {node.op!r}")

    def _decision(self, node, in_ts, seg, output_ragged):
        facts = [t.mapped_facts() for t in in_ts]
        decision = self.rules.infer(node.op, seg.layers, facts, output_ragged)
        self.rules.assert_no_violation(decision)
        return decision

    def _metered_zeros(self, shape, dtype, *, note):
        """通过计量器分配零缓冲（归约/scatter 等输出）。"""
        return self.meter.alloc(shape, dtype, kind="temp", fill=0, note=note)

    # ---- gather --------------------------------------------------------
    def _do_gather(self, node, in_ts, seg):
        table_t, idx_t = in_ts
        axis = table_t.resolve_axis(node.axis)
        decision = self._decision(node, in_ts, seg, True)
        flat_idx = idx_t.data.astype(np.int64)
        out = ops.batch_gather(
            table_t.data, flat_idx,
            axis=axis, axis_name=node.axis,
            positions=seg.positions, combo_ids=seg.combo_ids,
        )
        axes = tuple("sequence" if i == axis else a
                     for i, a in enumerate(table_t.axes))
        lt = LogicalTensor.ragged(node.name, axes, out, seg)
        return lt, TapeRecord(node, tuple(in_ts), lt,
                              {"axis": axis, "decision": decision})

    # ---- add -----------------------------------------------------------
    def _do_add(self, node, in_ts, seg):
        decision = self._decision(node, in_ts, seg, any(t.is_ragged for t in in_ts))
        a, b = in_ts
        if a.is_ragged or b.is_ragged:
            ragged_t = a if a.is_ragged else b
            aa = self._align_flat(a, seg)
            bb = self._align_flat(b, seg)
            out = ops.batch_add(aa, bb)
            axes = self._merge_axes(a, b)
            lt = LogicalTensor.ragged(node.name, axes, out, seg,
                                      ragged_t.seq_axis)
        else:
            out = ops.batch_add(a.data, b.data)
            axes = self._merge_axes(a, b)
            lt = LogicalTensor.dense(node.name, axes, out,
                                     decision.output_mapped, seg)
        return lt, TapeRecord(node, tuple(in_ts), lt, {"decision": decision})

    # ---- matmul --------------------------------------------------------
    def _do_matmul(self, node, in_ts, seg):
        a, b = in_ts
        decision = self._decision(node, in_ts, seg, a.is_ragged or b.is_ragged)
        if a.is_ragged:
            # a: 扁平有效行 (N, f)；b: 按 replica (R, f, h) 或共享 (f, h)
            bb = self._align_flat(b, seg)          # (N, f, h)
            out = np.einsum("nf,nfh->nh", a.data, bb, optimize=True)
            axes = ("sequence", b.axes[-1])
            lt = LogicalTensor.ragged(node.name, axes, out, seg)
        else:
            out = ops.single_matmul(a.data, b.data)
            lt = LogicalTensor.dense(node.name, a.axes[:-1] + (b.axes[-1],),
                                     out, decision.output_mapped, seg)
        return lt, TapeRecord(node, tuple(in_ts), lt, {"decision": decision})

    # ---- reduce --------------------------------------------------------
    def _do_reduce(self, node, in_ts, seg):
        (x,) = in_ts
        axis = x.resolve_axis(node.axis)
        if not x.is_ragged or x.axes[axis] != x.seq_axis:
            raise GraphError("当前引擎对 reduce 支持 ragged sequence 轴归约")
        decision = self._decision(node, in_ts, seg, False)
        reduced, unavailable = ops.batch_segment_reduce(
            x.data, seg.offsets, node.reduce
        )
        # 空 sum 为零且适用；mean/max 在空段不适用（ops 已据零长度判定）
        if node.reduce == "sum":
            unavailable = set()
        out_shape = seg.grid + reduced.shape[1:]
        # 输出缓冲经计量器分配，便于审计是否存在无界补齐
        out = self._metered_zeros(
            out_shape, reduced.dtype, note=f"{node.name}.reduce"
        )
        out.reshape(-1, *reduced.shape[1:])[:] = reduced
        trailing = tuple(a for i, a in enumerate(x.axes) if i != axis)
        lt = LogicalTensor.dense(node.name, trailing, out, seg.layers, seg)
        lt.unavailable = frozenset(unavailable)
        return lt, TapeRecord(
            node, tuple(in_ts), lt,
            {"axis": axis, "mode": node.reduce, "decision": decision,
             "unavailable": frozenset(unavailable)},
        )

    # ---- scatter_add ---------------------------------------------------
    def _do_scatter(self, node, in_ts, seg):
        rows_t, idx_t = in_ts
        # node.axis 是输出上新开的 lookup 轴名；输入行不含该轴，
        # 其在输出逻辑轴中的位置由节点声明显式给出，缺省插在最前。
        axis = (
            rows_t.resolve_axis(node.axis)
            if node.axis in rows_t.data_axes
            else 0
        )
        decision = self._decision(node, in_ts, seg, False)
        out = ops.batch_scatter_add(
            rows_t.data, idx_t.data.astype(np.int64), seg.combo_ids,
            seg, node.size, axis=axis,
        )
        base_axes = list(rows_t.data_axes)
        base_axes.insert(axis, node.axis)
        lt = LogicalTensor.dense(node.name, tuple(base_axes), out,
                                 seg.layers, seg)
        return lt, TapeRecord(
            node, tuple(in_ts), lt,
            {"axis": axis, "size": node.size, "decision": decision},
        )

    # ---- plugin（受预算与空段守卫约束） --------------------------------
    def _do_plugin(self, node, in_ts, seg):
        spec = self.plugins.get(node.plugin)
        q_t, p_t = in_ts
        active = list(map(int, seg.active_combo_ids))
        empty = list(map(int, seg.empty_combo_ids))
        ctx = {"plugin": spec.name, "active": active, "empty": empty}
        facts = [t.mapped_facts() for t in in_ts]
        mapped_union: set = set()
        for f in facts:
            mapped_union |= set(f["mapped"])
        if not spec.has_vector_rule:
            covered, missing = (), tuple(seg.layers)
        elif spec.covers_layers is None:
            covered = tuple(l for l in seg.layers if l in mapped_union)
            missing = tuple(l for l in seg.layers if l not in mapped_union)
        else:
            covered = tuple(l for l in seg.layers
                            if l in spec.covers_layers and l in mapped_union)
            missing = tuple(l for l in seg.layers if l not in covered)
        decision = _PluginDecision(
            output_mapped=tuple(l for l in seg.layers if l in mapped_union),
            output_ragged=True,
            covered_layers=covered,
            missing_layers=missing,
            rule_name=f"plugin.{spec.name}",
        )
        axes = ("sequence",) + tuple(
            a for a in q_t.axes if a != q_t.seq_axis
        )

        # 路径 1：批化规则覆盖全部请求映射层 -> 一次组批
        if spec.has_vector_rule and not missing:
            qq = self._align_flat(q_t, seg)
            pp = self._align_flat(p_t, seg)
            out = spec.batched_fn(qq, pp)
            lt = LogicalTensor.ragged(node.name, axes, out, seg)
            ctx.update(path="vector", covered=covered, missing=(),
                       decision=decision, fallback_combos=[])
            return lt, TapeRecord(node, tuple(in_ts), lt, ctx)

        # 路径 2：批化规则只覆盖部分嵌套层 -> 按缺失层分片
        if spec.has_vector_rule and missing:
            return self._sharded_plugin(
                node, in_ts, seg, spec, covered, missing, active, empty,
                decision, axes, ctx,
            )

        # 无批化规则：逐样本回退。空段/全掩码组合不调用、不计预算。
        needed = len(active)
        self.meter.charge_fallback(needed, f"{node.name}.{spec.name}")
        chunks = []
        for cid in active:
            q = q_t.combo_view(cid)
            if q.shape[0] == 0:
                # 守卫：分段元数据与数据不一致时绝不把空轨迹发给插件
                raise UndefinedTrajectory(f"组合 {cid} 无有效行却进入回退")
            p = p_t.combo_view(cid) if p_t.mapped else p_t.data
            chunks.append(spec.fn(q, p))
        out = (np.concatenate(chunks, axis=0) if chunks
               else np.empty((0,) + q_t.data.shape[1:], dtype=q_t.data.dtype))
        axes = ("sequence",) + tuple(a for a in q_t.axes if a != q_t.seq_axis)
        lt = LogicalTensor.ragged(node.name, axes, out, seg)
        ctx.update(path="fallback", covered=(), missing=seg.layers,
                   fallback_combos=active, decision=decision)
        return lt, TapeRecord(node, tuple(in_ts), lt, ctx)

    def _sharded_plugin(self, node, in_ts, seg, spec, covered, missing,
                        active, empty, decision, axes, ctx):
        """部分覆盖：缺失层逐片，覆盖层在片内批化。

        片与片相互独立；预算在任何数值执行前按片数预扣，预算不足时不会写出
        半次输出。空片（无有效组合）既不调用插件也不消耗预算。
        """
        q_t, p_t = in_ts
        miss_idx = [seg.layers.index(layer) for layer in missing]
        shards: dict[tuple, list] = {}
        for cid in active:
            key = tuple(seg.decode(cid)[i] for i in miss_idx)
            shards.setdefault(key, []).append(cid)
        self.meter.charge_fallback(
            len(shards), f"{node.name}.{spec.name}.sharded"
        )
        out = np.empty_like(q_t.data)
        for cids in shards.values():
            row_sel = np.concatenate([
                np.arange(seg.offsets[c], seg.offsets[c + 1]) for c in cids
            ])
            qq = q_t.data[row_sel]
            if p_t.mapped:
                pp = self._align_flat(p_t, seg)[row_sel]
            else:
                pp = np.broadcast_to(
                    p_t.data, (row_sel.size,) + p_t.data.shape
                )
            out[row_sel] = spec.batched_fn(qq, pp)
        lt = LogicalTensor.ragged(node.name, axes, out, seg)
        ctx.update(
            path="sharded", covered=covered, missing=missing,
            shard_keys=sorted(shards),
            fallback_combos=sorted(c for cids in shards.values() for c in cids),
            decision=decision,
        )
        return lt, TapeRecord(node, tuple(in_ts), lt, ctx)

    # ------------------------------------------------------------------
    # 辅助：把任意输入对齐到扁平有效行
    # ------------------------------------------------------------------
    def _align_flat(self, t: LogicalTensor, seg: RaggedSegments) -> np.ndarray:
        if t.is_ragged:
            return t.data
        if not t.mapped:
            # 共享：广播到每个有效行（视图，不复制）
            shape = (seg.total_valid_rows,) + t.data.shape
            return self.meter.view(
                np.broadcast_to(t.data, shape), note=f"align.{t.name}.shared"
            )
        # 稠密映射：按每行所属组合取副本
        if len(t.mapped) == 1 and t.mapped[0] in seg.layers:
            layer = t.mapped[0]
            layer_idx = seg.layers.index(layer)
            row_values = np.array(
                [seg.decode(int(cid))[layer_idx] for cid in seg.combo_ids],
                dtype=np.int64,
            )
            return self.meter.view(
                t.data[row_values], note=f"align.{t.name}.{layer}"
            )
        # 多映射层稠密（理论上插件 bias 不走到这里，保留通用路径）
        coords = np.array([seg.decode(int(cid)) for cid in seg.combo_ids],
                          dtype=np.int64)
        picks = tuple(
            coords[:, seg.layers.index(layer)] for layer in t.mapped
        )
        return self.meter.view(t.data[picks], note=f"align.{t.name}.multi")

    def _merge_axes(self, a: LogicalTensor, b: LogicalTensor) -> tuple:
        axes = list(a.axes)
        for ax in b.axes:
            if ax not in axes:
                axes.append(ax)
        return tuple(axes)

    def _trace_of(self, node, rec: TapeRecord, seg) -> RuleTrace:
        ctx = rec.ctx
        path = ctx.get("path", "vector")
        return RuleTrace(
            node=node.name, op=node.op,
            rule=ctx.get("decision").rule_name if "decision" in ctx
            else f"plugin.{ctx.get('plugin')}",
            path=path,
            covered=ctx.get("covered", seg.layers),
            missing=ctx.get("missing", ()),
            fallback_combos=ctx.get("fallback_combos", []),
            skipped_empty_combos=ctx.get("empty", [])
            if node.op == "plugin" else [],
            output_mapped=rec.output.mapped,
            shard_keys=ctx.get("shard_keys", []),
        )

    # ------------------------------------------------------------------
    # 反向
    # ------------------------------------------------------------------
    def backward(self, seeds: dict[str, np.ndarray]) -> dict:
        """按前向磁带反传。

        ``seeds`` 给出每个最终输出的梯度（稠密结果的不适用位置传零）。
        映射到同一共享权重的各项梯度在传播中直接累加；前向若完整成功，
        覆盖的就是全部有效域，不会返回“半次”参数梯度。
        """
        tape = self._tape
        seg = self._grid
        # 当前每个张量名的梯度（物理数组）
        seed: dict[str, np.ndarray] = {
            name: np.asarray(g) for name, g in seeds.items()
        }

        def accum(name, g):
            seed[name] = g if name not in seed else seed[name] + g

        for rec in reversed(tape):
            node = rec.node
            gout = seed.get(node.name)
            if gout is None:
                continue
            in_ts = [self._find(n) for n in node.inputs]

            if node.op == "gather":
                table_t, idx_t = in_ts
                axis = rec.ctx["axis"]
                accum(table_t.name,
                      ops.gather_vjp(gout, idx_t.data, table_t.data.shape,
                                     axis=axis))

            elif node.op == "add":
                a, b = in_ts
                if rec.output.is_ragged:
                    accum(a.name, self._unbroadcast_to(gout, a, seg))
                    accum(b.name, self._unbroadcast_to(gout, b, seg))
                else:
                    ga, gb = ops.add_vjp(gout, a.data, b.data)
                    accum(a.name, ga)
                    accum(b.name, gb)

            elif node.op == "matmul":
                a, b = in_ts
                if a.is_ragged:
                    b_aligned = self._align_flat(b, seg)           # (N,f,h)
                    ga = np.einsum("nh,nfh->nf", gout, b_aligned,
                                   optimize=True)
                    gb_flat = np.einsum("nf,nh->nfh", a.data, gout,
                                        optimize=True)
                    accum(a.name, ga)
                    accum(b.name, self._reduce_flat(gb_flat, b, seg))
                else:
                    ga, gb = ops.matmul_vjp(gout, a.data, b.data)
                    accum(a.name, ga)
                    accum(b.name, gb)

            elif node.op == "reduce":
                (x,) = in_ts
                mode = rec.ctx["mode"]
                # 先把可能混入的下游额外行维在网格之后回收
                g_combo = gout
                grid_ndim = len(seg.grid)
                target_ndim = grid_ndim + (x.data.ndim - 1)
                while g_combo.ndim > target_ndim:
                    g_combo = g_combo.sum(axis=grid_ndim)
                # 物理数组按网格坐标排布；按组合 id 解码取每组合的归约输出梯度，
                # 绝不能把网格数组直接 reshape 成扁平组合维。
                trailing = g_combo.shape[grid_ndim:]
                per_combo = np.empty((seg.combos,) + trailing, dtype=g_combo.dtype)
                for cid in range(seg.combos):
                    per_combo[cid] = g_combo[seg.decode(cid)]
                gx = ops.reduce_vjp_flat(per_combo, x.data, seg.offsets, mode)
                accum(x.name, gx)

            elif node.op == "scatter_add":
                rows_t, idx_t = in_ts
                g_rows = ops.scatter_vjp(
                    gout, idx_t, seg, axis=rec.ctx["axis"]
                )
                accum(rows_t.name, g_rows)

            elif node.op == "plugin":
                spec = self.plugins.get(rec.ctx["plugin"])
                q_t, p_t = in_ts
                if spec.vjp is None and spec.batched_vjp is None:
                    raise GraphError(f"插件 {spec.name} 未登记伴随，无法反向")
                # 上游梯度按网格坐标排布；插件/ragged 按扁平有效行工作，
                # 先用分段布局把梯度重排到有效行（空段没有行，自然不参与）。
                gout_flat = self._grid_to_flat_rows(gout, seg)
                if spec.has_vector_rule:
                    qq = self._align_flat(q_t, seg)
                    pp = self._align_flat(p_t, seg)
                    gq, gp_batched = spec.batched_vjp(
                        qq, pp, rec.output.data, gout_flat
                    )
                    # 批化伴随的 gp 已对全部有效行求和：
                    # 共享偏置即最终梯度；replica 偏置需要按副本分桶。
                    if not p_t.mapped:
                        gp_flat = gp_batched
                    elif p_t.mapped == ("replica",):
                        # 用行级 g 重新按副本聚合（batched 伴随给的是总和）
                        z = qq + pp
                        grow = 2.0 * z * gout_flat
                        gp_flat = self._aggregate_replica_rows(grow, p_t, seg)
                    else:
                        gp_flat = gp_batched
                else:
                    gq_chunks, gp_combo = [], []
                    active = rec.ctx["active"]
                    # 反向伴随不耗前向 fallback_budget，单独计量
                    self.meter.charge_backward(len(active))
                    for cid in active:
                        sl = seg.slice(cid)
                        gq_c, gp_c = spec.vjp(
                            q_t.combo_view(cid),
                            p_t.combo_view(cid) if p_t.mapped else p_t.data,
                            rec.output.data[sl], gout_flat[sl],
                        )
                        gq_chunks.append(gq_c)
                        # gp_c 已是该组合内行维求和后的偏置梯度，
                        # 每个组合只能计一次，绝不能广播回该组合的每一行。
                        gp_combo.append((cid, gp_c))
                    gq = (np.concatenate(gq_chunks, axis=0)
                          if gq_chunks
                          else np.empty((0,) + q_t.data.shape[1:]))
                    # 偏置按其真实映射形态聚合（共享 / replica），不经过行布局
                    if not p_t.mapped:
                        gp_flat = sum((gp for _, gp in gp_combo),
                                      np.zeros_like(p_t.data))
                    else:
                        gp_flat = self._aggregate_replica_combo(
                            gp_combo, p_t, seg
                        )
                accum(q_t.name, gq)
                # 偏置是稠密（共享或 replica）张量，gp_flat 已按其形态聚合
                accum(p_t.name, gp_flat)

        return self._finalize_grads(seed, seg)

    def _unbroadcast_to(self, g_flat, t: LogicalTensor, seg: RaggedSegments):
        """把扁平有效行上的梯度归并回某个输入的物理形状。"""
        if t.is_ragged:
            return g_flat
        return self._reduce_flat(g_flat, t, seg)

    def _aggregate_replica_combo(self, gp_combo, p_t: LogicalTensor,
                                 seg: RaggedSegments):
        """把逐组合（已对行求和）的偏置梯度聚合到 replica 权重形状。"""
        layer_idx = seg.layers.index("replica")
        out = np.zeros_like(p_t.data)
        for cid, gp in gp_combo:
            r = seg.decode(int(cid))[layer_idx]
            out[r] += gp
        return out

    def _aggregate_replica_rows(self, grow, p_t: LogicalTensor,
                                seg: RaggedSegments):
        """把行级偏置梯度按 replica 分桶并对各自行求和。"""
        layer_idx = seg.layers.index("replica")
        out = np.zeros_like(p_t.data)
        row_rep = np.array(
            [seg.decode(int(cid))[layer_idx] for cid in seg.combo_ids],
            dtype=np.int64,
        )
        np.add.at(out, row_rep, grow)
        return out

    def _grid_to_flat_rows(self, g, seg: RaggedSegments):
        """把上游梯度规整到扁平有效行布局。

        两种来源都要支持：
        * 已是扁平 ragged（行数 == 有效总行数）：直接返回；
        * 按映射网格坐标排布：逐组合解码取尾形，再按 ``combo_ids`` 拼回。
        """
        if g.shape[0] == seg.total_valid_rows:
            return g
        grid_ndim = len(seg.grid)
        trailing = g.shape[grid_ndim:]
        flat = np.empty((seg.total_valid_rows,) + trailing, dtype=g.dtype)
        for row, cid in enumerate(seg.combo_ids):
            flat[row] = g[seg.decode(int(cid))]
        return flat

    def _reduce_flat(self, flat_grad, t: LogicalTensor, seg: RaggedSegments):
        """把按有效行对齐的梯度合并回稠密（共享/副本）张量的形状。

        扁平梯度形状为 ``(有效行数,) + 逻辑尾形``。第 0 维是有效行（映射 +
        sequence 压平），必须求和；逻辑尾维只在真正的 NumPy 广播（size==1）
        时才回收，不能把正常的特征维当广播维。
        """
        target = t.data.shape
        if not t.mapped:
            g = flat_grad
            # 折叠所有前置的有效行/映射维，直到与目标逻辑秩相同
            while g.ndim > len(target):
                g = g.sum(axis=0)
            for axis, size in enumerate(target):
                if size == 1 and g.shape[axis] != 1:
                    g = g.sum(axis=axis, keepdims=True)
            return g.reshape(target)
        if len(t.mapped) == 1:
            layer = t.mapped[0]
            layer_idx = seg.layers.index(layer)
            g = flat_grad
            trailing = t.data.shape[1:]
            # 折叠行内多余序列维，保留映射层对应的那一个前置维
            while g.ndim > 1 + len(trailing):
                g = g.sum(axis=1)
            out = np.zeros((seg.grid[layer_idx],) + tuple(trailing),
                           dtype=flat_grad.dtype)
            row_values = np.array(
                [seg.decode(int(cid))[layer_idx] for cid in seg.combo_ids],
                dtype=np.int64,
            )
            np.add.at(out, row_values, g)
            for ax, size in enumerate(trailing, start=1):
                if size == 1 and out.shape[ax] != 1:
                    out = out.sum(axis=ax, keepdims=True)
            return out.reshape(t.data.shape)
        raise GraphError("多映射层稠密梯度合并需要按完整网格聚合（当前场景未使用）")

    def _finalize_grads(self, seed: dict, seg) -> dict:
        result = {}
        for name in self._inputs:
            if name not in seed:
                continue
            src = self._inputs[name]
            if src.data.dtype.kind not in ("f", "c"):
                continue  # 整数索引没有梯度
            g = seed[name]
            result[name] = {
                "data": g,
                "axes": src.axes,
                "mapped": src.mapped,
                "complete": True,
                "domain_rows": seg.total_valid_rows,
            }
        return result

    def _find(self, name) -> LogicalTensor:
        if name in self._inputs:
            return self._inputs[name]
        for rec in self._tape:
            if rec.node.name == name:
                return rec.output
        raise GraphError(f"找不到张量 {name!r}")
