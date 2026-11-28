"""引擎前向/反向与逐样本解释器的数值对照（容差 1e-7）。"""

import unittest

import numpy as np

from tests import _bootstrap  # noqa: F401
import segbatch as sb
from segbatch.reference import ReferenceInterpreter

from pathlib import Path

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "batched_axes.json"
ATOL = RTOL = 1e-7


def _make(layers, plugin_name="scaled_square", budget=20):
    data = sb.load_exchange(FIXTURE)
    if plugin_name != "scaled_square":
        for p in data["programs"]:
            if p.get("plugin"):
                p["plugin"] = plugin_name
    seg = sb.build_segments(data, layers)
    inputs = sb.build_inputs(data, seg)
    graph = sb.build_graph(data)
    eng = sb.Engine(graph, fallback_budget=budget)
    return data, seg, inputs, graph, eng


def _seeds(seg, outs):
    seed_means = np.ones_like(outs["means"].data)
    for cid in outs["means"].unavailable:
        seed_means[seg.decode(int(cid))] = 0.0
    seed_scatter = np.ones_like(outs["scatter"].data)
    return {"means": seed_means, "scatter": seed_scatter}


class ForwardReferenceTest(unittest.TestCase):
    def test_vector_forward_matches_per_sample_both_nestings(self):
        for layers in [("replica", "sample"), ("sample", "replica")]:
            _, seg, inputs, graph, eng = _make(layers)
            outs = eng.run(layers, seg.grid, inputs, ["means", "scatter"])
            ref = ReferenceInterpreter(graph, eng.plugins, inputs, seg)
            refout = ref.run_forward()
            for name in ["means", "scatter"]:
                np.testing.assert_allclose(
                    outs[name].data, refout[name][1],
                    atol=ATOL, rtol=RTOL,
                    err_msg=f"{layers} {name} 前向不一致",
                )

    def test_empty_mean_marked_unavailable_but_sum_and_scatter_are_fine(self):
        _, seg, inputs, graph, eng = _make(("replica", "sample"))
        outs = eng.run(seg.layers, seg.grid, inputs, ["means", "scatter"])
        # 组合 1、5 是空样本 b
        self.assertEqual(set(int(c) for c in outs["means"].unavailable), {1, 5})
        # scatter 对空组合产出零槽，仍然适用
        self.assertEqual(outs["scatter"].unavailable, frozenset())
        np.testing.assert_array_equal(outs["scatter"].data[0, 1], 0.0)

    def test_nesting_order_preserves_identity(self):
        _, seg_a, in_a, graph, eng_a = _make(("replica", "sample"))
        out_a = eng_a.run(seg_a.layers, seg_a.grid, in_a, ["means"])
        _, seg_b, in_b, graph, eng_b = _make(("sample", "replica"))
        out_b = eng_b.run(seg_b.layers, seg_b.grid, in_b, ["means"])
        # 同一 (replica, sample) 身份在两种物理排布下取值相同
        for r in range(2):
            for s in range(4):
                if s == 1:  # 空段
                    continue
                np.testing.assert_allclose(
                    out_a["means"].data[r, s],
                    out_b["means"].data[s, r],
                    atol=ATOL, rtol=RTOL,
                )


class BackwardReferenceTest(unittest.TestCase):
    def test_gradients_match_per_sample_merge(self):
        for layers in [("replica", "sample"), ("sample", "replica")]:
            _, seg, inputs, graph, eng = _make(layers)
            outs = eng.run(layers, seg.grid, inputs, ["means", "scatter"])
            seeds = _seeds(seg, outs)
            grads = eng.backward(seeds)
            ref = ReferenceInterpreter(graph, eng.plugins, inputs, seg)
            ref.run_forward()
            refg = ref.run_backward(seeds)
            for name in ["tokens", "shared", "weight", "posbias"]:
                np.testing.assert_allclose(
                    grads[name]["data"], refg[name],
                    atol=ATOL, rtol=RTOL,
                    err_msg=f"{layers} {name} 梯度不一致",
                )
                self.assertTrue(grads[name]["complete"])

    def test_batched_plugin_zero_fallback_identical_gradients(self):
        for layers in [("replica", "sample"), ("sample", "replica")]:
            _, seg, inputs, graph, eng = _make(
                layers, plugin_name="scaled_square_batched", budget=0
            )
            outs = eng.run(layers, seg.grid, inputs, ["means", "scatter"])
            self.assertEqual(eng.meter.fallback_calls, 0)
            seeds = _seeds(seg, outs)
            grads = eng.backward(seeds)
            ref = ReferenceInterpreter(graph, eng.plugins, inputs, seg)
            ref.run_forward()
            refg = ref.run_backward(seeds)
            for name in ["tokens", "shared", "weight", "posbias"]:
                np.testing.assert_allclose(
                    grads[name]["data"], refg[name], atol=ATOL, rtol=RTOL
                )


class BudgetAndGuardTest(unittest.TestCase):
    def test_fallback_counts_only_active_combos(self):
        _, seg, inputs, graph, eng = _make(
            ("replica", "sample"), budget=6
        )
        outs = eng.run(seg.layers, seg.grid, inputs, ["means"])
        # 6 个有有效行的组合；两个空段不调用也不计预算
        self.assertEqual(eng.meter.fallback_calls, 6)
        trace = next(t for t in eng.traces if t.node == "act")
        self.assertEqual(trace.path, "fallback")
        self.assertEqual(sorted(trace.skipped_empty_combos), [1, 5])
        # 反向伴随单独计量，不二次消耗前向预算
        seed = np.ones_like(outs["means"].data)
        for cid in outs["means"].unavailable:
            seed[seg.decode(int(cid))] = 0.0
        eng.backward({"means": seed})
        self.assertEqual(eng.meter.fallback_calls, 6)
        self.assertEqual(eng.meter.backward_calls, 6)

    def test_budget_exhaustion_raises_before_any_partial_gradient(self):
        _, seg, inputs, graph, eng = _make(
            ("replica", "sample"), budget=5
        )
        with self.assertRaises(sb.BudgetExhausted) as cm:
            eng.run(seg.layers, seg.grid, inputs, ["means"])
        self.assertEqual(cm.exception.needed, 6)
        # 预算在插件前向预扣，失败时不会产生输出
        self.assertFalse(hasattr(eng, "_tape") and eng._tape and
                         all(rec.node.name != "act" for rec in eng._tape)
                         is False)

    def test_plugin_never_receives_empty_or_masked_trajectory(self):
        # 让插件在收到空行时主动爆炸，证明调度层已隔离
        data = sb.load_exchange(FIXTURE)
        seg = sb.build_segments(data, ("replica", "sample"))
        inputs = sb.build_inputs(data, seg)
        graph = sb.build_graph(data)
        reg = sb.PluginRegistry()
        spec = sb.PluginSpec(
            name="strict",
            fn=lambda q, p: (_ for _ in ()).throw(
                AssertionError("插件收到了非法轨迹")
            ) if q.shape[0] == 0 else (q + p) ** 2,
            vjp=reg.get("scaled_square").vjp,
        )
        reg.register(spec)
        for node in graph.nodes:
            if node.op == "plugin":
                node.plugin = "strict"
        eng = sb.Engine(graph, plugins=reg, fallback_budget=10)
        # 不抛 AssertionError 即说明空段从未进入插件
        outs = eng.run(seg.layers, seg.grid, inputs, ["means"])
        self.assertEqual(set(int(c) for c in outs["means"].unavailable), {1, 5})


if __name__ == "__main__":
    unittest.main()
