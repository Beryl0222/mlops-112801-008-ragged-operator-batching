"""嵌套批化的分片执行：规则只覆盖一层时，片与回退在执行层面分开。"""

import unittest
from pathlib import Path

import numpy as np

from tests import _bootstrap  # noqa: F401
import segbatch as sb
from segbatch.plugins import PluginSpec, scaled_square, scaled_square_batched, scaled_square_batched_vjp, scaled_square_vjp
from segbatch.reference import ReferenceInterpreter

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "batched_axes.json"
ATOL = RTOL = 1e-7


def _register_sample_only(registry):
    """批化规则只声明覆盖 sample 一层（replica 缺失 -> 按副本分片）。"""
    registry.register(PluginSpec(
        name="sample_only_square",
        fn=scaled_square,
        vjp=scaled_square_vjp,
        batched_fn=scaled_square_batched,
        batched_vjp=scaled_square_batched_vjp,
        covers_layers=("sample",),
        note="只覆盖 sample 嵌套层",
    ))


class ShardingTest(unittest.TestCase):
    def setUp(self):
        self.data = sb.load_exchange(FIXTURE)
        for p in self.data["programs"]:
            if p.get("plugin"):
                p["plugin"] = "sample_only_square"

    def _run(self, layers, budget):
        seg = sb.build_segments(self.data, layers)
        inputs = sb.build_inputs(self.data, seg)
        graph = sb.build_graph(self.data)
        reg = sb.PluginRegistry()
        _register_sample_only(reg)
        eng = sb.Engine(graph, plugins=reg, fallback_budget=budget)
        outs = eng.run(layers, seg.grid, inputs, ["means", "scatter"])
        return seg, inputs, graph, reg, eng, outs

    def test_sharded_path_separates_covered_and_missing_layers(self):
        seg, inputs, graph, reg, eng, outs = self._run(
            ("replica", "sample"), budget=2
        )
        trace = next(t for t in eng.traces if t.node == "act")
        self.assertEqual(trace.path, "sharded")
        self.assertEqual(trace.covered, ("sample",))
        self.assertEqual(trace.missing, ("replica",))
        # 两个 replica -> 两片
        self.assertEqual(len(trace.shard_keys), 2)
        # 按片计预算（2 片），而不是按 6 个组合
        self.assertEqual(eng.meter.fallback_calls, 2)

    def test_sharded_result_matches_per_sample_reference(self):
        for layers in [("replica", "sample"), ("sample", "replica")]:
            seg, inputs, graph, reg, eng, outs = self._run(layers, budget=4)
            ref = ReferenceInterpreter(graph, reg, inputs, seg)
            refout = ref.run_forward()
            for name in ["means", "scatter"]:
                np.testing.assert_allclose(
                    outs[name].data, refout[name][1],
                    atol=ATOL, rtol=RTOL,
                )

    def test_sharded_backward_gradients_match_reference(self):
        for layers in [("replica", "sample"), ("sample", "replica")]:
            seg, inputs, graph, reg, eng, outs = self._run(layers, budget=4)
            seed_means = np.ones_like(outs["means"].data)
            for cid in outs["means"].unavailable:
                seed_means[seg.decode(int(cid))] = 0
            seeds = {"means": seed_means,
                     "scatter": np.ones_like(outs["scatter"].data)}
            grads = eng.backward(seeds)
            ref = ReferenceInterpreter(graph, reg, inputs, seg)
            ref.run_forward()
            refg = ref.run_backward(seeds)
            for name in ["tokens", "shared", "weight", "posbias"]:
                np.testing.assert_allclose(
                    grads[name]["data"], refg[name], atol=ATOL, rtol=RTOL,
                    err_msg=f"{layers} {name}",
                )

    def test_shard_budget_exhaustion_leaves_no_partial_output(self):
        # 两片但只给 1 片预算：必须在执行前整体失败
        with self.assertRaises(sb.BudgetExhausted):
            self._run(("replica", "sample"), budget=1)

    def test_sharding_respects_empty_segments(self):
        seg, inputs, graph, reg, eng, outs = self._run(
            ("replica", "sample"), budget=4
        )
        trace = next(t for t in eng.traces if t.node == "act")
        # 空组合 1、5 被跳过，不进入任何片
        flat_in_shards = set(trace.fallback_combos)
        self.assertNotIn(1, flat_in_shards)
        self.assertNotIn(5, flat_in_shards)
        # 空段 mean 仍标记不适用
        self.assertEqual(
            set(int(c) for c in outs["means"].unavailable), {1, 5}
        )


if __name__ == "__main__":
    unittest.main()
