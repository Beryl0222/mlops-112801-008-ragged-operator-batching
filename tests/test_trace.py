"""调用方可查的规则追踪、回退原因与输出位置溯源。"""

import unittest
from pathlib import Path

import numpy as np

from tests import _bootstrap  # noqa: F401
import segbatch as sb

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "batched_axes.json"


class TraceTest(unittest.TestCase):
    def setUp(self):
        self.data = sb.load_exchange(FIXTURE)
        self.seg = sb.build_segments(self.data, ("replica", "sample"))
        self.inputs = sb.build_inputs(self.data, self.seg)
        self.graph = sb.build_graph(self.data)

    def test_trace_lists_rules_fallback_and_empty_skips(self):
        eng = sb.Engine(self.graph, fallback_budget=6)
        eng.run(self.seg.layers, self.seg.grid, self.inputs,
                ["means", "scatter"])
        by_node = {t.node: t for t in eng.traces}
        # 内置算子全部走向量规则
        for n in ["embed", "feature", "pre", "means", "scatter"]:
            self.assertEqual(by_node[n].path, "vector")
            self.assertEqual(by_node[n].fallback_combos, [])
        # 插件无批化规则 -> 回退，并说明为什么必须逐样本
        act = by_node["act"]
        self.assertEqual(act.path, "fallback")
        self.assertEqual(act.rule, "plugin.scaled_square")
        self.assertEqual(len(act.fallback_combos), 6)
        self.assertEqual(act.missing, ("replica", "sample"))
        self.assertEqual(sorted(act.skipped_empty_combos), [1, 5])

    def test_trace_vector_rule_when_plugin_registers_batched(self):
        for p in self.data["programs"]:
            if p.get("plugin"):
                p["plugin"] = "scaled_square_batched"
        graph = sb.build_graph(self.data)
        eng = sb.Engine(graph, fallback_budget=0)
        eng.run(self.seg.layers, self.seg.grid, self.inputs, ["means"])
        act = next(t for t in eng.traces if t.node == "act")
        self.assertEqual(act.path, "vector")
        self.assertEqual(act.covered, ("replica", "sample"))

    def test_output_position_provenance_back_to_sample_and_position(self):
        eng = sb.Engine(self.graph, fallback_budget=6)
        outs = eng.run(self.seg.layers, self.seg.grid, self.inputs,
                       ["means", "scatter"])
        # 逐行来源：扁平有效行 -> replica/sample 坐标 + 原始序列位置
        prov = self.seg.provenance_rows()
        self.assertEqual(len(prov), self.seg.total_valid_rows)
        # 样本 c（index 2）被掩码位置 1，其首个有效行原始位置应为 0、再为 2
        c_rows = [p for p in prov if p["sample"] == 2 and p["replica"] == 0]
        # 样本 c 的 valid=[T,F,T,T]，掩码位置 1 后保留原始位置 0、2、3
        self.assertEqual([p["sequence_position"] for p in c_rows], [0, 2, 3])
        # 稠密输出按网格坐标可直接定位到具体副本/样本
        means = outs["means"]
        self.assertEqual(means.axes, ("filter",))
        self.assertEqual(means.mapped, ("replica", "sample"))
        # 空样本 b（sample=1）被标记为不适用
        self.assertIn(
            self.seg.encode((0, 1)), set(int(c) for c in means.unavailable)
        )

    def test_meter_records_allocations_and_views(self):
        eng = sb.Engine(self.graph, fallback_budget=6)
        eng.run(self.seg.layers, self.seg.grid, self.inputs, ["means"])
        snap = eng.meter.snapshot()
        self.assertEqual(snap["fallback_calls"], 6)
        self.assertGreater(snap["alloc_bytes"], 0)
        # 共享偏置广播只登记视图，不复制数据
        self.assertGreater(snap["view_bytes"], 0)
        self.assertTrue(all(r.shape for r in snap["allocations"]))


if __name__ == "__main__":
    unittest.main()
