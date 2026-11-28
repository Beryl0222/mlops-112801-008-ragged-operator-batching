"""短轨迹不会在嵌套批化中被无界补齐；压力规模下的数值与工作区验证。"""

import unittest
from pathlib import Path

import numpy as np

from tests import _bootstrap  # noqa: F401
import segbatch as sb
from segbatch.reference import ReferenceInterpreter
from segbatch.stress import (
    PluginReadAudit,
    build_stress_segments,
    generate_stress,
    padding_audit,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "batched_axes.json"
ATOL = RTOL = 1e-7


def _stress_program():
    """与交换样例同构、但维度由压力资料决定的算子图（图可替换的证据）。"""
    return [
        {"name": "embed", "op": "gather", "in": ["shared", "indices"],
         "axis": "lookup"},
        {"name": "feature", "op": "add", "in": ["tokens", "embed"]},
        {"name": "pre", "op": "matmul", "in": ["feature", "weight"]},
        {"name": "act", "op": "plugin", "in": ["pre", "posbias"],
         "plugin": "scaled_square_batched"},
        {"name": "means", "op": "reduce", "in": ["act"],
         "axis": -2, "reduce": "mean"},
    ]


class StressGenerationTest(unittest.TestCase):
    def setUp(self):
        spec = sb.load_exchange(FIXTURE)["stress"]
        self.spec = spec
        self.data = generate_stress(spec)

    def test_pcg64_is_deterministic(self):
        again = generate_stress(self.spec)
        np.testing.assert_array_equal(
            np.asarray(self.data["examples"][0]["tokens"]),
            np.asarray(again["examples"][0]["tokens"]),
        )

    def test_indices_stay_inside_shared_rows_when_valid(self):
        limit = self.data["_lookup_size"]
        for ex in self.data["examples"]:
            for good, idx in zip(ex["valid"], ex["indices"], strict=True):
                if good:
                    self.assertLess(idx, limit)

    def test_padding_audit_shows_dense_exceeds_workspace_ragged_fits(self):
        seg = build_stress_segments(self.data)
        audit = padding_audit(seg, self.spec["feature"])
        # 与资料中的算术结论一致
        self.assertGreater(audit["padded_bytes"], self.spec["workspace_bytes"])
        self.assertLess(audit["ragged_bytes"], self.spec["workspace_bytes"])
        # 最长轨迹长度只属于少数样本
        self.assertEqual(audit["longest_length"], self.spec["long_length"])


class NoUnboundedPaddingTest(unittest.TestCase):
    def setUp(self):
        spec = sb.load_exchange(FIXTURE)["stress"]
        # 缩小规模以便测试快速，同时保留长短悬殊
        spec = dict(spec)
        spec["examples"] = 18
        spec["replicas"] = 3
        spec["long_examples"] = [0, 9]
        self.spec = spec
        self.data = generate_stress(spec)

    def test_engine_allocation_never_pads_to_longest(self):
        seg = build_stress_segments(self.data)
        inputs = sb.build_inputs(self.data, seg)
        graph = sb.Graph.from_program(_stress_program())
        eng = sb.Engine(graph, fallback_budget=0)
        eng.run(seg.layers, seg.grid, inputs, ["means"])
        longest = self.spec["long_length"]
        # 审计每条分配记录：任何含 sequence 语义的缓冲，其第 0 维都不得
        # 等于 combos*longest（稠密补齐的指纹）
        for rec in eng.meter.records:
            self.assertNotEqual(
                rec.shape[0], seg.combos * longest,
                msg=f"发现按最长轨迹补齐的分配 {rec}",
            )
        # ragged 数据的有效行总数远小于补齐规模
        self.assertLess(
            seg.total_valid_rows, seg.combos * longest
        )

    def test_plugin_only_ever_reads_valid_rows(self):
        seg = build_stress_segments(self.data)
        inputs = sb.build_inputs(self.data, seg)
        graph = sb.Graph.from_program(_stress_program())
        reg = sb.PluginRegistry()
        base = reg.get("scaled_square_batched")
        audit_fn = PluginReadAudit(base.batched_fn)
        # 用带审计的批化回调替换
        reg.register(sb.PluginSpec(
            name="scaled_square_batched",
            fn=base.fn, vjp=base.vjp,
            batched_fn=audit_fn, batched_vjp=base.batched_vjp,
        ))
        eng = sb.Engine(graph, plugins=reg, fallback_budget=0)
        eng.run(seg.layers, seg.grid, inputs, ["means"])
        # 插件读到的总行数必须恰好等于分段声明的有效行数
        self.assertEqual(
            audit_fn.total_rows_seen, seg.total_valid_rows
        )
        # 单次调用读到的最大行数不超过有效行总数，绝不可能是补齐规模
        self.assertLessEqual(
            audit_fn.max_rows_seen, seg.total_valid_rows
        )

    def test_stress_forward_matches_reference_both_nestings(self):
        for layers in [("replica", "sample"), ("sample", "replica")]:
            seg = build_stress_segments(self.data, layers)
            inputs = sb.build_inputs(self.data, seg)
            graph = sb.Graph.from_program(_stress_program())
            eng = sb.Engine(graph, fallback_budget=0)
            outs = eng.run(layers, seg.grid, inputs, ["means"])
            ref = ReferenceInterpreter(graph, eng.plugins, inputs, seg)
            refout = ref.run_forward()
            np.testing.assert_allclose(
                outs["means"].data, refout["means"][1],
                atol=ATOL, rtol=RTOL,
            )
            # 空/全掩码组合的 mean 必须两边都标记或都为零占位
            self.assertEqual(
                set(int(c) for c in outs["means"].unavailable),
                set(np.flatnonzero(seg.lengths.reshape(-1) == 0).tolist()),
            )


if __name__ == "__main__":
    unittest.main()
