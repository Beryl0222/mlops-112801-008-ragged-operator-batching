"""模型组可拿另一张算子图、另一种批形状继续调用（资料图只是交换样例）。"""

import unittest

import numpy as np

from tests import _bootstrap  # noqa: F401
import segbatch as sb
from segbatch.reference import ReferenceInterpreter


ATOL = RTOL = 1e-7


def _synthetic_inputs(layers, n_rep, n_smp, seed=11):
    rng = np.random.default_rng(seed)
    valid = []
    for s in range(n_smp):
        length = int(rng.integers(1, 5))
        v = np.ones(length, dtype=bool)
        if length >= 2 and s % 2 == 0:
            v[int(rng.integers(0, length))] = False
        valid.append(v)
    grid = (n_rep, n_smp) if layers == ("replica", "sample") \
        else (n_smp, n_rep)
    if layers == ("replica", "sample"):
        per_combo = [valid[s] for _ in range(n_rep) for s in range(n_smp)]
    else:
        per_combo = [valid[s] for s in range(n_smp) for _ in range(n_rep)]
    seg = sb.RaggedSegments.from_valid(layers, grid, per_combo)
    rows = []
    for coords in __import__("itertools").product(
        *(range(g) for g in grid)
    ):
        s = coords[layers.index("sample")]
        v = valid[s]
        rows.append(rng.normal(size=(v.sum(), 3)))
    tokens = sb.make_ragged(
        "tokens", ("sequence", "feature"), "sequence", rows, seg
    )
    weight = sb.LogicalTensor.dense(
        "weight", ("feature", "filter"),
        rng.normal(size=(n_rep, 3, 3)), mapped=("replica",), grid=seg,
    )
    bias = sb.LogicalTensor.shared("bias", ("feature",), rng.normal(size=3))
    return seg, {"tokens": tokens, "weight": weight, "bias": bias}


class AlternativeGraphTest(unittest.TestCase):
    def _graph_variant_a(self):
        # matmul -> 插件 -> max 归约
        return sb.Graph([
            sb.Node("pre", "matmul", ("tokens", "weight")),
            sb.Node("act", "plugin", ("pre", "bias"),
                    plugin="scaled_square_batched"),
            sb.Node("top", "reduce", ("act",), axis=-2, reduce="max"),
        ])

    def _graph_variant_b(self):
        # 另一张图：先 sum 归约（空段也适用），再插件不可用于稠密，
        # 改成 matmul 后直接 sum，验证纯内置算子图
        return sb.Graph([
            sb.Node("pre", "matmul", ("tokens", "weight")),
            sb.Node("total", "reduce", ("pre",), axis=-2, reduce="sum"),
        ])

    def test_alternative_graph_and_batch_shape_run_and_match_reference(self):
        for layers in [("replica", "sample"), ("sample", "replica")]:
            n_rep, n_smp = (3, 5) if layers == ("replica", "sample") \
                else (5, 3)
            seg, inputs = _synthetic_inputs(layers, n_rep, n_smp)
            graph = self._graph_variant_a()
            eng = sb.Engine(graph, fallback_budget=0)
            outs = eng.run(layers, seg.grid, inputs, ["top"])
            ref = ReferenceInterpreter(graph, eng.plugins, inputs, seg)
            refout = ref.run_forward()
            np.testing.assert_allclose(
                outs["top"].data, refout["top"][1], atol=ATOL, rtol=RTOL
            )
            # max 在空段不适用
            empties = set(np.flatnonzero(
                seg.lengths.reshape(-1) == 0
            ).tolist())
            self.assertEqual(
                set(int(c) for c in outs["top"].unavailable), empties
            )

    def test_sum_graph_handles_empty_segments_as_zero(self):
        layers = ("replica", "sample")
        seg, inputs = _synthetic_inputs(layers, 2, 4)
        graph = self._graph_variant_b()
        eng = sb.Engine(graph)
        outs = eng.run(layers, seg.grid, inputs, ["total"])
        ref = ReferenceInterpreter(graph, eng.plugins, inputs, seg)
        refout = ref.run_forward()
        np.testing.assert_allclose(
            outs["total"].data, refout["total"][1], atol=ATOL, rtol=RTOL
        )
        # sum 对空段适用（零），没有不适用位置
        self.assertEqual(outs["total"].unavailable, frozenset())

    def test_max_tie_gradient_split_equally(self):
        # 构造并列最大项，验证梯度在实际最大项间均分
        seg = sb.RaggedSegments.from_valid(
            ("sample",), (1,), [np.array([True, True])]
        )
        x = sb.LogicalTensor.ragged(
            "x", ("sequence", "feature"),
            np.array([[2.0, 1.0], [2.0, 0.0]]), seg,
        )
        graph = sb.Graph([
            sb.Node("m", "reduce", ("x",), axis=-2, reduce="max"),
        ])
        eng = sb.Engine(graph)
        outs = eng.run(("sample",), (1,), {"x": x}, ["m"])
        grads = eng.backward({"m": np.ones_like(outs["m"].data)})
        # 第一 feature 两个并列最大各分 0.5；第二 feature 只有第一项最大
        np.testing.assert_allclose(
            grads["x"]["data"], [[0.5, 1.0], [0.5, 0.0]], atol=ATOL
        )


class NoPartialGradientTest(unittest.TestCase):
    def test_rule_violation_aborts_before_any_parameter_gradient(self):
        # 登记一个错误的 gather 规则：它声称逐样本承载的 sequence 轴必须
        # “保持共享”。但该轴实际由 replica/sample 映射层逐样本变化承载，
        # 规则的共享承诺与真实映射事实冲突，必须在执行前判定违约。
        bad = sb.RuleSpec(
            "broken.gather", "gather", ("replica", "sample"),
            frozenset({"sequence"}),
        )
        rules = sb.RuleRegistry(extra={"gather": bad})
        seg, inputs = _synthetic_inputs(("replica", "sample"), 2, 3)
        rng = np.random.default_rng(3)
        idx_rows = [
            rng.integers(0, 3, size=seg.segment_length(c))
            for c in range(seg.combos)
        ]
        flat_idx = (np.concatenate(idx_rows) if seg.total_valid_rows
                    else np.empty(0, dtype=np.int64))
        indices = sb.LogicalTensor.ragged(
            "indices", ("sequence",), flat_idx, seg
        )
        table = sb.LogicalTensor.shared(
            "table", ("lookup", "feature"), rng.normal(size=(3, 3)),
        )
        inputs["indices"] = indices
        inputs["table"] = table
        graph = sb.Graph([
            sb.Node("e", "gather", ("table", "indices"), axis="lookup"),
        ])
        eng = sb.Engine(graph, rules=rules)
        with self.assertRaises(sb.RuleViolation) as cm:
            eng.run(seg.layers, seg.grid, inputs, ["e"])
        self.assertIn("sequence", str(cm.exception))
        # 违约在任何数值输出产生前抛出，调用方拿不到半次结果
        self.assertFalse(hasattr(eng, "_tape") and eng._tape)


class SharedGradientBatchingTest(unittest.TestCase):
    def test_shared_weight_gradient_is_full_domain_merge_not_local(self):
        rng = np.random.default_rng(0)
        # 含空段与掩码：有效行 = 3 + 0 + 1 = 4
        valid = [[True, True, True], [], [True, False]]
        seg = sb.RaggedSegments.from_valid(("sample",), (3,), valid)
        rows = [rng.normal(size=(int(np.asarray(v).sum()), 2)) for v in valid]
        tokens = sb.make_ragged(
            "tokens", ("sequence", "feature"), "sequence", rows, seg
        )
        weight = sb.LogicalTensor.shared(
            "weight", ("feature", "filter"), rng.normal(size=(2, 2))
        )
        bias = sb.LogicalTensor.shared(
            "bias", ("filter",), rng.normal(size=2)
        )
        graph = sb.Graph([
            sb.Node("pre", "matmul", ("tokens", "weight")),
            sb.Node("act", "plugin", ("pre", "bias"),
                    plugin="scaled_square_batched"),
            sb.Node("loss", "reduce", ("act",), axis=-2, reduce="sum"),
        ])
        eng = sb.Engine(graph, fallback_budget=0)
        out = eng.run(
            ("sample",), (3,),
            {"tokens": tokens, "weight": weight, "bias": bias}, ["loss"],
        )
        grads = eng.backward(
            {"loss": np.ones_like(out["loss"].data)}
        )
        ref = ReferenceInterpreter(
            graph, eng.plugins,
            {"tokens": tokens, "weight": weight, "bias": bias}, seg,
        )
        ref.run_forward()
        refg = ref.run_backward(
            {"loss": np.ones_like(out["loss"].data)}
        )
        # 共享权重梯度 == 逐样本在全部有效域上的合并，而非某个局部结果
        np.testing.assert_allclose(
            grads["weight"]["data"], refg["weight"], atol=ATOL, rtol=RTOL
        )
        self.assertTrue(grads["weight"]["complete"])
        self.assertEqual(
            grads["weight"]["domain_rows"], seg.total_valid_rows
        )
        self.assertEqual(seg.total_valid_rows, 4)


if __name__ == "__main__":
    unittest.main()
