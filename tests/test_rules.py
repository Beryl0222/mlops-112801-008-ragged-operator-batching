"""向量化规则的映射推断、违约检测与执行边界。"""

import unittest

import numpy as np

from tests import _bootstrap  # noqa: F401
import segbatch as sb
from segbatch.rules import RuleRegistry, RuleSpec


class RuleInferenceTest(unittest.TestCase):
    def test_output_mapped_is_union_of_input_mapping(self):
        reg = RuleRegistry()
        # gather：共享表（无映射）+ 逐样本索引（两层映射）
        facts = [
            {"mapped": set(), "is_ragged": False, "layers": None},
            {"mapped": {"replica", "sample"}, "is_ragged": True,
             "layers": ("replica", "sample")},
        ]
        d = reg.infer("gather", ("replica", "sample"), facts, True)
        self.assertEqual(set(d.output_mapped), {"replica", "sample"})
        self.assertTrue(d.fully_covered)
        self.assertIn("lookup", d.shared_preserved)

    def test_shared_input_does_not_become_mapped(self):
        reg = RuleRegistry()
        facts = [
            {"mapped": set(), "is_ragged": False, "layers": None},
        ]
        d = reg.infer("add", ("replica", "sample"), facts, False)
        # 纯共享计算不产生映射轴
        self.assertEqual(d.output_mapped, ())

    def test_rule_requiring_shared_axis_mapped_is_violation(self):
        # 恶意规则：声称把 lookup 当映射轴
        bad = RuleSpec(
            "bad.gather", "gather", ("replica", "sample"),
            frozenset({"lookup"}),
        )
        reg = RuleRegistry(extra={"gather": bad})
        facts = [
            {"mapped": {"lookup"}, "is_ragged": False, "layers": None,
             "mapped_axes": {"lookup"}},
        ]
        d = reg.infer("gather", ("replica", "sample"), facts, True)
        self.assertTrue(d.violations)
        with self.assertRaises(sb.RuleViolation):
            reg.assert_no_violation(d)

    def test_partial_coverage_splits_covered_and_missing_layers(self):
        # 只覆盖 sample 一层的规则
        partial = RuleSpec(
            "partial.reduce", "reduce", ("sample",), frozenset()
        )
        reg = RuleRegistry(extra={"reduce": partial})
        facts = [
            {"mapped": {"replica", "sample"}, "is_ragged": True,
             "layers": ("replica", "sample")},
        ]
        d = reg.infer("reduce", ("replica", "sample"), facts, False)
        self.assertEqual(d.covered_layers, ("sample",))
        self.assertEqual(d.missing_layers, ("replica",))
        self.assertFalse(d.fully_covered)


class ExecutionBoundaryTest(unittest.TestCase):
    def test_invalid_index_in_valid_segment_raises_with_coordinates(self):
        # 构造一个有效位上越界、且不被掩码掩盖的索引
        seg = sb.RaggedSegments.from_valid(
            ("sample",), (1,), [np.array([True, True])]
        )
        table = sb.LogicalTensor.shared(
            "t", ("lookup", "feature"), np.zeros((3, 2))
        )
        idx = sb.LogicalTensor.ragged(
            "idx", ("sequence",), np.array([0, 99]), seg, "sequence"
        )
        tokens = sb.LogicalTensor.ragged(
            "tok", ("sequence", "feature"), np.zeros((2, 2)), seg
        )
        graph = sb.Graph([
            sb.Node("e", "gather", ("t", "idx"), axis="lookup"),
        ])
        eng = sb.Engine(graph)
        with self.assertRaises(sb.InvalidIndexError) as cm:
            eng.run(("sample",), (1,), {"t": table, "idx": idx}, ["e"])
        self.assertEqual(cm.exception.index, 99)
        self.assertEqual(cm.exception.position, 1)

    def test_masked_bad_index_does_not_error(self):
        # 同一越界整数被掩码排除 -> 物理剔除后插件/索引都看不到
        seg = sb.RaggedSegments.from_valid(
            ("sample",), (1,), [np.array([True, False])]
        )
        table = sb.LogicalTensor.shared(
            "t", ("lookup", "feature"), np.arange(6).reshape(3, 2).astype(float)
        )
        idx = sb.LogicalTensor.ragged(
            "idx", ("sequence",), np.array([1]), seg, "sequence"
        )
        graph = sb.Graph([
            sb.Node("e", "gather", ("t", "idx"), axis="lookup"),
        ])
        eng = sb.Engine(graph)
        out = eng.run(("sample",), (1,), {"t": table, "idx": idx}, ["e"])
        np.testing.assert_array_equal(out["e"].data, table.data[[1]])


if __name__ == "__main__":
    unittest.main()
