"""广播、重复位置的 gather/scatter 伴随与归约细粒度语义。"""

import unittest

import numpy as np

from tests import _bootstrap  # noqa: F401
import segbatch as sb
from segbatch import ops
from segbatch.segments import RaggedSegments


def _seg():
    return RaggedSegments.from_valid(
        ("sample",), (1,), [np.array([True, True, True])]
    )


class GatherScatterSemanticsTest(unittest.TestCase):
    def test_gather_preserves_occurrences_and_scatter_sums_them(self):
        seg = _seg()
        # 重复位置 1 出现两次
        table = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        indices = np.array([1, 1, 0])
        gathered = ops.batch_gather(
            table, indices, axis=0, axis_name="lookup",
            positions=seg.positions, combo_ids=seg.combo_ids,
        )
        # 保留发生次数
        np.testing.assert_array_equal(
            gathered, [[3.0, 4.0], [3.0, 4.0], [1.0, 2.0]]
        )
        # scatter_add 把两次发生相加而非覆盖：槽 0 一次、槽 1 两次
        rows = np.ones((3, 2))
        scattered = ops.batch_scatter_add(
            rows, indices, seg.combo_ids, seg, size=3, axis=0
        )
        np.testing.assert_array_equal(
            scattered[0], [[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]]
        )

    def test_gather_vjp_accumulates_duplicate_positions(self):
        table_shape = (3, 2)
        indices = np.array([1, 1, 0])
        gout = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        g_table = ops.gather_vjp(gout, indices, table_shape, axis=0)
        # 位置 0 收到第 3 次发生（值 3）；位置 1 累计前两次发生（1+2=3）
        np.testing.assert_array_equal(
            g_table, [[3.0, 3.0], [3.0, 3.0], [0.0, 0.0]]
        )

    def test_scatter_vjp_reads_per_occurrence(self):
        seg = _seg()
        gout_grid = np.array([[[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]])
        idx_t = sb.LogicalTensor.ragged(
            "idx", ("sequence",), np.array([1, 1, 0]), seg
        )
        g_rows = ops.scatter_vjp(gout_grid, idx_t, seg, axis=0)
        # 重复位置的伴随按各次发生取回相同的槽梯度
        np.testing.assert_array_equal(
            g_rows, [[30.0, 40.0], [30.0, 40.0], [10.0, 20.0]]
        )

    def test_empty_sum_is_zero_but_mean_max_unavailable(self):
        seg = RaggedSegments.from_valid(
            ("sample",), (2,), [np.array([True, True]), np.array([], dtype=bool)]
        )
        flat = np.array([[1.0, 2.0], [3.0, 4.0]])
        for mode, expected in [
            ("sum", [[4.0, 6.0]]),
        ]:
            out, unavail = ops.batch_segment_reduce(flat, seg.offsets, mode)
            np.testing.assert_array_equal(out[0], expected[0])
            np.testing.assert_array_equal(out[1], [0.0, 0.0])
            self.assertEqual(unavail, set())
        for mode in ("mean", "max"):
            out, unavail = ops.batch_segment_reduce(flat, seg.offsets, mode)
            self.assertEqual(unavail, {1})
            # 不适用位置是零占位而不是 NaN
            self.assertTrue(np.all(np.isfinite(out[1])))

    def test_max_tie_splits_gradient_equally(self):
        seg = _seg()
        flat = np.array([[2.0, 1.0], [2.0, 5.0], [0.0, 5.0]])
        gout = np.ones((1, 2))
        gx = ops.reduce_vjp_flat(gout, flat, seg.offsets, "max")
        # 第一列两个并列最大各 0.5；第二列两个并列 5 各 0.5
        np.testing.assert_allclose(
            gx, [[0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]
        )


class BroadcastTest(unittest.TestCase):
    def test_numpy_position_broadcast_in_add(self):
        seg = RaggedSegments.from_valid(
            ("sample",), (1,), [np.array([True, True])]
        )
        a = sb.LogicalTensor.ragged(
            "a", ("sequence", "feature"),
            np.array([[1.0, 2.0], [3.0, 4.0]]), seg,
        )
        b = sb.LogicalTensor.shared("b", ("feature",), np.array([10.0, 20.0]))
        graph = sb.Graph([sb.Node("y", "add", ("a", "b"))])
        eng = sb.Engine(graph)
        out = eng.run(("sample",), (1,), {"a": a, "b": b}, ["y"])
        np.testing.assert_array_equal(
            out["y"].data, [[11.0, 22.0], [13.0, 24.0]]
        )
        # 共享偏置的梯度把两行贡献相加
        grads = eng.backward({"y": np.ones_like(out["y"].data)})
        np.testing.assert_array_equal(grads["b"]["data"], [2.0, 2.0])
        np.testing.assert_array_equal(grads["a"]["data"], np.ones((2, 2)))


if __name__ == "__main__":
    unittest.main()
