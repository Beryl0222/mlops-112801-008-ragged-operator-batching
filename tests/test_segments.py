"""分段表示与映射身份的基础语义。"""

import unittest

import numpy as np

from tests import _bootstrap  # noqa: F401
from segbatch import RaggedSegments
from segbatch.errors import SegbatchError


class SegmentsTest(unittest.TestCase):
    def setUp(self):
        # 2 副本 x 3 样本；样本 1 为空，样本 2 带掩码
        self.valid = [
            [True, True],          # s0
            [],                    # s1 空段
            [True, False, True],   # s2 掩码中间位
        ]
        self.grid = (2, 3)
        self.seg = RaggedSegments.from_valid(
            ("replica", "sample"), self.grid,
            self.valid * 2,
        )

    def test_masked_rows_physically_removed(self):
        # 每副本有效行 = 2 + 0 + 2 = 4；两副本共 8
        self.assertEqual(self.seg.total_valid_rows, 8)
        np.testing.assert_array_equal(
            self.seg.lengths.ravel(), [2, 0, 2, 2, 0, 2]
        )

    def test_empty_segment_is_not_a_zero_row(self):
        # 空段长度为 0，offsets 区间为空，而不是一行零
        self.assertEqual(self.seg.segment_length(1), 0)
        s = self.seg.slice(1)
        self.assertEqual(s.start, s.stop)
        self.assertEqual(set(self.seg.empty_combo_ids.tolist()), {1, 4})

    def test_positions_preserve_original_index(self):
        # 样本 2 掩码了位置 1，保留下来的原始位置应为 0 和 2
        s2 = self.seg.slice(2)
        np.testing.assert_array_equal(
            self.seg.positions[s2], [0, 2]
        )

    def test_nesting_order_changes_physical_layout_not_identity(self):
        # (sample, replica)：网格 3 x 2，组合按 sample 外层展开
        valid_b = [v for s in range(3) for _ in range(2)
                   for v in [self.valid[s]]]
        seg_b = RaggedSegments.from_valid(
            ("sample", "replica"), (3, 2), valid_b,
        )
        # 同样的有效行总数
        self.assertEqual(seg_b.total_valid_rows, 8)
        # 扁平组合 id 解码出的身份坐标随嵌套顺序变化，但身份本身一致
        self.assertEqual(self.seg.decode(0), (0, 0))
        self.assertEqual(seg_b.decode(0), (0, 0))
        self.assertEqual(seg_b.decode(1), (0, 1))
        # 反向编码：replica=1,sample=0 在两种布局下 id 不同
        self.assertEqual(self.seg.encode((1, 0)), 3)
        self.assertEqual(seg_b.encode((0, 1)), 1)

    def test_offsets_representation_roundtrip(self):
        offsets = self.seg.offsets
        seg2 = RaggedSegments.from_offsets(
            self.seg.layers, self.seg.grid, offsets
        )
        np.testing.assert_array_equal(seg2.lengths, self.seg.lengths)
        np.testing.assert_array_equal(seg2.offsets, offsets)

    def test_provenance_rows(self):
        prov = self.seg.provenance_rows()
        self.assertEqual(len(prov), 8)
        self.assertEqual(prov[0], {
            "flat_row": 0, "combo": 0,
            "replica": 0, "sample": 0, "sequence_position": 0,
        })

    def test_non_monotonic_offsets_rejected(self):
        with self.assertRaises(ValueError):
            RaggedSegments.from_offsets(
                ("replica",), (2,), np.array([0, 2, 1], dtype=np.int64)
            )


if __name__ == "__main__":
    unittest.main()
