"""分段（ragged sequence）表示。

映射身份有两个：``replica``（模型副本）与 ``sample``（样本）。每个请求选择
映射层的嵌套顺序，物理存放的维序不能改变身份——因此分段对象始终显式携带
层名，而不是依赖数组维度的位置约定。

关键语义（见《交换约定》）：

* ``examples`` 表示 ragged 的 sequence 轴；每个 sample 给一个序列和等长的
  ``valid`` 记录，另外允许用一维整数 ``offsets`` 表示。
* 值为 false 的位置不送入插件、索引算子和有效归约；该位保持无结果。
* 空段（没有任何有效行的映射组合）不是一段值为零的序列——它没有行，
  在调度上与“有行但全为掩码”的组合一样属于无结果域，而不是零贡献域。
* 每个有效行都带原始序列位置，输出位置可以回溯到具体 sample 与位置。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RaggedSegments:
    """映射网格上的分段布局。

    属性
    ------
    layers:
        映射层名，按物理嵌套（外层 -> 内层）排列，例如 ``("replica", "sample")``。
    grid:
        各层的大小，与 ``layers`` 对齐。
    lengths:
        形状为 ``grid`` 的整数数组，元素是该映射组合的**有效行数**
        （掩码位已剔除），空段与全掩码组合均为 0。
    offsets:
        长度为 ``combos + 1`` 的一维整数数组，把组合 id 映射到扁平数据中的
        行区间 ``[offsets[c], offsets[c+1])``。
    positions:
        长度为 ``total_valid_rows`` 的整数数组，记录每行在其原始单样本序列中
        的位置（掩码前），用于输出位置回溯。
    combo_ids:
        长度为 ``total_valid_rows`` 的整数数组，记录每行所属的映射组合 id。
    """

    layers: tuple
    grid: tuple
    lengths: np.ndarray
    offsets: np.ndarray
    positions: np.ndarray
    combo_ids: np.ndarray

    # ---- 基本量 ---------------------------------------------------------
    @property
    def combos(self) -> int:
        return int(np.prod(self.grid))

    @property
    def total_valid_rows(self) -> int:
        return int(self.offsets[-1])

    @property
    def active_combo_ids(self) -> np.ndarray:
        """至少有一个有效行的组合 id。"""
        return np.flatnonzero(self.lengths.reshape(-1))

    @property
    def empty_combo_ids(self) -> np.ndarray:
        """没有有效行的组合 id（空段 / 全掩码）。"""
        return np.flatnonzero(self.lengths.reshape(-1) == 0)

    def decode(self, combo_id: int) -> tuple:
        """把扁平组合 id 解码成各映射层的坐标。"""
        coords = ()
        x = int(combo_id)
        for size in reversed(self.grid):
            coords = (x % size,) + coords
            x //= size
        return coords

    def encode(self, coords: tuple) -> int:
        """把各映射层坐标编码成扁平组合 id（行优先，与物理嵌套一致）。"""
        cid = 0
        for size, coord in zip(self.grid, coords):
            cid = cid * size + int(coord)
        return cid

    def layer_value(self, combo_id: int, layer: str):
        return self.decode(combo_id)[self.layers.index(layer)]

    def slice(self, combo_id: int) -> slice:
        return slice(int(self.offsets[combo_id]), int(self.offsets[combo_id + 1]))

    def segment_length(self, combo_id: int) -> int:
        return int(self.lengths.reshape(-1)[combo_id])

    # ---- 构造 -----------------------------------------------------------
    @classmethod
    def from_valid(cls, layers: tuple, grid: tuple, valid_per_combo) -> "RaggedSegments":
        """从每个映射组合的有效标记构造。

        ``valid_per_combo[c]`` 是组合 c 的布尔序列（按原始位置给出）。
        掩码位在此被物理剔除；被剔除行的原始位置由 ``positions`` 保留。
        """
        layers = tuple(layers)
        grid = tuple(int(g) for g in grid)
        combos = int(np.prod(grid))
        if len(valid_per_combo) != combos:
            raise ValueError(f"期望 {combos} 个组合的 valid，实际 {len(valid_per_combo)}")
        lengths = np.empty(combos, dtype=np.int64)
        pos_chunks = []
        combo_chunks = []
        for c, valid in enumerate(valid_per_combo):
            valid = np.asarray(valid, dtype=bool)
            (good,) = np.nonzero(valid)
            lengths[c] = good.size
            if good.size:
                pos_chunks.append(good.astype(np.int64))
                combo_chunks.append(np.full(good.size, c, dtype=np.int64))
        offsets = np.zeros(combos + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        positions = (
            np.concatenate(pos_chunks).astype(np.int64)
            if pos_chunks
            else np.empty(0, dtype=np.int64)
        )
        combo_ids = (
            np.concatenate(combo_chunks).astype(np.int64)
            if combo_chunks
            else np.empty(0, dtype=np.int64)
        )
        return cls(
            layers=layers,
            grid=grid,
            lengths=lengths.reshape(grid),
            offsets=offsets,
            positions=positions,
            combo_ids=combo_ids,
        )

    @classmethod
    def from_offsets(
        cls,
        layers: tuple,
        grid: tuple,
        offsets: np.ndarray,
        *,
        positions: np.ndarray | None = None,
    ) -> "RaggedSegments":
        """用 NumPy 一维整数 offsets 直接构造（《约定》允许的另一种表示）。"""
        offsets = np.asarray(offsets, dtype=np.int64)
        combos = int(np.prod(grid))
        if offsets.shape != (combos + 1,):
            raise ValueError("offsets 长度必须等于组合数 + 1")
        if np.any(np.diff(offsets) < 0):
            raise ValueError("offsets 必须单调不减")
        lengths = np.diff(offsets)
        total = int(offsets[-1])
        if positions is None:
            # 无掩码压缩时原始位置即段内位置
            positions = np.concatenate(
                [np.arange(n, dtype=np.int64) for n in lengths]
            ) if total else np.empty(0, dtype=np.int64)
        combo_ids = np.repeat(np.arange(combos, dtype=np.int64), lengths)
        return cls(
            layers=tuple(layers),
            grid=tuple(int(g) for g in grid),
            lengths=lengths.reshape(grid),
            offsets=offsets,
            positions=np.asarray(positions, dtype=np.int64),
            combo_ids=combo_ids,
        )

    # ---- 审计 -----------------------------------------------------------
    def padded_dense_shape(self) -> tuple:
        """若按最长轨迹补齐，稠密缓冲需要的形状——仅用于对照审计，不用于执行。"""
        return tuple(self.grid) + (int(self.lengths.max()) if self.combos else 0,)

    def provenance_rows(self) -> list[dict]:
        """逐行的来源记录：输出行 -> 映射层坐标 + 原始序列位置。"""
        out = []
        for row, cid in enumerate(self.combo_ids):
            coords = self.decode(int(cid))
            out.append(
                {
                    "flat_row": row,
                    "combo": int(cid),
                    **{layer: int(v) for layer, v in zip(self.layers, coords)},
                    "sequence_position": int(self.positions[row]),
                }
            )
        return out
