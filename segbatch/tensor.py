"""逻辑张量：命名轴 + 映射轴事实 + 两种物理形态。

物理形态
--------
* **稠密**：映射层成为前置维，形状为 ``映射维... + 单样本逻辑维...``。
  共享张量没有任何映射维；按 replica 取出的张量带 ``replica`` 维；
  逐样本张量带嵌套的映射维。
* **分段（ragged）**：sequence 轴不做稠密补齐，所有有效行沿第 0 维压平，
  分段布局由 :class:`RaggedSegments` 描述。空段没有行，而不是一行零。

轴名在一个张量声明中必须唯一；负轴按单样本逻辑张量的秩索引，不计外层
已经映射掉的轴（《交换约定》）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .segments import RaggedSegments


@dataclass
class LogicalTensor:
    name: str
    axes: tuple                     # 单样本逻辑轴名（含 sequence 轴时为逻辑声明）
    data: np.ndarray                # 物理数组
    mapped: tuple = ()              # 本张量实际承载的映射层（物理嵌套顺序）
    segments: RaggedSegments | None = None
    seq_axis: str | None = None     # ragged 时，逻辑 sequence 轴的名字
    # 稠密不适用位置：被标记组合 id 的输出保持无结果（如空段 mean）
    unavailable: frozenset = frozenset()
    # 引擎挂接的完整映射网格（用于把组合 id 解码到稠密前置维）
    grid: RaggedSegments | None = None

    def __post_init__(self):
        self.axes = tuple(self.axes)
        self.mapped = tuple(self.mapped)
        if len(set(self.axes)) != len(self.axes):
            raise ValueError(f"张量 {self.name!r} 的轴名必须唯一：{self.axes}")
        if self.segments is not None and self.seq_axis not in self.axes:
            raise ValueError("seq_axis 必须出现在逻辑轴声明中")
        if self.segments is None and self.data.ndim != len(self.axes) + len(self.mapped):
            raise ValueError(
                f"张量 {self.name!r} 物理秩 {self.data.ndim} 与声明 "
                f"({len(self.mapped)} 映射 + {len(self.axes)} 逻辑) 不一致"
            )
        if self.segments is not None:
            rest = [a for a in self.axes if a != self.seq_axis]
            if self.data.ndim != 1 + len(rest):
                raise ValueError(
                    f"ragged 张量 {self.name!r} 数据形状 {self.data.shape} "
                    f"与非序列逻辑轴 {rest} 不一致"
                )

    # ---- 轴解析 ---------------------------------------------------------
    def index(self, axis: str) -> int:
        return self.axes.index(axis)

    def resolve_axis(self, axis) -> int:
        """把名称或负数解析为单样本逻辑轴下标（映射维不参与）。"""
        if isinstance(axis, str):
            return self.axes.index(axis)
        rank = len(self.axes)
        i = int(axis)
        if i < 0:
            i += rank
        if not 0 <= i < rank:
            raise ValueError(f"轴 {axis} 超出逻辑秩 {rank}")
        return i

    # ---- 映射事实 -------------------------------------------------------
    @property
    def is_ragged(self) -> bool:
        return self.segments is not None

    @property
    def data_axes(self) -> tuple:
        """除去 sequence 轴后的逻辑轴。"""
        return tuple(a for a in self.axes if a != self.seq_axis)

    def mapped_facts(self) -> dict:
        """供规则推断使用的映射事实。

        ``mapped_axes`` 列出被当成映射轴的逻辑轴名（正常应为空——映射身份
        只有 replica/sample，逻辑轴保持共享）。规则据此识别“错误地要求共享
        轴变成映射轴”的违约。
        """
        mapped_axes = set()
        if self.is_ragged:
            # ragged 的 sequence 是被映射层承载的逐样本轴，需显式告知规则
            mapped_axes.add(self.seq_axis)
        return {
            "mapped": set(self.mapped),
            "is_ragged": self.is_ragged,
            "layers": self.segments.layers if self.is_ragged else None,
            "mapped_axes": mapped_axes,
            "logical_axes": set(self.axes),
        }

    def combo_view(self, combo_id: int) -> np.ndarray:
        """取某个映射组合的单样本物理视图。"""
        if self.is_ragged:
            return self.data[self.segments.slice(combo_id)]
        if not self.mapped:
            return self.data
        if self.grid is None:
            raise RuntimeError(f"稠密张量 {self.name!r} 缺少引擎挂接的映射网格")
        full_coords = self.grid.decode(combo_id)
        coords = tuple(
            full_coords[self.grid.layers.index(layer)] for layer in self.mapped
        )
        return self.data[coords]

    # ---- 工厂 -----------------------------------------------------------
    @classmethod
    def shared(cls, name: str, axes: tuple, values: np.ndarray) -> "LogicalTensor":
        return cls(name=name, axes=tuple(axes), data=np.asarray(values), mapped=())

    @classmethod
    def dense(
        cls,
        name: str,
        axes: tuple,
        values: np.ndarray,
        mapped: tuple,
        grid: RaggedSegments | None = None,
    ) -> "LogicalTensor":
        return cls(
            name=name,
            axes=tuple(axes),
            data=np.asarray(values),
            mapped=tuple(mapped),
            grid=grid,
        )

    @classmethod
    def ragged(
        cls,
        name: str,
        axes: tuple,
        flat_rows: np.ndarray,
        segments: RaggedSegments,
        seq_axis: str = "sequence",
    ) -> "LogicalTensor":
        return cls(
            name=name,
            axes=tuple(axes),
            data=np.asarray(flat_rows),
            mapped=segments.layers,
            segments=segments,
            seq_axis=seq_axis,
            grid=segments,
        )


def make_ragged(
    name: str,
    axes: tuple,
    seq_axis: str,
    rows_per_combo,
    segments: RaggedSegments,
    *,
    dtype=np.float64,
) -> LogicalTensor:
    """按分段布局拼出扁平 ragged 张量。

    ``rows_per_combo[c]`` 为组合 c 的单样本数组（第 0 维是 sequence），
    空段 / 全掩码组合传空数组或 None；掩码位必须在调用前剔除。
    """
    rest_axes = [a for a in axes if a != seq_axis]
    trailing = tuple(
        # 形状由第一个非空段确定；全空时退化为零个逻辑轴
        np.asarray(rows).shape[1:]
        for rows in rows_per_combo
        if rows is not None and len(np.asarray(rows))
    )
    trailing_shape = trailing[0] if trailing else (0,) * len(rest_axes)
    chunks = []
    for c in range(segments.combos):
        n = segments.segment_length(c)
        if n == 0:
            continue
        arr = np.asarray(rows_per_combo[c], dtype=dtype)
        if arr.shape[0] != n:
            raise ValueError(
                f"{name} 组合 {c} 给了 {arr.shape[0]} 行，分段只认 {n} 个有效行"
            )
        chunks.append(arr)
    flat = (
        np.concatenate(chunks, axis=0)
        if chunks
        else np.empty((0,) + trailing_shape, dtype=dtype)
    )
    return LogicalTensor.ragged(name, axes, flat, segments, seq_axis)
