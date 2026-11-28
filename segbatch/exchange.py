"""把交换资料（``batched_axes.json`` 一类的记录）装成引擎输入。

资料中的图只是交换样例：本模块只负责字段到逻辑张量的机械转换，
模型组完全可以拿另一张图、另一种批形状继续调用。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .engine import Graph
from .segments import RaggedSegments
from .tensor import LogicalTensor, make_ragged


def load_exchange(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_segments(data: dict, layers: tuple) -> RaggedSegments:
    """按请求的嵌套层构造分段；掩码位在此时物理剔除。"""
    n_rep = len(data["replicas"])
    n_smp = len(data["examples"])
    if tuple(layers) == ("replica", "sample"):
        grid = (n_rep, n_smp)
    elif tuple(layers) == ("sample", "replica"):
        grid = (n_smp, n_rep)
    else:
        raise ValueError(f"不支持的嵌套层 {layers}")
    valid_per_combo = []
    # 组合按行优先与 layers 对齐；valid 只取决于 sample
    for coords in _iter_grid(layers, grid):
        sample_idx = coords[layers.index("sample")]
        valid_per_combo.append(np.asarray(
            data["examples"][sample_idx]["valid"], dtype=bool
        ))
    return RaggedSegments.from_valid(layers, grid, valid_per_combo)


def _iter_grid(layers, grid):
    import itertools
    for coords in itertools.product(*(range(g) for g in grid)):
        yield coords


def build_inputs(data: dict, segments: RaggedSegments) -> dict[str, LogicalTensor]:
    examples = data["examples"]
    # ---- 逐样本 ragged：tokens、indices ----------------------------------
    token_rows, index_rows = [], []
    for coords in _iter_grid(segments.layers, segments.grid):
        s = coords[segments.layers.index("sample")]
        ex = examples[s]
        valid = np.asarray(ex["valid"], dtype=bool)
        tokens = np.asarray(ex["tokens"], dtype=np.float64)
        indices = np.asarray(ex["indices"], dtype=np.int64)
        token_rows.append(tokens[valid] if tokens.size else tokens)
        index_rows.append(indices[valid] if indices.size else indices)
    tokens_t = make_ragged(
        "tokens", ("sequence", "feature"), "sequence", token_rows, segments
    )
    flat_idx = (
        np.concatenate([r for r in index_rows if r.size], dtype=np.int64)
        if any(r.size for r in index_rows)
        else np.empty(0, dtype=np.int64)
    )
    indices_t = LogicalTensor.ragged(
        "indices", ("sequence",), flat_idx, segments, "sequence"
    )

    # ---- 共享 / 副本 / 共享偏置 ------------------------------------------
    tensors = data["tensors"]
    shared_t = LogicalTensor.shared(
        "shared", tuple(tensors["shared"]["axes"]),
        np.asarray(tensors["shared"]["values"], dtype=np.float64),
    )
    weight_t = LogicalTensor.dense(
        "weight", tuple(tensors["weight"]["axes"][1:]),
        np.asarray(tensors["weight"]["values"], dtype=np.float64),
        mapped=("replica",), grid=segments,
    )
    posbias_t = LogicalTensor.shared(
        "posbias", tuple(tensors["posbias"]["axes"]),
        np.asarray(tensors["posbias"]["values"], dtype=np.float64),
    )
    return {
        "tokens": tokens_t,
        "indices": indices_t,
        "shared": shared_t,
        "weight": weight_t,
        "posbias": posbias_t,
    }


def build_graph(data: dict) -> Graph:
    return Graph.from_program(data["programs"])
