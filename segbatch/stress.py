"""压力资料生成与“无界补齐”审计。

压力资料的索引从可用共享行中以 PCG64 按 seed 生成（《交换约定》）。
绝大多数样本是短轨迹，少数是长轨迹；如果按最长轨迹稠密补齐，工作区会
超过 ``workspace_bytes``，而分段表示保持在预算内——这正是 ragged 执行
必要性的量化证据。
"""

from __future__ import annotations

import numpy as np

from .segments import RaggedSegments


def generate_stress(spec: dict, *, layers=("replica", "sample")):
    """按 stress 配置生成合成记录。

    返回 dict，结构与交换资料的 examples/tensors 对齐。
    索引只从 ``lookup_size`` 个共享行中抽取，因此天然合法；掩码位上可以
    放置越界整数，用来证明掩码位不会被索引算子读到。
    """
    rng = np.random.default_rng(np.random.PCG64(spec["seed"]))
    n_ex = spec["examples"]
    n_rep = spec["replicas"]
    feature = spec["feature"]
    short = spec["short_length"]
    long = spec["long_length"]
    long_set = set(spec["long_examples"])
    lookup_size = max(7, feature)

    examples = []
    for s in range(n_ex):
        length = long if s in long_set else short
        tokens = rng.normal(size=(length, feature))
        valid = np.ones(length, dtype=bool)
        # 在短轨迹里随机掩码一些位置（不动长轨迹，保证对照清晰）
        if length == short and length > 1:
            n_mask = int(rng.integers(0, max(1, length // 2 + 1)))
            for _ in range(n_mask):
                valid[int(rng.integers(0, length))] = False
        indices = rng.integers(0, lookup_size, size=length).astype(np.int64)
        # 掩码位写入越界整数：若掩码失效会立即触发 InvalidIndexError
        for pos in np.flatnonzero(~valid):
            indices[pos] = lookup_size + 1000
        examples.append(
            {
                "id": f"s{s}",
                "tokens": tokens.tolist(),
                "valid": valid.tolist(),
                "indices": indices.tolist(),
            }
        )

    shared = rng.normal(size=(lookup_size, feature))
    weight = rng.normal(size=(n_rep, feature, feature))
    posbias = rng.normal(size=feature)
    return {
        "replicas": [f"r{i}" for i in range(n_rep)],
        "examples": examples,
        "tensors": {
            "shared": {"axes": ["lookup", "feature"], "values": shared.tolist()},
            "weight": {
                "axes": ["replica", "feature", "filter"],
                "values": weight.tolist(),
            },
            "posbias": {"axes": ["filter"], "values": posbias.tolist()},
        },
        "_lookup_size": lookup_size,
    }


def build_stress_segments(data, layers=("replica", "sample")):
    n_rep = len(data["replicas"])
    n_smp = len(data["examples"])
    grid = (n_rep, n_smp) if tuple(layers) == ("replica", "sample") \
        else (n_smp, n_rep)
    valid_per_combo = []
    import itertools
    for coords in itertools.product(*(range(g) for g in grid)):
        s = coords[layers.index("sample")]
        valid_per_combo.append(
            np.asarray(data["examples"][s]["valid"], dtype=bool)
        )
    return RaggedSegments.from_valid(layers, grid, valid_per_combo)


def padding_audit(segments: RaggedSegments, feature_dim: int, itemsize: int = 8):
    """对比“按最长轨迹补齐”与“分段扁平”两种存储的字节数。"""
    combos = segments.combos
    longest = int(segments.lengths.max()) if combos else 0
    padded_bytes = combos * longest * feature_dim * itemsize
    ragged_bytes = segments.total_valid_rows * feature_dim * itemsize
    return {
        "combos": combos,
        "longest_length": longest,
        "padded_bytes": padded_bytes,
        "ragged_bytes": ragged_bytes,
        "saved_ratio": (
            1.0 - ragged_bytes / padded_bytes if padded_bytes else 0.0
        ),
        "padded_dense_shape": segments.padded_dense_shape(),
    }


class PluginReadAudit:
    """包装插件，记录它每次实际读到的行数与最大值，独立证明无补齐。

    若引擎把任何掩码/补齐行送进插件，这里记录的行数会超过分段声明的
    有效行数，审计即失败。
    """

    def __init__(self, fn):
        self.fn = fn
        self.calls = 0
        self.max_rows_seen = 0
        self.total_rows_seen = 0

    def __call__(self, q, p):
        self.calls += 1
        self.max_rows_seen = max(self.max_rows_seen, q.shape[0])
        self.total_rows_seen += q.shape[0]
        return self.fn(q, p)
