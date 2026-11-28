"""插件登记。

插件协议（《交换约定》）：可调用 Python 函数接收并返回 NumPy 张量。
算子作者可以登记：

* ``fn``            —— 只对**合法轨迹**（非空、掩码已剔除的单样本行）有定义；
* ``batched_fn``    —— 合格的批化回调（一次吃掉多个组合的有效行）；
* ``vjp``           —— 单样本伴随；
* ``batched_vjp``   —— 批化伴随。

只登记 ``fn`` 而没有批化回调的插件，只能在明确的逐样本预算内回退；
空段与被掩码排除的轨迹不会调用 ``fn``（见 engine 调度守卫）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class PluginSpec:
    name: str
    fn: Callable                 # 单样本前向 (q, p) -> out
    vjp: Callable | None = None  # 单样本伴随 (q, p, out, gout) -> (gq, gp)
    batched_fn: Callable | None = None    # 批化前向
    batched_vjp: Callable | None = None   # 批化伴随
    # 批化规则实际覆盖的映射层；None 表示覆盖输入承载的全部映射层。
    # 只声明部分层时，引擎对覆盖层走批化、缺失层在逐样本预算内分片回退。
    covers_layers: tuple | None = None
    note: str = ""

    @property
    def has_vector_rule(self) -> bool:
        return self.batched_fn is not None


# ---- 默认插件：scaled_square ----------------------------------------------
def scaled_square(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """(q + p)^2。只接受非空的有效单样本行矩阵。"""
    if q.ndim != 2 or q.shape[0] == 0:
        raise ValueError("scaled_square 只对非空有效单样本行矩阵有定义")
    z = q + p
    return z * z


def scaled_square_vjp(q, p, out, gout):
    # d/dq (q+p)^2 = 2(q+p)，对 p 同样；gp 在单样本内按行求和到 p 的形状
    z = q + p
    g = 2.0 * z * gout
    return g, g.sum(axis=0)


def scaled_square_batched(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """批化版本：第 0 维是压平的有效行，内部即逐行数学，结果必须与逐样本一致。"""
    if q.shape[0] == 0:
        # 批化回调可以收到“整批为空”——此时无需触碰任何未定义轨迹
        return np.empty((0, q.shape[1]), dtype=q.dtype)
    z = q + p
    return z * z


def scaled_square_batched_vjp(q, p, out, gout):
    z = q + p
    g = 2.0 * z * gout
    return g, g.sum(axis=0)


DEFAULT_PLUGINS: dict[str, PluginSpec] = {
    "scaled_square": PluginSpec(
        name="scaled_square",
        fn=scaled_square,
        vjp=scaled_square_vjp,
        note="仅单样本规则；批化需回退或由作者补登记 batched_fn",
    ),
    "scaled_square_batched": PluginSpec(
        name="scaled_square_batched",
        fn=scaled_square,
        vjp=scaled_square_vjp,
        batched_fn=scaled_square_batched,
        batched_vjp=scaled_square_batched_vjp,
        note="作者登记的合格批化与伴随规则",
    ),
}


class PluginRegistry:
    def __init__(self, specs: dict[str, PluginSpec] | None = None):
        self._specs = dict(DEFAULT_PLUGINS)
        if specs:
            self._specs.update(specs)

    def register(self, spec: PluginSpec) -> None:
        self._specs[spec.name] = spec

    def get(self, name: str) -> PluginSpec:
        if name not in self._specs:
            raise KeyError(f"未登记的插件 {name!r}")
        return self._specs[name]

    def names(self) -> list[str]:
        return sorted(self._specs)
