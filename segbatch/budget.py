"""预算与工作区计量。

《交换约定》对预算有两类口径：

1. ``fallback_budget``：无向量化规则且实际调用的完整单样本组合数。
   空段、被掩码排除的组合不消耗；已有规则覆盖的部分也不消耗。
2. 存储计量：重排、实际新增的张量视图、插件回退临时量、反向临时量都计入；
   用户输入与主动返回的有效结果不计入。

所有引擎内部分配都必须经过 :class:`BudgetMeter`，这样审计代码才能独立证明
没有按最长轨迹做无界补齐：任何一次分配的形状都会留下记录。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import BudgetExhausted, SegbatchError


@dataclass
class AllocationRecord:
    shape: tuple
    dtype: str
    kind: str           # input_staging / temp / backward_temp / output / view / shard
    nbytes: int
    note: str = ""


class WorkspaceLimit(SegbatchError):
    def __init__(self, requested, limit, shape):
        self.requested = requested
        self.limit = limit
        self.shape = shape
        super().__init__(
            f"工作区超限：需要 {requested} 字节，上限 {limit}，形状 {shape}"
        )


class BudgetMeter:
    def __init__(self, fallback_budget: int | None = None, workspace_limit: int | None = None):
        self.fallback_budget = fallback_budget
        self.workspace_limit = workspace_limit
        self.fallback_calls = 0          # 前向逐样本组合调用（受预算约束）
        self.backward_calls = 0          # 反向伴随调用（计临时量，不耗前向预算）
        # 分类字节数（视图按名义字节单列，不与实分配混在一起）
        self.alloc_bytes = 0
        self.view_bytes = 0
        self.backward_temp_bytes = 0
        self.live_bytes = 0
        self.peak_live_bytes = 0
        self.records: list[AllocationRecord] = []
        self._live: dict[int, AllocationRecord] = {}

    # ---- 逐样本回退预算 -------------------------------------------------
    def charge_fallback(self, n: int, path: str) -> None:
        """登记 n 次实际发生的前向单样本组合调用；预算耗尽立即报错。"""
        if (
            self.fallback_budget is not None
            and self.fallback_calls + n > self.fallback_budget
        ):
            raise BudgetExhausted(
                needed=n,
                remaining=self.fallback_budget - self.fallback_calls,
                rule_path=path,
            )
        self.fallback_calls += n

    def charge_backward(self, n: int) -> None:
        """登记反向伴随调用次数。

        反向不消耗前向 ``fallback_budget``（那是前向完整单样本组合数的口径），
        但单独计数并体现在追踪与计量快照里。
        """
        self.backward_calls += n

    # ---- 存储计量 -------------------------------------------------------
    def alloc(
        self,
        shape: tuple,
        dtype=np.float64,
        *,
        kind: str = "temp",
        fill=None,
        note: str = "",
    ) -> np.ndarray:
        """分配一个被计量的数组。"""
        dtype = np.dtype(dtype)
        nbytes = (
            int(np.prod(shape)) * dtype.itemsize
            if len(shape)
            else dtype.itemsize
        )
        if (
            self.workspace_limit is not None
            and self.live_bytes + nbytes > self.workspace_limit
        ):
            raise WorkspaceLimit(self.live_bytes + nbytes, self.workspace_limit, shape)
        if fill is None:
            arr = np.empty(shape, dtype=dtype)
        elif fill == 0:
            arr = np.zeros(shape, dtype=dtype)
        else:
            arr = np.full(shape, fill, dtype=dtype)
        rec = AllocationRecord(
            tuple(int(s) for s in shape), dtype.str, kind, nbytes, note
        )
        self.records.append(rec)
        self.alloc_bytes += nbytes
        if kind == "backward_temp":
            self.backward_temp_bytes += nbytes
        self._live[id(arr)] = rec
        self.live_bytes += nbytes
        self.peak_live_bytes = max(self.peak_live_bytes, self.live_bytes)
        return arr

    def view(self, arr: np.ndarray, *, kind: str = "view", note: str = "") -> np.ndarray:
        """登记一个视图（不复制数据，但按《约定》计入名义新增）。"""
        nbytes = arr.size * arr.dtype.itemsize
        rec = AllocationRecord(tuple(arr.shape), arr.dtype.str, kind, nbytes, note)
        self.records.append(rec)
        self.view_bytes += nbytes
        return arr

    def release(self, arr: np.ndarray) -> None:
        rec = self._live.pop(id(arr), None)
        if rec is not None:
            self.live_bytes -= rec.nbytes

    def snapshot(self) -> dict:
        return {
            "fallback_budget": self.fallback_budget,
            "fallback_calls": self.fallback_calls,
            "backward_calls": self.backward_calls,
            "fallback_remaining": (
                None
                if self.fallback_budget is None
                else self.fallback_budget - self.fallback_calls
            ),
            "alloc_bytes": self.alloc_bytes,
            "view_bytes": self.view_bytes,
            "backward_temp_bytes": self.backward_temp_bytes,
            "peak_live_bytes": self.peak_live_bytes,
            "largest_alloc": max((r.nbytes for r in self.records), default=0),
            "allocations": list(self.records),
        }
