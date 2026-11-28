"""分段批化库的显式错误类型。

所有违反《交换约定》的情况都抛出具体异常，而不是用掩码把错误吞掉：
- 合法轨迹上的非法位置索引 -> InvalidIndexError
- 逐样本回退预算耗尽 -> BudgetExhausted
- 向量化规则要求把共享轴当作映射轴 / 只覆盖部分嵌套且无法分片 -> RuleViolation
- 在空段或被掩码排除的轨迹上调用只对合法轨迹有定义的插件 -> UndefinedTrajectory（属于内部守卫）
"""

from __future__ import annotations

from dataclasses import dataclass


class SegbatchError(Exception):
    """本库所有异常的基类。"""


class GraphError(SegbatchError):
    """算子图声明本身不合法（未知算子、轴名缺失、环等）。"""


class InvalidIndexError(SegbatchError):
    """有效段内出现越界位置整数。

    非法位置只有出现在有效段时才算错误；被掩码排除的位置根本不会被读到。
    """

    def __init__(self, *, index: int, limit: int, combo: int, position: int, axis: str):
        self.index = index
        self.limit = limit
        self.combo = combo
        self.position = position
        self.axis = axis
        super().__init__(
            f"轴 {axis!r} 上的位置 {index} 越界（合法区间 [0,{limit})）；"
            f"发生在映射组合 #{combo} 的原序列位置 {position}"
        )


class BudgetExhausted(SegbatchError):
    """无向量化规则可用，且剩余逐样本预算不足以覆盖实际需要调用的组合。"""

    def __init__(self, *, needed: int, remaining: int, rule_path: str):
        self.needed = needed
        self.remaining = remaining
        self.rule_path = rule_path
        super().__init__(
            f"逐样本预算不足：该路径还需 {needed} 次完整单样本组合调用，"
            f"剩余预算只有 {remaining}（路径：{rule_path}）"
        )


class RuleViolation(SegbatchError):
    """登记的向量化规则与实际张量的映射事实冲突。

    典型情况：规则要求把某个共享轴当成映射轴；或规则声明的映射层与本次
    请求的嵌套结构对不上，且无法通过分片继续执行。
    """

    def __init__(self, *, rule: str, reason: str, details: dict | None = None):
        self.rule = rule
        self.reason = reason
        self.details = details or {}
        super().__init__(f"规则 {rule!r} 违反映射约定：{reason}（细节：{self.details}）")


class UndefinedTrajectory(SegbatchError):
    """内部守卫：插件被发到了空段或全掩码轨迹。

    正常调度下该异常不应发生；它的存在是为了证明“先掩码后调用”的顺序
    是被执行层面强制的，而不是靠插件内部容忍。
    """


class UnavailableResult(SegbatchError):
    """对不适用的输出位置强行取值（例如对空段取 mean/max）。"""
