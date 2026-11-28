"""向量化规则登记与映射轴推断。

规则要回答的问题（《交换约定》）：

1. 给定一组输入的实际映射事实，输出的哪些轴是**映射得到**的、哪些必须
   **保留共享**？
2. 该规则覆盖请求的哪些映射层？只覆盖部分嵌套时，覆盖部分与缺失部分
   分别判断（分片继续执行 vs. 逐样本回退）。
3. 规则是否错误地要求把共享轴变成映射轴？这种冲突在执行前判定为
   :class:`RuleViolation`，不能留下半次参数梯度。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .errors import RuleViolation


@dataclass
class RuleSpec:
    name: str
    op: str
    # 规则声称能向量化的映射层；空元组表示与映射无关的纯共享计算
    covers: tuple
    # 共享逻辑轴：规则承诺这些轴不会被当作映射轴
    shared_axes: frozenset = frozenset()
    note: str = ""


# 内置算子的合格向量化规则：两层映射（replica、sample）都可整批向量化
BUILTIN_RULES: dict[str, RuleSpec] = {
    "gather": RuleSpec(
        "gather.vector", "gather", ("replica", "sample"),
        frozenset({"lookup", "feature"}),
        "位置整数按有效行收集；表的 lookup/feature 轴保持共享",
    ),
    "add": RuleSpec(
        "add.vector", "add", ("replica", "sample"),
        frozenset(),
        "NumPy 命名轴广播；共享输入保持共享",
    ),
    "matmul": RuleSpec(
        "matmul.vector", "matmul", ("replica", "sample"),
        frozenset({"filter"}),
        "标准矩阵乘，共享/副本轴按逻辑名广播，收缩轴不映射",
    ),
    "reduce": RuleSpec(
        "reduce.vector", "reduce", ("replica", "sample"),
        frozenset(),
        "分段归约；空 sum 为零，空 mean/max 不适用",
    ),
    "scatter_add": RuleSpec(
        "scatter_add.vector", "scatter_add", ("replica", "sample"),
        frozenset({"lookup", "feature"}),
        "np.add.at 累加发生次数；lookup/feature 轴保持共享",
    ),
}


@dataclass
class MappingDecision:
    """一次算子调用的映射推断结果。"""

    output_mapped: tuple               # 输出承载的映射层
    output_ragged: bool
    covered_layers: tuple              # 本规则实际覆盖的映射层
    missing_layers: tuple              # 需要分片/回退的映射层
    rule_name: str
    shared_preserved: list = field(default_factory=list)
    violations: list = field(default_factory=list)

    @property
    def fully_covered(self) -> bool:
        return not self.missing_layers and not self.violations


class RuleRegistry:
    def __init__(self, extra: dict[str, RuleSpec] | None = None):
        self._rules = dict(BUILTIN_RULES)
        if extra:
            self._rules.update(extra)

    def get(self, op: str) -> RuleSpec:
        if op not in self._rules:
            raise KeyError(f"算子 {op!r} 没有登记向量化规则")
        return self._rules[op]

    def infer(
        self,
        op: str,
        request_layers: tuple,
        input_facts: list[dict],
        output_ragged: bool,
    ) -> MappingDecision:
        """根据输入的真实映射事实推断输出映射轴，并检查规则冲突。

        推断原则
        --------
        * 输出映射层 = 所有输入映射层的并集（任一输入按某层变化，输出就按该层变化）；
        * 某输入没有的映射层通过该层广播，不改变该输入的共享性质；
        * 规则的共享轴若出现在任何输入的映射层中，属于规则违约。
        """
        rule = self.get(op)
        mapped_union: set = set()
        shared_preserved = []
        violations = []
        for facts in input_facts:
            mapped_union |= set(facts["mapped"])
        for axis in rule.shared_axes:
            shared_preserved.append(axis)
            for i, facts in enumerate(input_facts):
                mapped_axes = facts.get("mapped_axes", set())
                # 违约 1：规则要求共享的逻辑轴，被输入当成逐样本映射轴承载。
                if axis in mapped_axes:
                    violations.append(
                        f"输入 #{i} 的逻辑轴 {axis!r} 被当成映射轴，"
                        f"但规则 {rule.name} 要求它保持共享"
                    )
                # 违约 2：规则要求该轴在映射层上保持共享，却声明自己覆盖
                # 一个会让该轴随样本变化的映射层，而对应输入确实按该层映射。
                logical = facts.get("logical_axes", set())
                for layer in rule.covers:
                    if (
                        layer in facts.get("mapped", set())
                        and facts.get("is_ragged")
                        and layer in request_layers
                        and axis in logical
                    ):
                        violations.append(
                            f"规则 {rule.name} 要求轴 {axis!r} 共享，"
                            f"却要把映射层 {layer!r} 的逐样本变化并入该轴"
                        )
        # 规则覆盖范围按请求嵌套层逐层判断
        covered = tuple(layer for layer in request_layers if layer in rule.covers)
        missing = tuple(layer for layer in request_layers if layer not in rule.covers)
        decision = MappingDecision(
            output_mapped=tuple(l for l in request_layers if l in mapped_union),
            output_ragged=output_ragged,
            covered_layers=covered,
            missing_layers=missing,
            rule_name=rule.name,
            shared_preserved=shared_preserved,
            violations=sorted(set(violations)),
        )
        return decision

    def assert_no_violation(self, decision: MappingDecision) -> None:
        if decision.violations:
            raise RuleViolation(
                rule=decision.rule_name,
                reason="；".join(decision.violations),
                details={
                    "covered": decision.covered_layers,
                    "missing": decision.missing_layers,
                },
            )
