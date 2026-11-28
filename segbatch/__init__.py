"""segbatch：CPU 分段批化张量变换库。

公开入口覆盖《交换约定》的全部能力面：
分段表示、逻辑张量、插件/规则登记、引擎、交换资料加载与参考解释器。
"""

from .budget import BudgetMeter
from .engine import Engine, Graph, Node, RuleTrace
from .errors import (
    BudgetExhausted,
    GraphError,
    InvalidIndexError,
    RuleViolation,
    SegbatchError,
    UnavailableResult,
    UndefinedTrajectory,
)
from .exchange import build_graph, build_inputs, build_segments, load_exchange
from .plugins import PluginRegistry, PluginSpec
from .rules import RuleRegistry, RuleSpec
from .segments import RaggedSegments
from .tensor import LogicalTensor, make_ragged

__all__ = [
    "BudgetMeter",
    "Engine",
    "Graph",
    "Node",
    "RuleTrace",
    "SegbatchError",
    "GraphError",
    "InvalidIndexError",
    "BudgetExhausted",
    "RuleViolation",
    "UnavailableResult",
    "UndefinedTrajectory",
    "RaggedSegments",
    "LogicalTensor",
    "make_ragged",
    "PluginRegistry",
    "PluginSpec",
    "RuleRegistry",
    "RuleSpec",
    "load_exchange",
    "build_segments",
    "build_inputs",
    "build_graph",
]
