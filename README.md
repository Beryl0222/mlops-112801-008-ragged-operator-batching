# 分段批化交换（segbatch）

把只认识单样本轴的 Python 算子，组合成**可声明不同批维**的 CPU 张量前向与反向。
模型副本（`replica`）与样本位置（`sample`）是两个映射身份；请求自由选择两层
映射的嵌套顺序，物理维序不改变身份。ragged 的 sequence 轴不做稠密补齐，
空段不是一行零，掩码位不进插件、索引算子和有效归约。

## 解决的问题

插件作者写的算子只认单样本逻辑张量，但产品要把文本片段、成组候选、多个模型
副本一起算。候选数和片段长度不齐时，从外面硬套同一个批轴会把真正的归约维
移走；补齐出来的假位置前向可能恰好为零，反向却会把误差写进每个候选共享的
权重。本库用**命名逻辑轴 + 分段（ragged）表示 + 显式映射规则**解决：

- 广播、归约、矩阵乘、gather/scatter 全部用逻辑轴解释，负轴按单样本秩索引；
- 规则推断每个输出轴是映射得到还是保留共享，错误地要求共享轴变映射会在执行
  前报 `RuleViolation`，不留下半次参数梯度；
- 自定义插件没有批化规则时，只在显式 `fallback_budget` 内逐样本回退，空段、
  全掩码组合不调用、不计预算；
- 规则只覆盖一层嵌套时按缺失层**分片**：片内批化、片间独立，预算按片预扣；
- 反向把映射到同一共享权重的各项梯度在完整有效域上合并，可对照逐样本解释器
  与有限差分独立验证。

## 模块

| 模块 | 职责 |
| --- | --- |
| `segbatch/segments.py` | ragged 分段：offsets/valid、组合 id 编解码、逐行溯源 |
| `segbatch/tensor.py` | 逻辑张量：命名轴、映射轴事实、稠密/ragged 两种物理形态 |
| `segbatch/ops.py` | 单样本核与向量化核（gather/add/matmul/reduce/scatter） |
| `segbatch/plugins.py` | 插件登记：单样本 `fn/vjp` 与可选批化 `batched_fn/batched_vjp` |
| `segbatch/rules.py` | 向量化规则登记、映射轴推断、共享轴违约检测、部分覆盖 |
| `segbatch/engine.py` | 引擎：规则/分片/回退调度、空段守卫、前向磁带与反向 |
| `segbatch/reference.py` | 逐样本参考解释器（独立 oracle，前向+反向） |
| `segbatch/budget.py` | 逐样本预算与工作区/视图计量，所有内部分配留痕 |
| `segbatch/stress.py` | PCG64 压力资料生成、无界补齐审计、插件读取审计 |
| `segbatch/exchange.py` | 交换资料 JSON -> 分段张量与算子图 |

## 运行

使用 Python 3.11，依赖只有 NumPy：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

隔离模式（`-I`，忽略环境变量与 cwd 注入）下：

```bash
python -I -c "import sys; sys.path.insert(0,'.'); \
  import unittest; unittest.TextTestRunner(verbosity=2).run( \
  unittest.TestLoader().discover('tests'))"
```

端到端演示（两种嵌套顺序的前向/反向对照、规则追踪、压力审计）：

```bash
python -I scripts/demo_exchange.py
```

## 验证口径

- **数值对照**：向量化前向与反向都对照 `reference.py` 的逐样本解释器，
  容差绝对/相对 `1e-7`；解析梯度再用有限差分（`1e-5`）独立复核。
- **无界补齐**：`stress.py` 生成 144 样本 × 7 副本、最长 4096 的悬殊资料。
  稠密补齐约 396 MB（超 19 MB 上限），分段执行峰值工作区约 97 KB；
  `PluginReadAudit` 证明插件实际读到的行数恰好等于有效行数。
- **空段与掩码**：空段 mean/max 标记为不适用（零占位、无 NaN），空 sum 为零；
  掩码位上的越界索引不会被读到，有效段上的越界索引抛 `InvalidIndexError`
  并带副本/样本/位置坐标。
- **可观测**：`engine.traces` 给出每个节点走了哪条规则、vector/sharded/fallback
  路径、覆盖与缺失的映射层、回退组合与被跳过的空段；`meter.snapshot()` 给出
  预算余量、逐次分配形状与峰值工作区；分段的 `provenance_rows()` 把每个输出
  行回溯到具体副本、样本与原始序列位置。

## 资料中的图只是样例

`fixtures/batched_axes.json` 的算子图与批形状仅用于交换。模型组可以登记另一张
图、换一种嵌套层和批形状继续调用，见 `tests/test_alternate_graph.py` 与
`tests/test_sharding.py`。
