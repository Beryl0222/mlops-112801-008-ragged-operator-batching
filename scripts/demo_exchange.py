"""端到端演示：把资料里的单样本算子组合成可声明不同批维的前向与反向。

在项目根目录运行：

    python -I scripts/demo_exchange.py

演示内容
========
1. 两种嵌套顺序（replica,sample / sample,replica）各跑一次前向+反向；
2. 向量化路径与逐样本参考解释器逐项数值对照（1e-7）；
3. 打印走了哪些规则、哪些回退、为什么必须逐样本；
4. 打印输出位置到原始样本/参数的溯源；
5. 压力规模下证明短轨迹不会被按最长轨迹无界补齐。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from segbatch import (
    BudgetExhausted,
    Engine,
    PluginRegistry,
    build_graph,
    build_inputs,
    build_segments,
    load_exchange,
)
from segbatch.reference import ReferenceInterpreter
from segbatch.stress import (
    PluginReadAudit,
    build_stress_segments,
    generate_stress,
    padding_audit,
)

ATOL = RTOL = 1e-7


def main() -> None:
    data = load_exchange(ROOT / "fixtures" / "batched_axes.json")

    for request in data["requests"]:
        layers = tuple(request["layouts"])
        print("=" * 68)
        print(f"请求嵌套顺序：{layers}，预算：{request['fallback_budget']}")
        seg = build_segments(data, layers)
        inputs = build_inputs(data, seg)

        # 先用资料原样的单样本插件试跑：预算不足以覆盖 6 个有效组合时，
        # 引擎在执行前显式失败，而不是返回半次结果或用掩码吞错。
        trial_graph = build_graph(data)
        trial = Engine(trial_graph,
                       fallback_budget=request["fallback_budget"])
        try:
            trial.run(layers, seg.grid, inputs, request["outputs"])
            plugin_name = "scaled_square"
        except BudgetExhausted as exc:
            print(f"  单样本插件预算不足 -> {exc}")
            print("  改用作者登记的合格批化规则 scaled_square_batched（0 回退）")
            plugin_name = "scaled_square_batched"

        run_data = load_exchange(ROOT / "fixtures" / "batched_axes.json")
        if plugin_name != "scaled_square":
            for p in run_data["programs"]:
                if p.get("plugin"):
                    p["plugin"] = plugin_name
        graph = build_graph(run_data)
        eng = Engine(graph, fallback_budget=request["fallback_budget"])
        outs = eng.run(layers, seg.grid, inputs, request["outputs"])

        print(f"  有效组合 {len(seg.active_combo_ids)} / {seg.combos}，"
              f"空段组合 {sorted(map(int, seg.empty_combo_ids))}")
        for tr in eng.traces:
            why = ""
            if tr.path == "fallback":
                why = (f"（无批化规则，必须逐样本；"
                       f"回退 {len(tr.fallback_combos)} 组合，"
                       f"跳过空段 {tr.skipped_empty_combos}）")
            elif tr.path == "sharded":
                why = f"（分片：覆盖 {tr.covered}，缺失 {tr.missing}）"
            print(f"  节点 {tr.node:8s} 规则 {tr.rule:24s} "
                  f"路径 {tr.path:8s}{why}")
        print(f"  实际逐样本回退计数：{eng.meter.fallback_calls}")

        # 与逐样本参考解释器对照前向
        ref = ReferenceInterpreter(graph, eng.plugins, inputs, seg)
        refout = ref.run_forward()
        for name in request["outputs"]:
            np.testing.assert_allclose(
                outs[name].data, refout[name][1], atol=ATOL, rtol=RTOL
            )
            print(f"  前向 {name} 与逐样本参考一致（1e-7），"
                  f"不适用位置 {sorted(int(c) for c in outs[name].unavailable)}")

        # 反向对照
        seeds = {}
        for name in request["outputs"]:
            seed = np.ones_like(outs[name].data)
            for cid in outs[name].unavailable:
                seed[seg.decode(int(cid))] = 0.0
            seeds[name] = seed
        grads = eng.backward(seeds)
        refg = ref.run_backward(seeds)
        for name in ("shared", "weight", "posbias", "tokens"):
            np.testing.assert_allclose(
                grads[name]["data"], refg[name], atol=ATOL, rtol=RTOL
            )
        print("  反向 shared/weight/posbias/tokens 梯度与逐样本合并一致（1e-7）")

    # 溯源示例
    print("=" * 68)
    print("输出位置溯源（前 5 个有效行 -> 副本/样本/原始序列位置）：")
    seg0 = build_segments(data, ("replica", "sample"))
    for row in seg0.provenance_rows()[:5]:
        print(" ", row)

    # 压力规模：无界补齐审计
    print("=" * 68)
    spec = data["stress"]
    stress_data = generate_stress(spec)
    stress_seg = build_stress_segments(stress_data)
    audit = padding_audit(stress_seg, spec["feature"])
    print(f"压力规模：{stress_seg.combos} 组合，"
          f"{stress_seg.total_valid_rows} 有效行，最长轨迹 {audit['longest_length']}")
    print(f"  稠密补齐需要 {audit['padded_bytes']:>10,d} 字节"
          f"（上限 {spec['workspace_bytes']:,d}，超限）")
    print(f"  分段实际占用 {audit['ragged_bytes']:>10,d} 字节（在上限内）")

    # 独立插件读取审计：保证插件永远看不到补齐/掩码行
    stress_inputs = build_inputs(stress_data, stress_seg)
    program = [
        {"name": "embed", "op": "gather", "in": ["shared", "indices"],
         "axis": "lookup"},
        {"name": "feature", "op": "add", "in": ["tokens", "embed"]},
        {"name": "pre", "op": "matmul", "in": ["feature", "weight"]},
        {"name": "act", "op": "plugin", "in": ["pre", "posbias"],
         "plugin": "scaled_square_batched"},
        {"name": "means", "op": "reduce", "in": ["act"],
         "axis": -2, "reduce": "mean"},
    ]
    reg = PluginRegistry()
    base = reg.get("scaled_square_batched")
    read_audit = PluginReadAudit(base.batched_fn)
    reg.register(type(base)(
        name="scaled_square_batched", fn=base.fn, vjp=base.vjp,
        batched_fn=read_audit, batched_vjp=base.batched_vjp,
    ))
    stress_eng = Engine(
        _stress_graph(program),
        plugins=reg, fallback_budget=0,
        workspace_limit=spec["workspace_bytes"],
    )
    stress_eng.run(stress_seg.layers, stress_seg.grid,
                   stress_inputs, ["means"])
    assert read_audit.total_rows_seen == stress_seg.total_valid_rows
    print(f"  插件实际读取 {read_audit.total_rows_seen} 行，"
          f"恰好等于有效行数；峰值工作区 "
          f"{stress_eng.meter.peak_live_bytes:,d} 字节")
    print("演示完成。")


def _stress_graph(program):
    from segbatch import Graph
    return Graph.from_program(program)


if __name__ == "__main__":
    main()
