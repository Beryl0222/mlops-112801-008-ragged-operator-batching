"""有限差分校验：独立于解析 VJP 的梯度正确性，以及自定义算子回退预算。"""

import unittest

import numpy as np

from tests import _bootstrap  # noqa: F401
import segbatch as sb
from segbatch.plugins import PluginSpec


def _cube(q, p):
    """自定义可导算子：(q + p)^3，只登记单样本规则，无批化回调。"""
    z = q + p
    return z ** 3


def _cube_vjp(q, p, out, gout):
    z = q + p
    g = 3.0 * z * z * gout
    return g, g.sum(axis=0)


class FiniteDifferenceTest(unittest.TestCase):
    def _build(self, layers=("replica", "sample")):
        rng = np.random.default_rng(7)
        # 2 副本 x 3 样本，含一个空段和一个掩码位
        valid = [[True, True], [], [True, False, True]]
        grid = (2, 3) if layers == ("replica", "sample") else (3, 2)
        if layers == ("replica", "sample"):
            valid_per_combo = valid * 2
        else:
            valid_per_combo = [valid[s] for s in range(3) for _ in range(2)]
        seg = sb.RaggedSegments.from_valid(layers, grid, valid_per_combo)
        rows = []
        idxs = []
        import itertools
        for coords in itertools.product(*(range(g) for g in grid)):
            s = coords[layers.index("sample")]
            v = np.asarray(valid[s], bool)
            tok = rng.normal(size=(len(v), 2))
            rows.append(tok[v])
            idxs.append(np.arange(v.sum()) % 3)
        tokens = sb.make_ragged(
            "tokens", ("sequence", "feature"), "sequence", rows, seg
        )
        flat_idx = np.concatenate(idxs) if idxs else np.empty(0, np.int64)
        indices = sb.LogicalTensor.ragged(
            "indices", ("sequence",), flat_idx, seg
        )
        shared = sb.LogicalTensor.shared(
            "shared", ("lookup", "feature"), rng.normal(size=(3, 2))
        )
        weight = sb.LogicalTensor.dense(
            "weight", ("feature", "filter"), rng.normal(size=(2, 2, 2)),
            mapped=("replica",), grid=seg,
        )
        posbias = sb.LogicalTensor.shared(
            "posbias", ("filter",), rng.normal(size=2)
        )
        reg = sb.PluginRegistry()
        reg.register(PluginSpec("cube", fn=_cube, vjp=_cube_vjp))
        graph = sb.Graph([
            sb.Node("embed", "gather", ("shared", "indices"), axis="lookup"),
            sb.Node("feature", "add", ("tokens", "embed")),
            sb.Node("pre", "matmul", ("feature", "weight")),
            sb.Node("act", "plugin", ("pre", "posbias"), plugin="cube"),
            sb.Node("loss", "reduce", ("act",), axis=-2, reduce="sum"),
        ])
        return seg, {
            "tokens": tokens, "indices": indices, "shared": shared,
            "weight": weight, "posbias": posbias,
        }, graph, reg

    def test_analytic_grad_matches_finite_difference(self):
        seg, inputs, graph, reg = self._build()
        eng = sb.Engine(graph, plugins=reg, fallback_budget=10)
        outs = eng.run(seg.layers, seg.grid, inputs, ["loss"])
        seed = np.ones_like(outs["loss"].data)  # sum 对空段也适用
        grads = eng.backward({"loss": seed})

        eps = 1e-6
        def scalar_loss(tname, perturb):
            saved = inputs[tname].data
            data = saved + perturb
            inputs[tname].data = data
            e = sb.Engine(graph, plugins=reg, fallback_budget=10)
            o = e.run(seg.layers, seg.grid, inputs, ["loss"])
            inputs[tname].data = saved
            return float(o["loss"].data.sum())

        for tname in ["shared", "weight", "posbias"]:
            ana = grads[tname]["data"]
            num = np.zeros_like(ana)
            it = np.nditer(ana, flags=["multi_index"])
            while not it.finished:
                idx = it.multi_index
                pert = np.zeros_like(ana)
                pert[idx] = eps
                fp = scalar_loss(tname, pert)
                fm = scalar_loss(tname, -pert)
                num[idx] = (fp - fm) / (2 * eps)
                it.iternext()
            np.testing.assert_allclose(
                ana, num, atol=1e-5, rtol=1e-5,
                err_msg=f"{tname} 解析梯度与有限差分不符",
            )

    def test_custom_op_without_vector_rule_needs_budget(self):
        seg, inputs, graph, reg = self._build()
        # 4 个有效组合（2 副本 x 2 非空样本），预算给不足必须报错
        eng = sb.Engine(graph, plugins=reg, fallback_budget=3)
        with self.assertRaises(sb.BudgetExhausted):
            eng.run(seg.layers, seg.grid, inputs, ["loss"])
        # 给足预算则成功，且空段不消耗
        eng2 = sb.Engine(graph, plugins=reg, fallback_budget=4)
        eng2.run(seg.layers, seg.grid, inputs, ["loss"])
        self.assertEqual(eng2.meter.fallback_calls, 4)


if __name__ == "__main__":
    unittest.main()
