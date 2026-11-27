import json
from pathlib import Path
import unittest

import numpy as np


class ExchangeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / "fixtures" / "batched_axes.json"
        cls.data = json.loads(source.read_text(encoding="utf-8"))

    def test_mask_hides_bad_index_and_a_segment_is_empty(self):
        examples = self.data["examples"]
        self.assertFalse(examples[1]["tokens"])
        changed = examples[2]
        limit = len(self.data["tensors"]["shared"]["values"])
        self.assertGreaterEqual(max(changed["indices"]), limit)
        for good, idx in zip(changed["valid"], changed["indices"], strict=True):
            if good:
                self.assertLess(idx, limit)
        self.assertLess(len(set(examples[0]["indices"])), len(examples[0]["indices"]))

    def test_replica_and_shared_parameters_use_distinct_axes(self):
        tensors = self.data["tensors"]
        self.assertEqual(np.asarray(tensors["weight"]["values"]).shape[0], len(self.data["replicas"]))
        self.assertNotIn("replica", tensors["shared"]["axes"])
        self.assertEqual(self.data["requests"][0]["layouts"], list(reversed(self.data["requests"][1]["layouts"])))

    def test_dense_padding_exceeds_budget_at_realistic_skew(self):
        load = self.data["stress"]
        padded = load["examples"] * load["long_length"] * load["replicas"] * load["feature"] * 8
        ragged = ((load["examples"] - len(load["long_examples"])) * load["short_length"]
                  + len(load["long_examples"]) * load["long_length"]) * load["replicas"] * load["feature"] * 8
        self.assertGreater(padded, load["workspace_bytes"])
        self.assertLess(ragged, load["workspace_bytes"])
