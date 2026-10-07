"""Pure tests for random carton sizing, two-layer support and target heights."""

from pathlib import Path
import json
import unittest

import numpy as np

from palletizing.configuration import load_config
from palletizing.stacking import build_random_load

ROOT = Path(__file__).resolve().parents[1]


class RandomStackPlanTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "config/pybullet.yaml")

    def test_plan_has_seeded_random_count_and_valid_two_layer_columns(self):
        first, travel_z = build_random_load(self.config, 41)
        repeat, repeat_travel_z = build_random_load(self.config, 41)
        self.assertEqual(len(first), len(repeat))
        self.assertEqual(4 <= len(first) <= 8, True)
        self.assertEqual(travel_z, repeat_travel_z)
        json.dumps(first, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value.item())
        base = {item["slot"]: item for item in first if item["layer"] == 0}
        self.assertEqual(len(base), 4)
        for item in first[4:]:
            parent = base[item["slot"]]
            self.assertEqual(item["support_index"], parent["index"])
            self.assertTrue(np.all(item["dimensions"][:2] < parent["dimensions"][:2]))
            expected_z = parent["position"][2] + parent["dimensions"][2] / 2 + item["dimensions"][2] / 2
            self.assertAlmostEqual(item["position"][2], expected_z)
        self.assertLessEqual(travel_z, self.config["robot"]["max_reach_z"])

    def test_different_seeds_vary_load_count_or_carton_geometry(self):
        first, _ = build_random_load(self.config, 2)
        second, _ = build_random_load(self.config, 19)
        self.assertTrue(
            len(first) != len(second)
            or not np.allclose(
                np.asarray([item["dimensions"] for item in first[:4]]),
                np.asarray([item["dimensions"] for item in second[:4]]),
            )
        )


if __name__ == "__main__":
    unittest.main()
