import unittest
from xml.etree import ElementTree

from scripts.render_pallet_layout import render


class PalletLayoutVisualizationTests(unittest.TestCase):
    def test_complete_report_renders_valid_svg_with_placement_labels(self):
        report = {
            "state": "DONE",
            "seed": 7,
            "requested": 1,
            "committed": 1,
            "packing_volume_utilization": 0.4,
            "placements": [{
                "index": 0,
                "center": [0.0, 0.0, 0.22],
                "dimensions": [0.2, 0.2, 0.2],
                "layer": 0,
            }],
        }
        config = {
            "pallet": {
                "center_xy": [0.0, 0.0],
                "dimensions": [1.2, 0.8, 0.12],
            }
        }

        svg = render(report, config)
        root = ElementTree.fromstring(svg)
        self.assertEqual(root.tag, "{http://www.w3.org/2000/svg}svg")
        self.assertIn("Verified final layout", svg)
        self.assertIn("B00", svg)
        self.assertIn("40.0%", svg)

    def test_incomplete_run_cannot_be_presented_as_verified_layout(self):
        report = {"state": "FAULT", "requested": 2, "committed": 1, "placements": [{}]}
        config = {"pallet": {"center_xy": [0.0, 0.0], "dimensions": [1.2, 0.8, 0.12]}}
        with self.assertRaisesRegex(ValueError, "Only complete DONE reports"):
            render(report, config)


if __name__ == "__main__":
    unittest.main()
