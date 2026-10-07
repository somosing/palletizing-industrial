"""Full physical simulation regression; uses CPU rendering and no desktop."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from palletizing.configuration import load_config

ROOT = Path(__file__).resolve().parents[1]


class BulletTests(unittest.TestCase):
    def test_loaded_pallet_clearance_is_validated(self):
        import yaml

        c = load_config(ROOT / "config/pybullet.yaml")
        c["robot"]["travel_z"] = 0.89
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(yaml.safe_dump(c))
            with self.assertRaisesRegex(ValueError, "clearance"):
                load_config(path)

    def test_complete_depth_guided_cycle(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_simulation.py"),
                    "--headless",
                    "--output",
                    directory,
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=180,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((Path(directory) / "report.json").read_text())
            self.assertEqual(report["state"], "DONE")
            self.assertEqual(len(report["verified_placements"]), 4)
            self.assertEqual(len(set(report["body_ids"])), 4)
            self.assertEqual(report["remaining_constraints"], 0)
            self.assertEqual(len(report["perception_checks"]), 4)
            self.assertTrue(
                all(
                    p["error_m"] < 0.025 and p["supported"]
                    for p in report["verified_placements"]
                )
            )
            self.assertTrue(
                all(p["center_error_m"] < 0.005 for p in report["perception_checks"])
            )
            self.assertTrue((Path(directory) / "pallet_complete.png").exists())


if __name__ == "__main__":
    unittest.main()
