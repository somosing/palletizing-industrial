"""Label integrity and fail-closed learned perception tests."""

import unittest
from types import SimpleNamespace
import numpy as np
from palletizing.dataset import instance_polygons
from palletizing.learned_perception import MaskedCamera
from palletizing.rendering import configure_renderer


class LearningTests(unittest.TestCase):
    def test_encoded_link_ids_and_background(self):
        segmentation = np.full((40, 40), -1, dtype=np.int64)
        segmentation[5:30, 8:32] = 7 + (3 << 24)
        labels, instances = instance_polygons(segmentation, [7])
        self.assertEqual(len(labels), 1)
        self.assertEqual(int((instances == 1).sum()), 600)
        coordinates = np.array(labels[0].split()[1:], dtype=float)
        self.assertTrue(np.all((coordinates >= 0) & (coordinates <= 1)))

    def test_hole_is_not_silently_labeled(self):
        segmentation = np.full((40, 40), -1, dtype=np.int64)
        segmentation[2:38, 2:38] = 7
        segmentation[10:30, 10:30] = -1
        with self.assertRaisesRegex(ValueError, "hole"):
            instance_polygons(segmentation, [7])

    def test_empty_predictions_do_not_fall_back_to_depth(self):
        depth = np.ones((20, 20))
        camera = SimpleNamespace(
            number=1, rgb=np.zeros((20, 20, 3), np.uint8), get_depth=lambda: depth
        )
        calls = []

        def predict(**kwargs):
            calls.append(kwargs)
            return [SimpleNamespace(masks=None)]

        wrapped = MaskedCamera(camera, SimpleNamespace(predict=predict), {}, "cpu")
        self.assertTrue(np.isnan(wrapped.get_depth()).all())
        self.assertTrue(np.isnan(wrapped.get_depth()).all())
        self.assertEqual(len(calls), 1)
        camera.number = 2
        wrapped.get_depth()
        self.assertEqual(len(calls), 2)

    def test_gui_renderer_rejects_headless(self):
        p = SimpleNamespace(setRealTimeSimulation=lambda mode: None)
        with self.assertRaisesRegex(ValueError, "requires the GUI"):
            configure_renderer(p, "opengl", True)


if __name__ == "__main__":
    unittest.main()
