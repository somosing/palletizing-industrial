"""CPU-only contract checks. These do not validate PhysX, rendering or streaming."""

import copy
from pathlib import Path
import unittest
import numpy as np
from palletizing.perception import BoxPose, EyeInHandPerception, fit_box
from palletizing.kinematics import rotation, rotation_error, yaw_quaternion
from palletizing.state_machine import PalletizingStateMachine, State, pallet_grid
from scripts.run_simulation import load_config

ROOT = Path(__file__).resolve().parents[1]


def config():
    return load_config(ROOT / "config/parameters.yaml")


def cloud(yaw=0.0):
    xy = np.stack(
        np.meshgrid(np.linspace(-0.15, 0.15, 70), np.linspace(-0.15, 0.15, 70)), axis=-1
    ).reshape(-1, 2)
    return np.c_[xy, np.zeros(len(xy))] @ rotation(yaw_quaternion(yaw)).T + [
        0.6,
        -0.35,
        0.5,
    ]


class GeometryTests(unittest.TestCase):
    def test_cube_center_and_yaw_symmetry(self):
        for yaw in np.linspace(-np.pi, np.pi, 41):
            result = fit_box(cloud(yaw), config())
            np.testing.assert_allclose(result.position, [0.6, -0.35, 0.35], atol=1e-6)
            self.assertLess(abs(np.angle(np.exp(4j * (result.yaw - yaw)))), 1e-6)

    def test_rectangular_carton_dimensions_and_yaw(self):
        yaw = 0.7
        xy = np.stack(
            np.meshgrid(np.linspace(-0.15, 0.15, 55), np.linspace(-0.12, 0.12, 45)),
            axis=-1,
        ).reshape(-1, 2)
        points = np.c_[xy, np.zeros(len(xy))] @ rotation(yaw_quaternion(yaw)).T + [
            0.6,
            -0.35,
            0.5,
        ]
        result = fit_box(points, config())
        np.testing.assert_allclose(result.dimensions, [0.3, 0.24, 0.3], atol=1e-6)
        self.assertLess(abs(np.angle(np.exp(2j * (result.yaw - yaw)))), 1e-6)

    def test_pick_zone_fit_ignores_second_carton_near_camera_roi(self):
        target = cloud()
        nearby = target.copy()
        nearby[:, 0] += 0.35
        result = fit_box(np.vstack([target, nearby]), config())
        np.testing.assert_allclose(result.position, [0.6, -0.35, 0.35], atol=1e-6)
        np.testing.assert_allclose(result.dimensions, [0.3, 0.3, 0.3], atol=1e-6)

    def test_reject_incomplete_face(self):
        points = cloud()
        with self.assertRaises(ValueError):
            fit_box(points[points[:, 0] < 0.6], config())

    def test_reject_tilt(self):
        points = cloud()
        points[:, 2] += 0.12 * (points[:, 0] - 0.6)
        with self.assertRaises(ValueError):
            fit_box(points, config())

    def test_quaternion_double_cover(self):
        for yaw in np.linspace(-np.pi, np.pi, 41):
            q = yaw_quaternion(yaw)
            np.testing.assert_allclose(rotation_error(q, -q), np.zeros(3), atol=1e-9)

    def test_stale_frames_not_counted_as_independent(self):
        class Camera:
            number = 1

            def get_current_frame(self):
                return {"rendering_frame": self.number}

            def get_depth(self):
                return np.full((480, 640), 0.45)

            def get_world_points_from_image_coords(self, pixels, depth):
                return cloud(0.2)

        perception = object.__new__(EyeInHandPerception)
        perception.config, perception.camera = config(), Camera()
        perception.last_frame, perception.samples, perception.last_error = -1, [], ""
        self.assertIsNone(perception.detect())
        self.assertIsNone(perception.detect())
        self.assertEqual(len(perception.samples), 1)
        perception.camera.number = 2
        self.assertIsNone(perception.detect())
        perception.camera.number = 3
        self.assertIsNotNone(perception.detect())


class Controller:
    def set_route(self, route):
        self.route = list(route)

    def update(self, dt):
        return True

    def hold(self):
        self.held = True


class Perception:
    last_error = "Synthetic detection unavailable"

    def reset(self):
        pass

    def detect(self):
        return BoxPose(np.array([0.6, -0.35, 0.35]), 0.0, 0.5, 4900)


class Gripper:
    index = None

    def close(self, index):
        if self.index is not None:
            raise RuntimeError("Double grasp")
        self.index = index

    def open(self):
        if self.index is None:
            raise RuntimeError("Release without grasp")
        self.index = None

    def check(self):
        pass


class ProcessTests(unittest.TestCase):
    def test_four_slots_and_inventory_are_not_reused(self):
        fed, verified = [], []
        machine = PalletizingStateMachine(
            config(),
            Controller(),
            Perception(),
            Gripper(),
            fed.append,
            lambda i, p: verified.append(i),
        )
        states = set()
        for _ in range(2000):
            states.add(machine.state)
            machine.update(0.05)
            if machine.state == State.DONE:
                break
        self.assertEqual(machine.state, State.DONE)
        self.assertEqual(fed, [0, 1, 2, 3])
        self.assertEqual(verified, [0, 1, 2, 3, 0, 1, 2, 3])
        self.assertTrue(machine.filled.all())
        self.assertTrue(set(State) - {State.DONE, State.FAULT} <= states)

    def test_failed_release_is_never_counted(self):
        class BrokenGripper(Gripper):
            def open(self):
                raise RuntimeError("Release failed")

        robot = Controller()
        machine = PalletizingStateMachine(
            config(),
            robot,
            Perception(),
            BrokenGripper(),
            lambda i: None,
            lambda i, p: None,
        )
        with self.assertRaisesRegex(RuntimeError, "Release failed"):
            for _ in range(200):
                machine.update(0.05)
        self.assertEqual(machine.state, State.FAULT)
        self.assertFalse(machine.filled.any())
        self.assertTrue(robot.held)

    def test_detection_timeout(self):
        class Missing(Perception):
            def detect(self):
                return None

        machine = PalletizingStateMachine(
            config(),
            Controller(),
            Missing(),
            Gripper(),
            lambda i: None,
            lambda i, p: None,
        )
        with self.assertRaisesRegex(RuntimeError, "Detection timeout"):
            for _ in range(300):
                machine.update(0.05)
        self.assertEqual(machine.state, State.FAULT)

    def test_grid_inside_pallet(self):
        c = config()
        grid = pallet_grid(c)
        extent = (
            np.abs(grid[:, :2] - c["pallet"]["center_xy"])
            + np.array(c["box"]["dimensions"][:2]) / 2
        )
        self.assertTrue(np.all(extent <= np.array(c["pallet"]["dimensions"][:2]) / 2))

    def test_invalid_rate_config_is_rejected(self):
        import tempfile
        import yaml

        c = copy.deepcopy(config())
        c["simulation"]["render_hz"] = 29
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(yaml.safe_dump(c))
            with self.assertRaisesRegex(ValueError, "rate ratios"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
