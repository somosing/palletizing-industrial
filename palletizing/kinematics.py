"""Six-DOF UR10 articulation adapter and damped differential IK."""

import logging
import math
import numpy as np

LOG = logging.getLogger(__name__)
DOWN = np.array([0.0, 1.0, 0.0, 0.0])


def multiply(a, b):
    w, x, y, z = a
    W, X, Y, Z = b
    return np.array(
        [
            w * W - x * X - y * Y - z * Z,
            w * X + x * W + y * Z - z * Y,
            w * Y - x * Z + y * W + z * X,
            w * Z + x * Y - y * X + z * W,
        ]
    )


def conjugate(q):
    return np.asarray(q) * [1.0, -1.0, -1.0, -1.0]


def yaw_quaternion(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def rotation(q):
    w, x, y, z = np.asarray(q) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotation_error(target, current):
    q = multiply(target, conjugate(current))
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    length = np.linalg.norm(q[1:])
    return (
        2 * q[1:] if length < 1e-10 else q[1:] * (2 * math.atan2(length, q[0]) / length)
    )


def skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def limit(v, maximum):
    return v * min(1.0, maximum / max(np.linalg.norm(v), 1e-12))


class RobotController:
    def __init__(self, robot, view, wrist, config):
        from omni.isaac.core.utils.types import ArticulationAction
        from omni.isaac.universal_robots import KinematicsSolver

        self.Action = ArticulationAction
        self.robot, self.view, self.wrist = robot, view, wrist
        self.c = config["robot"]
        if robot.num_dof != 6:
            raise ValueError("This controller configuration requires a six-DOF UR10")
        self.drive = robot.get_articulation_controller()
        self.drive.set_gains(kps=np.full(6, self.c["kp"]), kds=np.full(6, self.c["kd"]))
        robot.set_solver_position_iteration_count(32)
        robot.set_solver_velocity_iteration_count(8)
        bounds = np.asarray(view.get_dof_limits())[0]
        self.low, self.high = bounds[:, 0] + 0.03, bounds[:, 1] - 0.03
        self.body = view.get_body_index(self.c["wrist_name"])
        self.jindex = self.body - 1
        J = np.asarray(view.get_jacobians())
        if self.jindex < 0 or J.shape[1:] != (view.num_bodies - 1, 6, 6):
            raise RuntimeError(f"Unexpected fixed-base Jacobian shape: {J.shape}")
        self.seed_solver = KinematicsSolver(
            robot, self.c["wrist_name"], attach_gripper=False
        )
        self.correct_com = False
        self.com = np.zeros(3)
        self.previous_velocity = np.zeros(6)
        self.route = []
        self.stable = self.elapsed = 0.0

    def pose(self):
        return self.wrist.get_world_pose()

    def bootstrap(self):
        """One startup teleport before inventory enters the cell; never used in a cycle."""
        action, ok = self.seed_solver.compute_inverse_kinematics(
            np.array(self.c["scan_position"]), np.array(self.c["scan_orientation"])
        )
        if not ok:
            raise RuntimeError("Scan Home is unreachable for the configured UR10")
        self.robot.set_joint_positions(
            action.joint_positions, joint_indices=action.joint_indices
        )
        self.robot.set_joint_velocities(np.zeros(6))
        self.drive.apply_action(action)

    def validate_jacobian(self):
        """Check PhysX Jacobian against finite-difference Lula FK at startup."""
        p, q = self.pose()
        if (
            np.linalg.norm(p - self.c["scan_position"]) > 0.015
            or np.linalg.norm(rotation_error(self.c["scan_orientation"], q)) > 0.05
        ):
            raise RuntimeError("Lula/physics scan frames disagree")
        solver = self.seed_solver.get_kinematics_solver()
        indices = np.array(
            [self.robot.get_dof_index(n) for n in solver.get_joint_names()]
        )
        joints = self.robot.get_joint_positions().astype(float)
        name, h = self.c["wrist_name"], 1e-4
        _, R = solver.compute_forward_kinematics(name, joints[indices])
        numeric = np.zeros((6, 6))
        for i in range(6):
            plus, minus = joints.copy(), joints.copy()
            plus[i] += h
            minus[i] -= h
            pp, Rp = solver.compute_forward_kinematics(name, plus[indices])
            pm, Rm = solver.compute_forward_kinematics(name, minus[indices])
            numeric[:3, i] = (pp - pm) / (2 * h)
            omega = ((Rp - Rm) / (2 * h)) @ R.T
            numeric[3:, i] = (
                np.array(
                    [
                        omega[2, 1] - omega[1, 2],
                        omega[0, 2] - omega[2, 0],
                        omega[1, 0] - omega[0, 1],
                    ]
                )
                / 2
            )
        coms, _ = self.view.get_body_coms()
        self.com = np.asarray(coms)[0, self.body]
        raw = np.asarray(self.view.get_jacobians())[0, self.jindex].astype(float)
        corrected = raw.copy()
        corrected[:3] += skew(rotation(q) @ self.com) @ raw[3:]
        errors = [np.linalg.norm(raw - numeric), np.linalg.norm(corrected - numeric)]
        if min(errors) > 0.03:
            raise RuntimeError(f"Joint ordering/Jacobian reference mismatch: {errors}")
        self.correct_com = errors[1] < errors[0]
        LOG.info("Jacobian/FK check passed: error=%.6f", min(errors))

    def hold(self):
        self.previous_velocity[:] = 0.0
        self.drive.apply_action(
            self.Action(
                joint_positions=self.robot.get_joint_positions(),
                joint_velocities=np.zeros(6),
            )
        )

    def set_route(self, waypoints):
        self.route = [(np.array(p, float), np.array(q, float)) for p, q in waypoints]
        self.elapsed = self.stable = 0.0
        self.hold()

    def update(self, dt):
        """One physics-tick update. Return True only after the route has settled."""
        if not self.route:
            return True
        self.elapsed += dt
        if self.elapsed > self.c["waypoint_timeout"]:
            raise RuntimeError(f"Differential IK timeout at {self.route[0][0]}")
        target_p, target_q = self.route[0]
        p, q = self.pose()
        ep, er = target_p - p, rotation_error(target_q, q)
        joints = self.robot.get_joint_positions()
        J = np.asarray(self.view.get_jacobians())[0, self.jindex].astype(float)
        if self.correct_com:
            J[:3] += skew(rotation(q) @ self.com) @ J[3:]
        if not np.isfinite(J).all() or not np.isfinite(joints).all():
            raise RuntimeError("Nonfinite robot state")
        W = np.diag([1.0, 1.0, 1.0, 0.25, 0.25, 0.25])
        A = W @ J
        desired = (
            W
            @ np.r_[
                limit(3.5 * ep, self.c["max_linear_speed"]),
                limit(3.0 * er, self.c["max_angular_speed"]),
            ]
        )
        minimum = np.linalg.svd(A, compute_uv=False)[-1]
        damping = 0.008 + 0.08 * max(0.0, 1.0 - minimum / 0.06) ** 2
        velocity = A.T @ np.linalg.solve(
            A @ A.T + damping * damping * np.eye(6), desired
        )
        vmax = np.array(self.c["max_joint_speed"])
        velocity /= max(1.0, np.max(np.abs(velocity) / vmax))
        delta = self.c["max_joint_acceleration"] * dt
        velocity = np.clip(
            velocity, self.previous_velocity - delta, self.previous_velocity + delta
        )
        command = np.clip(joints + velocity * dt, self.low, self.high)
        self.previous_velocity = (command - joints) / dt
        self.drive.apply_action(
            self.Action(
                joint_positions=command, joint_velocities=self.previous_velocity.copy()
            )
        )
        arrived = (
            np.linalg.norm(ep) < self.c["position_tolerance"]
            and np.linalg.norm(er) < self.c["orientation_tolerance"]
            and np.max(np.abs(self.robot.get_joint_velocities())) < 0.045
        )
        self.stable = self.stable + dt if arrived else 0.0
        if self.stable >= 0.10:
            self.route.pop(0)
            self.elapsed = self.stable = 0.0
            self.hold()
        return not self.route
