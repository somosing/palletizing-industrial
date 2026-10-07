"""PyBullet adapters; all application quaternions are wxyz, Bullet uses xyzw."""

import logging
import numpy as np
from ..kinematics import rotation_error, rotation, limit
from ..perception import EyeInHandPerception as DepthFitter

LOG = logging.getLogger(__name__)


def xyzw(q):
    return np.asarray(q)[[1, 2, 3, 0]].tolist()


def wxyz(q):
    return np.asarray(q)[[3, 0, 1, 2]]


class RobotController:
    def __init__(self, p, robot, config):
        self.p, self.robot, self.c = p, robot, config["robot"]
        self.joints = [
            i
            for i in range(p.getNumJoints(robot))
            if p.getJointInfo(robot, i)[2] == p.JOINT_REVOLUTE
        ]
        if len(self.joints) != 6:
            raise ValueError("Expected six revolute joints")
        self.wrist = self.joints[-1]
        self.low = np.array([p.getJointInfo(robot, i)[8] + 0.02 for i in self.joints])
        self.high = np.array([p.getJointInfo(robot, i)[9] - 0.02 for i in self.joints])
        self.previous_velocity = np.zeros(6)
        self.route = []
        self.elapsed = self.stable = 0.0

    def state(self):
        states = self.p.getJointStates(self.robot, self.joints)
        return np.array([s[0] for s in states]), np.array([s[1] for s in states])

    def pose(self):
        state = self.p.getLinkState(
            self.robot, self.wrist, computeForwardKinematics=True
        )
        return np.array(state[4]), wxyz(state[5])

    def command(self, q):
        self.p.setJointMotorControlArray(
            self.robot,
            self.joints,
            self.p.POSITION_CONTROL,
            targetPositions=list(q),
            forces=[1500.0] * 6,
            positionGains=[0.45] * 6,
            velocityGains=[1.0] * 6,
        )

    def bootstrap(self):
        # Startup only. During cycles every joint moves through simulated motors.
        for i, q in zip(self.joints, [0, -1.2, 1.7, 0, 1.0, 0]):
            self.p.resetJointState(self.robot, i, q)
        q = self.p.calculateInverseKinematics(
            self.robot,
            self.wrist,
            self.c["scan_position"],
            xyzw(self.c["scan_orientation"]),
            lowerLimits=self.low.tolist(),
            upperLimits=self.high.tolist(),
            jointRanges=(self.high - self.low).tolist(),
            restPoses=[0, -1.2, 1.7, 0, 1.0, 0],
            maxNumIterations=1000,
            residualThreshold=1e-7,
        )
        for i, value in zip(self.joints, q):
            self.p.resetJointState(self.robot, i, value)
        self.command(q)
        pos, ori = self.pose()
        if (
            np.linalg.norm(pos - self.c["scan_position"]) > 0.005
            or np.linalg.norm(rotation_error(self.c["scan_orientation"], ori)) > 0.03
        ):
            raise RuntimeError("Configured scan pose is unreachable")

    def hold(self):
        self.previous_velocity[:] = 0
        self.command(self.state()[0])

    def set_route(self, route):
        self.route = [(np.array(pos, float), np.array(q, float)) for pos, q in route]
        self.elapsed = self.stable = 0.0
        self.hold()

    def update(self, dt):
        if not self.route:
            return True
        self.elapsed += dt
        target, qtarget = self.route[0]
        pos, q = self.pose()
        ep, er = target - pos, rotation_error(qtarget, q)
        joints, speed = self.state()
        if self.elapsed > self.c["waypoint_timeout"]:
            raise RuntimeError(
                f"IK timeout: target={target}, position_error={ep}, angle_error={er}"
            )
        linear, angular = self.p.calculateJacobian(
            self.robot, self.wrist, [0, 0, 0], joints.tolist(), [0.0] * 6, [0.0] * 6
        )
        J = np.vstack([linear, angular])
        W = np.diag([1, 1, 1, 0.3, 0.3, 0.3])
        A = W @ J
        desired = (
            W
            @ np.r_[
                limit(3.5 * ep, self.c["max_linear_speed"]),
                limit(3 * er, self.c["max_angular_speed"]),
            ]
        )
        sigma = np.linalg.svd(A, compute_uv=False)[-1]
        damping = 0.008 + 0.08 * max(0.0, 1 - sigma / 0.06) ** 2
        velocity = A.T @ np.linalg.solve(A @ A.T + damping**2 * np.eye(6), desired)
        velocity /= max(1.0, np.max(np.abs(velocity) / self.c["max_joint_speed"]))
        acceleration = self.c["max_joint_acceleration"] * dt
        velocity = np.clip(
            velocity,
            self.previous_velocity - acceleration,
            self.previous_velocity + acceleration,
        )
        command = np.clip(joints + velocity * dt, self.low, self.high)
        if not np.isfinite(command).all():
            raise RuntimeError("Nonfinite joint command")
        self.previous_velocity = (command - joints) / dt
        self.command(command)
        arrived = (
            np.linalg.norm(ep) < self.c["position_tolerance"]
            and np.linalg.norm(er) < self.c["orientation_tolerance"]
            and np.max(np.abs(speed)) < 0.045
        )
        self.stable = self.stable + dt if arrived else 0.0
        if self.stable >= 0.10:
            self.route.pop(0)
            self.elapsed = self.stable = 0.0
            self.hold()
        return not self.route


class WristCamera:
    def __init__(self, p, controller, config):
        self.p, self.robot, self.c = p, controller, config["camera"]
        self.number = -1
        self.depth = None
        self.inverse = None
        self.rgb = None
        self.segmentation = None
        self.renderer = config["simulation"].get(
            "camera_renderer_id", p.ER_TINY_RENDERER
        )

    def capture(self, labels=False, **lighting):
        p = self.p
        c = self.c
        w, h = c["resolution"]
        origin, q = self.robot.pose()
        R = rotation(q)
        center = origin + R @ np.array(c["local_translation"])
        R = R @ rotation(c["local_orientation_ros"])
        view = p.computeViewMatrix(center, center + R[:, 2], -R[:, 1])
        near, far = c["clipping_range"]
        fov = np.degrees(
            2 * np.arctan(c["horizontal_aperture"] * h / w / (2 * c["focal_length"]))
        )
        projection = p.computeProjectionMatrixFOV(fov, w / h, near, far)
        _, _, rgba, depth, segmentation = p.getCameraImage(
            w,
            h,
            view,
            projection,
            renderer=self.renderer,
            flags=(
                p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
                if labels
                else p.ER_NO_SEGMENTATION_MASK
            ),
            **lighting,
        )
        self.segmentation = np.asarray(segmentation).reshape(h, w) if labels else None
        self.depth = np.asarray(depth).reshape(h, w)
        self.rgb = np.asarray(rgba, dtype=np.uint8).reshape(h, w, 4)
        self.inverse = np.linalg.inv(
            np.array(projection).reshape(4, 4, order="F")
            @ np.array(view).reshape(4, 4, order="F")
        )
        self.number += 1

    def get_current_frame(self):
        return {"rendering_frame": self.number}

    def get_depth(self):
        if self.depth is None:
            return None
        near, far = self.c["clipping_range"]
        return far * near / (far - (far - near) * self.depth)

    def get_world_points_from_image_coords(self, pixels, depth):
        # Invert the same OpenGL projection that produced the nonlinear depth buffer.
        w, h = self.c["resolution"]
        u, v = pixels.T
        ndc = np.c_[
            2 * (u + 0.5) / w - 1,
            1 - 2 * (v + 0.5) / h,
            2 * self.depth[v, u] - 1,
            np.ones(len(u)),
        ]
        world = ndc @ self.inverse.T
        return world[:, :3] / world[:, 3, None]


class EyeInHandPerception(DepthFitter):
    def __init__(self, p, controller, config):
        self.config = config
        self.known_dimensions = True
        self.camera = WristCamera(p, controller, config)
        self.last_frame = -1
        self.samples = []
        self.last_error = "No camera frame yet"

    def initialize(self):
        pass


class Scene:
    def __init__(self, p, config, root, load_plan=None):
        self.p, self.c = p, config
        self.load_plan = load_plan
        self.rng = np.random.default_rng(config["simulation"]["seed"])
        self.boxes = []
        self.box_dimensions = []
        self.verified = {}
        self.spawned = set()
        p.setRealTimeSimulation(0)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1 / config["simulation"]["physics_hz"])
        p.setPhysicsEngineParameter(
            numSolverIterations=100, deterministicOverlappingPairs=1
        )
        self.floor = self.box([5, 5, 0.1], [0, 0, -0.05], [0.85, 0.87, 0.89, 1])
        self.surfaces = {}
        for key, color in [
            ("table", [0.28, 0.32, 0.36, 1]),
            ("pallet", [0.50, 0.32, 0.14, 1]),
        ]:
            d = config[key]["dimensions"]
            self.surfaces[key] = self.box(
                d, [*config[key]["center_xy"], d[2] / 2], color
            )
        self.robot = p.loadURDF(
            str(root / "assets/generic_arm6.urdf"),
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
        )
        p.resetDebugVisualizerCamera(2.8, 45, -35, [0, 0.15, 0.45])

    def box(self, d, pos, color, mass=0, yaw=0):
        p = self.p
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=np.array(d) / 2)
        visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=np.array(d) / 2, rgbaColor=color
        )
        body = p.createMultiBody(
            mass, collision, visual, pos, p.getQuaternionFromEuler([0, 0, yaw])
        )
        p.changeDynamics(
            body,
            -1,
            lateralFriction=0.7,
            restitution=0,
            linearDamping=0.1,
            angularDamping=0.1,
        )
        return body

    def feed(self, index):
        if index in self.spawned:
            raise RuntimeError("Inventory index reused")
        if index != len(self.boxes):
            raise RuntimeError("Inventory sequence mismatch")
        c = self.c
        if self.load_plan is None:
            dimensions = np.asarray(c["box"]["dimensions"], float)
        else:
            dimensions = np.asarray(self.load_plan[index]["dimensions"], float)
        xy = np.array(c["table"]["center_xy"]) + self.rng.uniform(
            -c["box"]["spawn_jitter_xy"], c["box"]["spawn_jitter_xy"], 2
        )
        body = self.box(
            dimensions,
            [*xy, c["table"]["dimensions"][2] + dimensions[2] / 2 + 0.002],
            self.rng.uniform([0.55, 0.30, 0.10], [0.82, 0.62, 0.28]).tolist() + [1],
            c["box"]["mass"],
            self.rng.uniform(*c["box"]["spawn_yaw_range"]),
        )
        self.boxes.append(body)
        self.box_dimensions.append(dimensions.copy())
        self.spawned.add(index)

    def verify(self, index, target):
        p = self.p
        body = self.boxes[index]
        pos, q = p.getBasePositionAndOrientation(body)
        error = np.linalg.norm(np.array(pos) - target)
        tilt = np.linalg.norm(np.array(p.getEulerFromQuaternion(q))[:2])
        v, w = p.getBaseVelocity(body)
        support_index = (
            self.load_plan[index]["support_index"] if self.load_plan is not None else None
        )
        support_body = (
            self.boxes[support_index]
            if support_index is not None
            else self.surfaces["pallet"]
        )
        support = bool(p.getContactPoints(body, support_body))
        if (
            error > self.c["process"]["placement_tolerance"]
            or tilt > 0.05
            or np.linalg.norm(v) > 0.03
            or np.linalg.norm(w) > 0.1
            or not support
        ):
            raise RuntimeError(
                f"Placement {index} failed: error={error:.4f}, tilt={tilt:.4f}, supported={support}"
            )
        self.verified[index] = {
            "position": list(pos),
            "target": list(target),
            "error_m": float(error),
            "supported": support,
        }


class VacuumGripper:
    def __init__(self, scene, robot):
        self.scene, self.robot, self.p = scene, robot, scene.p
        self.constraint = None
        self.body = None
        self.relative = None
        self.peak_force = 0.0
        self.peak_torque = 0.0

    def close(self, index):
        if self.constraint is not None:
            raise RuntimeError("Gripper already occupied")
        p = self.p
        body = self.scene.boxes[index]
        pos, q = self.robot.pose()
        bpos, bq = p.getBasePositionAndOrientation(body)
        tip = pos + rotation(q) @ np.array([0, 0, self.scene.c["robot"]["tool_length"]])
        dimensions = self.scene.box_dimensions[index]
        top = np.array(bpos) + [0, 0, dimensions[2] / 2]
        if np.linalg.norm(tip - top) > 0.02:
            raise RuntimeError("Vacuum tool is not at the detected top face")
        inv = p.invertTransform(pos, xyzw(q))
        rel = p.multiplyTransforms(*inv, bpos, bq)
        self.relative = rel
        self.body = body
        self.constraint = p.createConstraint(
            self.robot.robot,
            self.robot.wrist,
            body,
            -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            rel[0],
            [0, 0, 0],
            parentFrameOrientation=rel[1],
        )
        p.changeConstraint(
            self.constraint, maxForce=self.scene.c["process"]["break_force"]
        )
        for link in range(-1, p.getNumJoints(self.robot.robot)):
            p.setCollisionFilterPair(self.robot.robot, body, link, -1, 0)

    def open(self):
        if self.constraint is None:
            raise RuntimeError("Cannot release empty gripper")
        self.p.removeConstraint(self.constraint)
        for link in range(-1, self.p.getNumJoints(self.robot.robot)):
            self.p.setCollisionFilterPair(self.robot.robot, self.body, link, -1, 1)
        self.constraint = self.body = self.relative = None

    def check(self):
        for contact in self.p.getContactPoints(bodyA=self.robot.robot):
            if contact[3] >= 0 and contact[9] > 2.0:
                raise RuntimeError(
                    f"Robot contacted body {contact[2]} at link {contact[3]}"
                )
        if self.constraint is None:
            return
        force = np.asarray(self.p.getConstraintState(self.constraint))
        if force.shape != (6,) or not np.isfinite(force).all():
            raise RuntimeError(f"Invalid vacuum wrench: {force}")
        self.peak_force = max(self.peak_force, float(np.linalg.norm(force[:3])))
        self.peak_torque = max(self.peak_torque, float(np.linalg.norm(force[3:])))
        limits = self.scene.c["process"]
        if (
            np.linalg.norm(force[:3]) > limits["break_force"]
            or np.linalg.norm(force[3:]) > limits["break_torque"]
        ):
            raise RuntimeError(
                f"Vacuum load limit exceeded: force={np.linalg.norm(force[:3]):.3f} N "
                f"(limit {limits['break_force']}), torque={np.linalg.norm(force[3:]):.3f} Nm "
                f"(limit {limits['break_torque']}), wrench={force.tolist()}"
            )
        pos, q = self.robot.pose()
        expected = self.p.multiplyTransforms(pos, xyzw(q), *self.relative)
        actual = self.p.getBasePositionAndOrientation(self.body)
        if np.linalg.norm(np.array(expected[0]) - actual[0]) > 0.025:
            raise RuntimeError("Vacuum constraint lost the box")
