"""Nonblocking task orchestration; simulator side effects are injected adapters."""

from enum import Enum, auto
import logging
import math
import numpy as np
from .kinematics import DOWN, multiply, yaw_quaternion

LOG = logging.getLogger(__name__)


class State(Enum):
    MOVE_TO_SCAN = auto()
    DETECT_BOX = auto()
    APPROACH_BOX = auto()
    ACTUATE_GRIPPER = auto()
    MOVE_TO_PALLET_SLOT = auto()
    RELEASE_GRIPPER = auto()
    INCREMENT_PALLET_INDEX = auto()
    REPEAT = auto()
    DONE = auto()
    FAULT = auto()


def pallet_grid(config):
    c = config["pallet"]
    cx, cy = c["center_xy"]
    sx, sy = c["spacing_xy"]
    z = c["dimensions"][2] + config["box"]["dimensions"][2] / 2
    return np.array(
        [
            [
                cx + (col - (c["columns"] - 1) / 2) * sx,
                cy + (row - (c["rows"] - 1) / 2) * sy,
                z,
            ]
            for row in range(c["rows"])
            for col in range(c["columns"])
        ]
    )


class PalletizingStateMachine:
    def __init__(self, config, controller, perception, gripper, feed, verify, load_plan=None):
        self.c, self.robot, self.perception = config, controller, perception
        self.gripper, self.feed, self.verify = gripper, feed, verify
        self.load_plan = load_plan
        self.grid = (
            np.asarray([item["position"] for item in load_plan], float)
            if load_plan is not None
            else pallet_grid(config)
        )
        self.slot = 0
        self.filled = np.zeros(len(self.grid), dtype=bool)
        self.state = State.MOVE_TO_SCAN
        self.elapsed = self.total = 0.0
        self.failure = None
        self.box_pose = None
        self.feed(self.slot)
        self.enter(State.MOVE_TO_SCAN, [self.scan()])

    def scan(self):
        return self.c["robot"]["scan_position"], self.c["robot"]["scan_orientation"]

    def enter(self, state, route=()):
        self.state, self.elapsed = state, 0.0
        self.robot.set_route(route)
        LOG.info("%s | occupied=%d/%d", state.name, self.filled.sum(), len(self.grid))

    def arc(self, start, end, quaternion):
        a0, a1 = math.atan2(start[1], start[0]), math.atan2(end[1], end[0])
        difference = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
        r0, r1 = np.linalg.norm(start), np.linalg.norm(end)
        route = []
        for t in np.linspace(0.0, 1.0, 8)[1:]:
            a, r = a0 + t * difference, (1 - t) * r0 + t * r1
            route.append(
                (
                    [r * np.cos(a), r * np.sin(a), self.c["robot"]["travel_z"]],
                    quaternion,
                )
            )
        return route

    def update(self, dt):
        if self.state in (State.DONE, State.FAULT):
            return
        try:
            self._update(dt)
        except Exception as error:
            self.failure = str(error)
            self.state = State.FAULT
            self.robot.hold()
            LOG.exception("Cell fault at slot %d", self.slot)
            raise

    def _update(self, dt):
        self.elapsed += dt
        self.total += dt
        if self.total > self.c["simulation"]["max_seconds"]:
            raise RuntimeError("Cell cycle timeout")
        self.gripper.check()
        r, process = self.c["robot"], self.c["process"]
        if self.state == State.MOVE_TO_SCAN:
            if self.robot.update(dt):
                self.perception.reset()
                self.enter(State.DETECT_BOX)
        elif self.state == State.DETECT_BOX:
            if self.elapsed > self.c["camera"]["detection_timeout"]:
                raise RuntimeError("Detection timeout: " + self.perception.last_error)
            if self.elapsed < 0.6:
                return
            pose = self.perception.detect()
            if pose is not None:
                self.box_pose = pose
                self.box_dimensions = (
                    np.asarray(pose.dimensions, float)
                    if pose.dimensions is not None
                    else np.asarray(self.c["box"]["dimensions"], float)
                )
                self.pick_q = multiply(yaw_quaternion(pose.yaw), DOWN)
                grasp = np.r_[pose.position[:2], pose.top_z + r["tool_length"] + 0.002]
                self.enter(
                    State.APPROACH_BOX,
                    [
                        (
                            grasp + [0.0, 0.0, process["approach_clearance"]],
                            self.pick_q,
                        ),
                        (grasp, self.pick_q),
                    ],
                )
        elif self.state == State.APPROACH_BOX:
            if self.robot.update(dt):
                self.gripper.close(self.slot)
                self.enter(State.ACTUATE_GRIPPER)
        elif self.state == State.ACTUATE_GRIPPER:
            if self.elapsed >= process["grip_dwell"]:
                start = np.r_[self.box_pose.position[:2], r["travel_z"]]
                placement_yaw = (
                    float(self.load_plan[self.slot]["yaw"])
                    if self.load_plan is not None
                    else 0.0
                )
                place_orientation = multiply(yaw_quaternion(placement_yaw), DOWN)
                target = np.r_[
                    self.grid[self.slot, :2],
                    self.grid[self.slot, 2]
                    + self.box_dimensions[2] / 2
                    + r["tool_length"]
                    + process["release_gap"],
                ]
                route = [(start, self.pick_q)] + self.arc(
                    start[:2], target[:2], place_orientation
                )
                route += [
                    (target + [0.0, 0.0, process["approach_clearance"]], place_orientation),
                    (target, place_orientation),
                ]
                self.enter(State.MOVE_TO_PALLET_SLOT, route)
        elif self.state == State.MOVE_TO_PALLET_SLOT:
            if self.robot.update(dt):
                self.gripper.open()
                self.enter(State.RELEASE_GRIPPER)
        elif self.state == State.RELEASE_GRIPPER:
            if self.elapsed >= process["release_dwell"]:
                self.verify(self.slot, self.grid[self.slot])
                self.enter(State.INCREMENT_PALLET_INDEX)
        elif self.state == State.INCREMENT_PALLET_INDEX:
            self.filled[self.slot] = True
            start = np.r_[self.grid[self.slot, :2], r["travel_z"]]
            route = [(start, DOWN)] + self.arc(start[:2], r["scan_position"][:2], DOWN)
            route += [self.scan()]
            self.slot += 1
            self.enter(State.REPEAT, route)
        elif self.state == State.REPEAT:
            if self.robot.update(dt):
                if self.slot == len(self.grid):
                    for i, target in enumerate(self.grid):
                        self.verify(i, target)
                    self.enter(State.DONE)
                    LOG.info("SUCCESS: %d pallet placements verified", len(self.grid))
                else:
                    self.feed(self.slot)
                    self.enter(State.MOVE_TO_SCAN, [self.scan()])
