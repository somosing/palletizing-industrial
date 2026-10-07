#!/usr/bin/env python3
"""Standalone Isaac Sim 4.2 launcher. All Omni imports follow SimulationApp."""

import argparse
import logging
import os
from pathlib import Path
import signal
import sys
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LOG = logging.getLogger("palletizing")


from palletizing.configuration import load_config


def run(c, args, app):
    import omni.kit.app
    import omni.usd
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, PhysxSchema
    from omni.isaac.core import SimulationContext
    from omni.isaac.core.articulations import ArticulationView
    from omni.isaac.core.objects import DynamicCuboid, FixedCuboid, GroundPlane
    from omni.isaac.core.prims import RigidPrim
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.utils.extensions import enable_extension
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    from omni.isaac.core.utils.stage import add_reference_to_stage, create_new_stage
    from omni.isaac.core.utils.viewports import set_camera_view
    from palletizing.kinematics import (
        RobotController,
        rotation,
        multiply,
        conjugate,
        rotation_error,
        yaw_quaternion,
    )
    from palletizing.perception import EyeInHandPerception
    from palletizing.state_machine import PalletizingStateMachine, State, pallet_grid

    s, r = c["simulation"], c["robot"]
    extensions = ["omni.isaac.sensor", "omni.isaac.universal_robots"]
    if s["streaming"] != "none":
        app.set_setting("/app/window/drawMouse", True)
        app.set_setting("/app/livestream/proto", "ws")
        app.set_setting("/app/livestream/websocket/framerate_limit", s["render_hz"])
        app.set_setting("/ngx/enabled", False)
        extensions.append(
            "omni.services.streamclient.webrtc"
            if s["streaming"] == "webrtc"
            else "omni.kit.livestream.native"
        )
    if s["ros2_bridge"]:
        extensions.append("omni.isaac.ros2_bridge")
    for extension in extensions:
        enable_extension(extension)
    app.update()
    manager = omni.kit.app.get_app().get_extension_manager()
    for extension in extensions:
        if not manager.is_extension_enabled(extension):
            raise RuntimeError(
                f"Extension unavailable: {extension}; use Isaac Sim 4.2.0"
            )

    create_new_stage()
    dt = 1 / s["physics_hz"]
    render_every = s["physics_hz"] // s["render_hz"]
    sim = SimulationContext(
        stage_units_in_meters=1.0,
        physics_dt=dt,
        rendering_dt=1 / s["render_hz"],
        backend="numpy",
    )
    stage = omni.usd.get_context().get_stage()
    physics = sim.get_physics_context()
    physics.set_gravity(-9.81)
    physics.set_solver_type("TGS")
    physics.enable_ccd(True)
    GroundPlane(prim_path="/World/Ground", size=10.0)
    for key, color in [("table", [0.28, 0.32, 0.36]), ("pallet", [0.50, 0.32, 0.14])]:
        d = np.array(c[key]["dimensions"])
        FixedCuboid(
            prim_path="/World/" + key,
            name=key,
            size=1.0,
            scale=d,
            position=np.r_[c[key]["center_xy"], d[2] / 2],
            color=np.array(color),
        )
    UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr(1000.0)
    root = args.asset_root or get_assets_root_path()
    if not root:
        raise RuntimeError("Asset root unavailable; supply --asset-root")
    add_reference_to_stage(
        root.rstrip("/") + "/" + r["asset_relative_path"], r["prim_path"]
    )
    app.update()
    robot = Robot(prim_path=r["prim_path"], name="ur10", position=np.zeros(3))
    matches = [
        p
        for p in Usd.PrimRange(stage.GetPrimAtPath(r["prim_path"]))
        if p.GetName() == r["wrist_name"]
    ]
    if len(matches) != 1 or not matches[0].HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError("Expected a unique physical UR10 wrist link")
    wrist_path = str(matches[0].GetPath())
    tool = UsdGeom.Cylinder.Define(stage, wrist_path + "/VacuumTool")
    tool.CreateRadiusAttr(0.032)
    tool.CreateHeightAttr(r["tool_length"])
    tool.CreateAxisAttr("Z")
    tool.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, r["tool_length"] / 2))
    tool.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.12, 0.15)])
    perception = EyeInHandPerception(wrist_path, c)
    boxes = []
    side = c["box"]["dimensions"][0]
    for i in range(len(pallet_grid(c))):
        box = DynamicCuboid(
            prim_path=f"/World/Box_{i}",
            name=f"box_{i}",
            size=side,
            mass=c["box"]["mass"],
            position=np.array([2.0 + i * (side + 0.1), -1.5, side / 2 + 0.003]),
            color=np.array([0.72, 0.48, 0.18]),
        )
        api = PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath(box.prim_path))
        api.CreateEnableCCDAttr(True)
        api.CreateSolverPositionIterationCountAttr(16)
        api.CreateSolverVelocityIterationCountAttr(4)
        boxes.append(box)
    view = ArticulationView(prim_paths_expr=r["prim_path"], name="ur10_view")
    sim.initialize_physics()
    sim.play()
    robot.initialize()
    view.initialize()
    wrist = RigidPrim(prim_path=wrist_path, name="wrist", reset_xform_properties=False)
    wrist.initialize()
    for box in boxes:
        box.initialize()
    perception.initialize()
    controller = RobotController(robot, view, wrist, c)
    controller.bootstrap()
    set_camera_view(eye=np.array([2.4, -2.4, 2.2]), target=np.array([0.15, 0.15, 0.35]))
    for tick in range(s["physics_hz"]):
        sim.step(render=False)
        if tick % render_every == 0:
            sim.render()
    controller.validate_jacobian()

    class VacuumLatch:
        """Ideal simulated vacuum: gated, breakable fixed joint; no teleporting."""

        def __init__(self):
            self.path = "/World/VacuumLatch"
            self.index = None

        def close(self, index):
            if self.index is not None:
                raise RuntimeError("Vacuum already engaged")
            wp, wq = controller.pose()
            bp, bq = boxes[index].get_world_pose()
            contact = wp + rotation(wq) @ np.array([0.0, 0.0, r["tool_length"]])
            local = rotation(bq).T @ (contact - bp)
            if (
                np.linalg.norm(local[:2]) > min(0.06, side / 4)
                or abs(local[2] - side / 2) > 0.012
                or np.dot(rotation(wq)[:, 2], rotation(bq)[:, 2]) > -0.98
            ):
                raise RuntimeError(f"Vacuum contact gate failed: {local}")
            joint = UsdPhysics.FixedJoint.Define(stage, self.path)
            joint.CreateBody0Rel().SetTargets([Sdf.Path(wrist_path)])
            joint.CreateBody1Rel().SetTargets([Sdf.Path(boxes[index].prim_path)])
            joint.CreateLocalPos0Attr(
                Gf.Vec3f(*map(float, rotation(wq).T @ (contact - wp)))
            )
            joint.CreateLocalPos1Attr(Gf.Vec3f(*map(float, local)))
            joint.CreateLocalRot0Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0)))
            q = multiply(conjugate(bq), wq)
            joint.CreateLocalRot1Attr(
                Gf.Quatf(float(q[0]), Gf.Vec3f(*map(float, q[1:])))
            )
            joint.CreateExcludeFromArticulationAttr(True)
            joint.CreateCollisionEnabledAttr(False)
            joint.CreateBreakForceAttr(c["process"]["break_force"])
            joint.CreateBreakTorqueAttr(c["process"]["break_torque"])
            self.relative_p = rotation(wq).T @ (bp - wp)
            self.relative_q = multiply(conjugate(wq), bq)
            self.index = index

        def open(self):
            if self.index is None or not stage.RemovePrim(self.path):
                raise RuntimeError("Vacuum release failed")
            self.index = None

        def check(self):
            if self.index is None:
                return
            wp, wq = controller.pose()
            bp, bq = boxes[self.index].get_world_pose()
            if (
                np.linalg.norm(bp - wp - rotation(wq) @ self.relative_p) > 0.025
                or np.linalg.norm(rotation_error(multiply(wq, self.relative_q), bq))
                > 0.15
            ):
                raise RuntimeError("Vacuum joint slipped or broke")

    rng = np.random.default_rng(s["seed"])

    def feed(index):
        jitter = c["box"]["spawn_jitter_xy"]
        xy = np.array(c["table"]["center_xy"]) + rng.uniform(-jitter, jitter, 2)
        boxes[index].set_world_pose(
            np.r_[xy, c["table"]["dimensions"][2] + side / 2 + 0.002],
            yaw_quaternion(rng.uniform(*c["box"]["spawn_yaw_range"])),
        )
        boxes[index].set_linear_velocity(np.zeros(3))
        boxes[index].set_angular_velocity(np.zeros(3))

    def verify(index, target):
        box = boxes[index]
        p, q = box.get_world_pose()
        R = rotation(q)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        if (
            np.linalg.norm(p - target) > c["process"]["placement_tolerance"]
            or R[2, 2] < 0.99
            or abs(np.sin(2 * yaw)) > 0.10
            or np.linalg.norm(box.get_linear_velocity()) > 0.04
            or np.linalg.norm(box.get_angular_velocity()) > 0.10
        ):
            raise RuntimeError(
                f"Placement {index} failed validation: {p}, target={target}"
            )

    machine = PalletizingStateMachine(
        c, controller, perception, VacuumLatch(), feed, verify
    )
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    tick = 0
    try:
        while app.is_running() and not stopping:
            if sim.is_stopped():
                raise RuntimeError(
                    "Timeline stopped; restart the process to reset the cell"
                )
            if not sim.is_playing():
                sim.render()
                continue
            sim.step(render=False)
            if tick % render_every == 0:
                sim.render()
            machine.update(dt)
            tick += 1
            if machine.state == State.DONE and not s["keep_open"]:
                return 0
        return 0 if machine.state == State.DONE else 130
    finally:
        sim.pause()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/parameters.yaml")
    parser.add_argument("--asset-root")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--streaming", choices=("webrtc", "native", "none"))
    parser.add_argument("--exit-on-complete", action="store_true")
    parser.add_argument("--ros2", action="store_true")
    parser.add_argument("--check-config", action="store_true")
    args, kit_args = parser.parse_known_args()
    if any(not arg.startswith("--/") for arg in kit_args):
        parser.error("Unknown option; Kit overrides must use --/setting=value syntax")
    sys.argv = [sys.argv[0]] + kit_args
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    app = None
    try:
        c = load_config(args.config)
        if args.gui:
            c["simulation"]["headless"] = False
        if args.streaming:
            c["simulation"]["streaming"] = args.streaming
        if args.exit_on_complete:
            c["simulation"]["keep_open"] = False
        if args.ros2:
            c["simulation"]["ros2_bridge"] = True
        if args.check_config:
            LOG.info("Configuration validated")
            return 0
        if not c["simulation"]["headless"] and not os.environ.get("DISPLAY"):
            raise RuntimeError("GUI mode requires a working X display")
        from omni.isaac.kit import SimulationApp

        app = SimulationApp(
            {
                "headless": c["simulation"]["headless"],
                "width": 960,
                "height": 540,
                "renderer": "RayTracedLighting",
                "multi_gpu": False,
                "sync_loads": True,
            }
        )
        return run(c, args, app)
    except KeyboardInterrupt:
        return 130
    except Exception:
        LOG.exception("Simulation failed")
        return 1
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    sys.exit(main())
