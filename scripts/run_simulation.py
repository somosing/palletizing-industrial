#!/usr/bin/env python3
"""Local PyBullet palletizer. Isaac Sim is available through --backend isaac."""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Retained API for existing configuration tests and the Isaac launcher.
from palletizing.configuration import load_config


def json_default(value):
    """Serialize NumPy scalar/array values in diagnostic and plan reports."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main():
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument(
        "--backend", choices=["pybullet", "isaac"], default="pybullet"
    )
    selected, remaining = selector.parse_known_args()
    if selected.backend == "isaac":
        from scripts.run_isaac import main as isaac_main

        sys.argv = [sys.argv[0]] + remaining
        return isaac_main()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/pybullet.yaml")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--exit-on-complete", action="store_true")
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/latest")
    parser.add_argument("--renderer", choices=["tiny", "opengl", "egl"], default="tiny")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="GUI playback speed, e.g. 4 for 4x; does not change physics timestep",
    )
    parser.add_argument(
        "--fast", action="store_true", help="Run GUI without wall-clock pacing"
    )
    parser.add_argument(
        "--perception", choices=["geometric", "learned"], default="geometric"
    )
    parser.add_argument("--weights", type=Path, default=ROOT / "models/package_seg.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--confidence", type=float, default=0.5)
    args = parser.parse_args(remaining)
    if not np.isfinite(args.speed) or args.speed <= 0 or not 0 < args.confidence < 1:
        parser.error("speed must be positive and confidence must lie between 0 and 1")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = None
    machine = None
    scene = None
    elapsed = 0.0
    code = 1
    observations = []
    failure = None
    perception = None
    gripper = None
    try:
        c = load_config(args.config)
        if args.seed is not None:
            c["simulation"]["seed"] = args.seed
        if not np.isclose(c["robot"]["tool_length"], 0.18):
            raise ValueError(
                "Bundled URDF tool length is 0.18 m; edit the URDF to change it"
            )
        if args.check_config:
            print("Configuration validated (PyBullet)")
            return 0
        from pybullet_utils.bullet_client import BulletClient
        import pybullet
        from palletizing.backends.bullet import (
            Scene,
            RobotController,
            EyeInHandPerception,
            VacuumGripper,
        )
        from palletizing.state_machine import PalletizingStateMachine, State
        from palletizing.stacking import build_random_load
        from PIL import Image

        headless = args.headless or (c["simulation"]["headless"] and not args.gui)
        if (
            not headless
            and sys.platform.startswith("linux")
            and not os.environ.get("DISPLAY")
        ):
            raise RuntimeError(
                "No desktop display found. Use --headless, or run from your Ubuntu desktop terminal."
            )
        p = BulletClient(connection_mode=pybullet.DIRECT if headless else pybullet.GUI)
        args.output.mkdir(parents=True, exist_ok=True)
        from palletizing.rendering import configure_renderer

        renderer_id = configure_renderer(p, args.renderer, headless)
        c["simulation"]["camera_renderer_id"] = renderer_id
        load_plan = None
        if args.perception == "learned":
            load_plan, travel_z = build_random_load(c, c["simulation"]["seed"])
            c["robot"]["travel_z"] = travel_z
            logging.info(
                "Random load: %d cartons, up to 2 layers; transfer clearance %.2f m",
                len(load_plan),
                travel_z,
            )
        else:
            logging.info("Fixed cube load selected; learned perception enables mixed cartons")
        scene = Scene(p, c, ROOT, load_plan)
        controller = RobotController(p, scene.robot, c)
        controller.bootstrap()
        for _ in range(120):
            p.stepSimulation()
        if args.perception == "learned":
            from palletizing.learned_perception import LearnedEyeInHandPerception

            perception = LearnedEyeInHandPerception(
                p, controller, c, args.weights, args.device, args.confidence
            )
        else:
            perception = EyeInHandPerception(p, controller, c)
        gripper = VacuumGripper(scene, controller)
        machine = PalletizingStateMachine(
            c, controller, perception, gripper, scene.feed, scene.verify, load_plan
        )
        dt = 1 / c["simulation"]["physics_hz"]
        tick = 0
        wall = time.monotonic()
        capture_every = c["simulation"]["physics_hz"] // c["camera"]["fps"]
        saved = set()
        while p.isConnected():
            p.stepSimulation()
            if machine.state == State.DETECT_BOX and tick % capture_every == 0:
                perception.camera.capture()
            machine.update(dt)
            if (
                machine.box_pose is not None
                and machine.slot not in saved
                and machine.state == State.APPROACH_BOX
            ):
                Image.fromarray(perception.camera.rgb[:, :, :3]).save(
                    args.output / f"pick_{machine.slot}_rgb.png"
                )
                np.save(
                    args.output / f"pick_{machine.slot}_depth_m.npy",
                    perception.camera.get_depth(),
                )
                if args.perception == "learned":
                    Image.fromarray(perception.camera.overlay()).save(
                        args.output / f"pick_{machine.slot}_predicted_mask.png"
                    )
                pose = machine.box_pose
                # Ground truth is used only for evaluation after perception has finished.
                actual = p.getBasePositionAndOrientation(scene.boxes[machine.slot])[0]
                observations.append(
                    {
                        "slot": machine.slot,
                        "center_error_m": float(np.linalg.norm(pose.position - actual)),
                        "estimated_dimensions_m": (
                            pose.dimensions.tolist()
                            if pose.dimensions is not None
                            else c["box"]["dimensions"]
                        ),
                    }
                )
                (args.output / f"pick_{machine.slot}_pose.json").write_text(
                    json.dumps(
                        {
                            "position": pose.position.tolist(),
                            "yaw": pose.yaw,
                            "points": pose.point_count,
                        },
                        indent=2,
                    )
                )
                saved.add(machine.slot)
            tick += 1
            elapsed = tick * dt
            if machine.state == State.DONE:
                view = p.computeViewMatrixFromYawPitchRoll(
                    [0, 0.15, 0.45], 2.8, 45, -35, 0, 2
                )
                projection = p.computeProjectionMatrixFOV(50, 4 / 3, 0.05, 10)
                data = p.getCameraImage(
                    960, 720, view, projection, renderer=renderer_id
                )
                Image.fromarray(
                    np.asarray(data[2], dtype=np.uint8).reshape(720, 960, 4)[:, :, :3]
                ).save(args.output / "pallet_complete.png")
                code = 0
                if not headless and not args.exit_on_complete:
                    logging.info(
                        "Completed. Close the simulator window or press Ctrl+C to exit."
                    )
                    while p.isConnected():
                        time.sleep(0.05)
                break
            if not headless and not args.fast:
                delay = wall + tick * dt / args.speed - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
        if code != 0:
            logging.warning("Simulation window closed before completion")
    except KeyboardInterrupt:
        code = 0 if machine and machine.state.name == "DONE" else 130
    except Exception as error:
        failure = str(error)
        logging.exception("Simulation failed")
    finally:
        if scene is not None:
            report = {
                "backend": "pybullet",
                "physics_device": "cpu",
                "camera_renderer_requested": args.renderer,
                "perception": args.perception,
                "model_device_requested": (
                    args.device if args.perception == "learned" else None
                ),
                "peak_vacuum_force_n": gripper.peak_force if gripper else None,
                "peak_vacuum_torque_nm": gripper.peak_torque if gripper else None,
                "state": machine.state.name if machine else "STARTUP_FAILED",
                "seed": c["simulation"]["seed"],
                "simulated_seconds": elapsed,
                "verified_placements": list(scene.verified.values()),
                "failure": failure
                or (machine.failure if machine else "Startup failed"),
                "exit_code": code,
                "perception_checks": observations,
                "body_ids": scene.boxes,
                "load_plan": [
                    {
                        "index": item["index"],
                        "slot": item["slot"],
                        "layer": item["layer"],
                        "dimensions_m": item["dimensions"].tolist(),
                        "target_center_m": item["position"].tolist(),
                        "support_index": item["support_index"],
                    }
                    for item in load_plan
                ]
                if load_plan is not None
                else [],
                "remaining_constraints": (
                    p.getNumConstraints() if p.isConnected() else None
                ),
            }
            if perception is not None and args.perception == "learned":
                latencies = perception.camera.latencies
                report["model_predictions"] = len(latencies)
                report["model_latency_median_ms"] = (
                    float(np.median(latencies)) if latencies else None
                )
                report["model_latency_p95_ms"] = (
                    float(np.percentile(latencies, 95)) if latencies else None
                )
                report["last_model_error"] = perception.camera.prediction_error
                if perception.camera.rgb is not None:
                    Image.fromarray(perception.camera.overlay()).save(
                        args.output / "last_model_frame.png"
                    )
            (args.output / "report.json").write_text(
                json.dumps(report, indent=2, default=json_default)
            )
        if p is not None and p.isConnected():
            p.disconnect()
    return code


if __name__ == "__main__":
    sys.exit(main())
