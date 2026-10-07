#!/usr/bin/env python3
"""Create independent rendered scenes with exact visible package masks."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
from PIL import Image, ImageDraw
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from palletizing.configuration import load_config
from palletizing.dataset import instance_polygons
from palletizing.kinematics import DOWN, multiply, yaw_quaternion


def carton_texture(path, rng):
    color = tuple(int(x) for x in rng.integers(105, 230, 3))
    image = Image.new("RGB", (256, 256), color)
    draw = ImageDraw.Draw(image)
    tape = tuple(min(255, x + 25) for x in color)
    draw.rectangle((112, 0, 143, 255), fill=tape)
    if rng.random() < 0.8:
        x, y = int(rng.integers(15, 100)), int(rng.integers(30, 140))
        draw.rectangle((x, y, x + 85, y + 60), fill=(235, 235, 228))
        for i in range(20):
            xx = x + 5 + i * 3
            draw.line(
                (xx, y + 12, xx, y + 45),
                fill=(30, 30, 30),
                width=int(rng.integers(1, 3)),
            )
    image.save(path)


def render_scene(p, base, seed, texture_dir, renderer_id):
    from palletizing.backends.bullet import Scene, RobotController, WristCamera

    rng = np.random.default_rng(seed)
    c = copy.deepcopy(base)
    c["simulation"]["camera_renderer_id"] = renderer_id
    # Half the samples reproduce the cell's exact scan viewpoint and lighting.
    # The other half perturb camera and lighting for modest robustness.
    baseline_view = rng.random() < 0.5
    if not baseline_view:
        scan = np.array(c["robot"]["scan_position"]) + rng.uniform(
            [-0.04, -0.04, -0.035], [0.04, 0.04, 0.07]
        )
        c["robot"]["scan_position"] = scan.tolist()
        c["robot"]["scan_orientation"] = multiply(
            yaw_quaternion(rng.uniform(-0.10, 0.10)), DOWN
        ).tolist()
    p.resetSimulation()
    scene = Scene(p, c, ROOT)
    controller = RobotController(p, scene.robot, c)
    controller.bootstrap()
    if not baseline_view:
        for key, body in scene.surfaces.items():
            shade = rng.uniform(0.2, 0.8, 3).tolist()
            p.changeVisualShape(body, -1, rgbaColor=shade + [1])
        p.changeVisualShape(
            scene.floor, -1, rgbaColor=rng.uniform(0.55, 0.95, 3).tolist() + [1]
        )
    # 10% empty negatives, 60% exact single-cell cubes, 30% varied cartons/scenes.
    mode = rng.random()
    n = 0 if mode < 0.1 else (1 if mode < 0.7 else int(rng.integers(1, 4)))
    if n == 0:
        # Avoid identical fixed-view negative frames leaking across dataset splits.
        shade = rng.uniform(0.2, 0.8, 3).tolist() + [1]
        p.changeVisualShape(scene.surfaces["table"], -1, rgbaColor=shade)
    bodies = []
    records = []
    bounds = []
    table = np.array(c["table"]["center_xy"])
    half = np.array(c["table"]["dimensions"][:2]) / 2
    for index in range(n):
        for attempt in range(100):
            exact_cell_box = mode < 0.7
            pallet_profiles = np.asarray(
                [
                    [0.30, 0.30, 0.28],
                    [0.30, 0.24, 0.24],
                    [0.24, 0.30, 0.24],
                    [0.28, 0.22, 0.20],
                    [0.22, 0.28, 0.20],
                    [0.26, 0.26, 0.28],
                ]
            )
            if exact_cell_box:
                d = np.array([0.3, 0.3, 0.3])
            elif rng.random() < 0.65:
                d = pallet_profiles[int(rng.integers(len(pallet_profiles)))].copy()
            else:
                d = rng.uniform([0.12, 0.12, 0.15], [0.30, 0.30, 0.30])
            yaw = float(rng.uniform(-0.4, 0.4) if exact_cell_box else rng.uniform(-0.65, 0.65))
            footprint = (
                np.abs(
                    np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
                )
                @ d[:2]
                / 2
            )
            margin = half - footprint - 0.015
            xy = table + rng.uniform(-margin, margin)
            if all(
                np.any(np.abs(xy - oldxy) > footprint + oldhalf + 0.015)
                for oldxy, oldhalf in bounds
            ):
                break
        else:
            raise ValueError("Could not place non-overlapping packages")
        body = scene.box(
            d.tolist(),
            [*xy, c["table"]["dimensions"][2] + d[2] / 2 + 0.002],
            [1, 1, 1, 1],
            1,
            yaw,
        )
        # Match the live feeder's solid burnt-orange cube in a majority of the
        # exact-size examples. Retain plain color and texture variety elsewhere.
        appearance = rng.random()
        if exact_cell_box and appearance < 0.7:
            p.changeVisualShape(body, -1, rgbaColor=[0.75, 0.48, 0.18, 1])
        elif appearance < 0.82:
            carton = rng.uniform([0.35, 0.22, 0.10], [0.82, 0.68, 0.42]).tolist()
            p.changeVisualShape(body, -1, rgbaColor=carton + [1])
        else:
            texture = texture_dir / f"carton_{index}.png"
            carton_texture(texture, rng)
            p.changeVisualShape(body, -1, textureUniqueId=p.loadTexture(str(texture)))
        bodies.append(body)
        bounds.append((xy, footprint))
        records.append({"dimensions_m": d.tolist(), "yaw_rad": yaw})
    for _ in range(50):
        p.stepSimulation()
    for body, record in zip(bodies, records):
        record["position_m"] = list(p.getBasePositionAndOrientation(body)[0])
    camera = WristCamera(p, controller, c)
    lighting = {"labels": True}
    if not baseline_view:
        lighting.update(
            shadow=1,
            lightDirection=rng.uniform([-1, -1, 1], [1, 1, 3]).tolist(),
            lightColor=rng.uniform(0.85, 1.0, 3).tolist(),
            lightAmbientCoeff=float(rng.uniform(0.35, 0.65)),
            lightDiffuseCoeff=float(rng.uniform(0.4, 0.75)),
            lightSpecularCoeff=0.05,
        )
    camera.capture(**lighting)
    labels, mask = instance_polygons(camera.segmentation, bodies)
    meta = {
        "scene_seed": seed,
        "objects": records,
        "camera_inverse_view_projection": camera.inverse.tolist(),
        "resolution": c["camera"]["resolution"],
        "clipping_range": c["camera"]["clipping_range"],
        "label_source": "PyBullet object ID buffer; visible pixels only",
    }
    return camera, labels, mask, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/packages")
    parser.add_argument("--train", type=int, default=600)
    parser.add_argument("--val", type=int, default=120)
    parser.add_argument("--test", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--renderer", choices=["tiny", "egl"], default="tiny")
    parser.add_argument("--save-depth", action="store_true")
    args = parser.parse_args()
    if min(args.train, args.val, args.test) <= 0 or args.seed < 0:
        parser.error("Counts must be positive and seed nonnegative")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(
            "Output must be empty; choose a new directory to keep datasets separate"
        )
    import pybullet
    from pybullet_utils.bullet_client import BulletClient
    from palletizing.rendering import configure_renderer

    c = load_config(ROOT / "config/pybullet.yaml")
    p = BulletClient(connection_mode=pybullet.DIRECT)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    hashes = set()
    counts = {}
    previews = []
    try:
        renderer = configure_renderer(p, args.renderer, True)
        with tempfile.TemporaryDirectory() as tmp:
            for split_id, (split, count) in enumerate(
                [("train", args.train), ("val", args.val), ("test", args.test)]
            ):
                for kind in ["images", "labels", "masks", "metadata"]:
                    (args.output / kind / split).mkdir(parents=True, exist_ok=True)
                if args.save_depth:
                    (args.output / "depth" / split).mkdir(parents=True, exist_ok=True)
                objects = 0
                negatives = 0
                for i in range(count):
                    for attempt in range(40):
                        seed = int(
                            np.random.SeedSequence(
                                [args.seed, split_id, i, attempt]
                            ).generate_state(1)[0]
                        )
                        try:
                            camera, labels, mask, meta = render_scene(
                                p, c, seed, Path(tmp), renderer
                            )
                            break
                        except ValueError:
                            if attempt == 39:
                                raise
                    name = f"{split}_{i:05d}"
                    image = Image.fromarray(camera.rgb[:, :, :3])
                    image.save(
                        args.output / "images" / split / (name + ".jpg"), quality=95
                    )
                    digest = hashlib.sha256(camera.rgb.tobytes()).hexdigest()
                    if digest in hashes:
                        raise RuntimeError(
                            "Duplicate rendered scene across dataset; aborting"
                        )
                    hashes.add(digest)
                    (args.output / "labels" / split / (name + ".txt")).write_text(
                        "\n".join(labels) + ("\n" if labels else "")
                    )
                    Image.fromarray(mask).save(
                        args.output / "masks" / split / (name + ".png")
                    )
                    (args.output / "metadata" / split / (name + ".json")).write_text(
                        json.dumps(meta, indent=2)
                    )
                    if args.save_depth:
                        np.savez_compressed(
                            args.output / "depth" / split / (name + ".npz"),
                            depth_m=camera.get_depth(),
                        )
                    records.append(
                        {
                            "split": split,
                            "name": name,
                            "scene_seed": seed,
                            "objects": len(labels),
                            "rgba_sha256": digest,
                        }
                    )
                    objects += len(labels)
                    negatives += not labels
                    if i < 4:
                        preview = image.copy()
                        draw = ImageDraw.Draw(preview)
                        for line in labels:
                            points = (
                                np.array(line.split()[1:], float).reshape(-1, 2)
                                * c["camera"]["resolution"]
                            )
                            draw.line(
                                [tuple(x) for x in points] + [tuple(points[0])],
                                fill=(0, 255, 50),
                                width=3,
                            )
                        preview.thumbnail((320, 240))
                        previews.append(preview)
                    if (i + 1) % 25 == 0 or i + 1 == count:
                        print(f"{split}: {i+1}/{count}, labels={objects}", flush=True)
                counts[split] = {
                    "images": count,
                    "instances": objects,
                    "negative_images": negatives,
                }
        data = {
            "path": str(args.output.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {0: "package"},
        }
        (args.output / "dataset.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
        (args.output / "manifest.json").write_text(
            json.dumps(
                {
                    "seed": args.seed,
                    "renderer": args.renderer,
                    "splits": counts,
                    "scenes": records,
                },
                indent=2,
            )
        )
        montage = Image.new("RGB", (1280, 720), "white")
        for i, img in enumerate(previews):
            montage.paste(img, ((i % 4) * 320, (i // 4) * 240))
        montage.save(args.output / "preview.png")
        print("Dataset:", args.output / "dataset.yaml")
        print("Inspect labels:", args.output / "preview.png")
    finally:
        p.disconnect()


if __name__ == "__main__":
    main()
