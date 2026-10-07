#!/usr/bin/env python3
"""Evaluate the trained package model on held-out synthetic scenes."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "models/package_seg.pt")
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data/packages/dataset.yaml"
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/segmentation_test"
    )
    args = parser.parse_args()
    if not args.weights.is_file() or not args.data.is_file():
        parser.error("Weights and dataset YAML must exist")
    from ultralytics import YOLO
    import torch
    import yaml

    if args.device != "cpu" and not torch.cuda.is_available():
        parser.error(
            "CUDA unavailable; use --device cpu or install CUDA-enabled PyTorch"
        )
    torch.set_num_threads(4)
    model = YOLO(str(args.weights), task="segment")
    if list(model.names.values()) != ["package"]:
        parser.error("Expected a fine-tuned package checkpoint")
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = model.val(
        data=str(args.data.resolve()),
        split="test",
        device=args.device,
        imgsz=args.imgsz,
        batch=2,
        workers=0,
        plots=True,
        project=str(args.output.resolve()),
        name="metrics",
        exist_ok=True,
    )
    report = {k: float(v) for k, v in metrics.results_dict.items()}
    report["speed_ms_per_image"] = {k: float(v) for k, v in metrics.speed.items()}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2))
    config = yaml.safe_load(args.data.read_text())
    test = Path(config["path"]) / config["test"]
    for path in sorted(test.glob("*.jpg"))[:12]:
        result = model.predict(
            str(path), device=args.device, conf=0.5, imgsz=args.imgsz, verbose=False
        )[0]
        result.save(filename=str(args.output / (path.stem + "_prediction.jpg")))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
