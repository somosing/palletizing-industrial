#!/usr/bin/env python3
"""Fine-tune a package instance segmenter; default GPU settings target a 6 GB laptop."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data/packages/dataset.yaml"
    )
    parser.add_argument("--model", default="yolo26n-seg.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/segment")
    parser.add_argument("--name", default="packages_yolo26n")
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "models/package_seg.pt"
    )
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error("Generate the dataset first: " + str(args.data))
    if min(args.epochs, args.batch, args.imgsz, args.threads) <= 0:
        parser.error("epochs/batch/imgsz/threads must be positive")
    import torch
    import ultralytics
    from ultralytics import YOLO
    import yaml

    config = yaml.safe_load(args.data.read_text())
    if config.get("names") != {0: "package"}:
        parser.error("Expected the single package class dataset")
    if args.device != "cpu" and not torch.cuda.is_available():
        parser.error(
            "CUDA is unavailable in this Python environment. Check your PyTorch install, or pass --device cpu."
        )
    torch.set_num_threads(args.threads)
    print(
        "PyTorch:",
        torch.__version__,
        "CUDA:",
        torch.version.cuda,
        "device:",
        args.device,
        flush=True,
    )
    if args.device != "cpu":
        print("GPU:", torch.cuda.get_device_name(int(args.device)), flush=True)
    model = YOLO(args.model, task="segment")
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=False,
        pretrained=not args.model.endswith((".yaml", ".yml")),
        cache=False,
        amp=args.device != "cpu",
        mosaic=0.3,
        close_mosaic=min(5, args.epochs),
        fliplr=0.5,
        flipud=0.0,
        degrees=10.0,
        translate=0.05,
        scale=0.15,
        plots=True,
        patience=10,
    )
    best = Path(model.trainer.best)
    if not best.is_file():
        raise RuntimeError("Training did not produce best.pt")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.checkpoint)
    report = {
        "model": args.model,
        "epochs_requested": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "dataset": str(args.data.resolve()),
        "dataset_yaml_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "run_directory": str(model.trainer.save_dir),
        "checkpoint": str(args.checkpoint.resolve()),
        "validation_metrics": {k: float(v) for k, v in model.trainer.metrics.items()},
    }
    args.checkpoint.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print("Saved package model:", args.checkpoint.resolve())
    print("Next: evaluate on the held-out test split, then run --perception learned.")


if __name__ == "__main__":
    main()
