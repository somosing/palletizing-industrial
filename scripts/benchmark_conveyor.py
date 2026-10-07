#!/usr/bin/env python3
"""Compare finite-buffer conveyor scheduling policies without robot rendering."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from palletizing.industrial.conveyor import simulate_conveyor_batch
from palletizing.industrial.packing import PalletFull, SupportPacker


def inventory(seed: int, count: int) -> list[dict]:
    if not 1 <= count <= 40:
        raise ValueError("count must be in 1..40")
    rng = np.random.default_rng(seed)
    cartons = []
    for index in range(count):
        dimensions = rng.uniform([.22, .22, .16], [.31, .31, .27])
        cartons.append({
            "sku": f"seed-{seed}-carton-{index:03d}",
            "dimensions": [float(x) for x in dimensions],
            "mass": float(rng.uniform(.6, 1.4)),
            "capacity": 10.0,
        })
    return cartons


def make_packer(config: dict) -> SupportPacker:
    return SupportPacker(
        config["pallet"]["center_xy"], config["packing"]["usable_dimensions"],
        config["pallet"]["dimensions"][2], config["packing"]["max_height"],
        config["packing"]["gap"], config["packing"]["support_margin"],
        config["packing"]["max_layers"], config["packing"]["max_payload"],
        config["packing"]["min_support_fraction"], config["packing"]["max_overhang"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/industrial.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--counts", type=int, nargs="+", default=[8, 12])
    parser.add_argument("--buffer-capacities", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--policies", nargs="+", choices=["fifo", "largest-first", "rolling-horizon", "beam-search", "exact-window", "manifest-aware"],
                        default=["fifo", "largest-first", "rolling-horizon", "beam-search"])
    parser.add_argument("--arrival-interval-s", type=float, default=20.0)
    parser.add_argument("--arrival-jitter-fraction", type=float, default=0.15)
    parser.add_argument("--pick-service-s", type=float, default=75.0)
    parser.add_argument("--lookahead-cartons", type=int, default=2,
                        help="Maximum upcoming cartons visible to rolling-horizon selection")
    parser.add_argument("--beam-width", type=int, default=48,
                        help="Number of candidate online plans retained at each beam-search depth")
    parser.add_argument("--search-depth", type=int, default=6,
                        help="Maximum pick-and-placement decisions searched ahead (1..8)")
    parser.add_argument("--placement-branches", type=int, default=3,
                        help="Feasible placement alternatives explored per carton and state")
    parser.add_argument("--exact-node-limit", type=int, default=100000,
                        help="Maximum explored states per exact-window decision")
    parser.add_argument("--manifest-planner", choices=["ils", "joint-beam"], default="ils",
                        help="Full-manifest planner used by the manifest-aware policy")
    parser.add_argument("--manifest-beam-width", type=int, default=24)
    parser.add_argument("--manifest-placement-branches", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory to preserve existing benchmark evidence")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.counts)) != len(args.counts):
        parser.error("seeds and counts must not contain duplicates")
    if len(set(args.buffer_capacities)) != len(args.buffer_capacities):
        parser.error("buffer capacities must not contain duplicates")
    if len(set(args.policies)) != len(args.policies):
        parser.error("policies must not contain duplicates")
    if any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be non-negative")
    if any(capacity < 1 or capacity > 6 for capacity in args.buffer_capacities):
        parser.error("buffer capacities must be in 1..6")
    if not 0 <= args.lookahead_cartons <= 8:
        parser.error("lookahead cartons must be in 0..8")
    if not 1 <= args.beam_width <= 512 or not 1 <= args.search_depth <= 8:
        parser.error("beam width must be in 1..512 and search depth in 1..8")
    if not 1 <= args.placement_branches <= 16:
        parser.error("placement branches must be in 1..16")
    if not 1 <= args.exact_node_limit <= 5_000_000:
        parser.error("exact node limit must be in 1..5000000")
    if not 1 <= args.manifest_beam_width <= 512 or not 1 <= args.manifest_placement_branches <= 16:
        parser.error("manifest beam width must be in 1..512 and placement branches in 1..16")
    if any(not 1 <= count <= 40 for count in args.counts):
        parser.error("counts must be in 1..40")
    if args.arrival_interval_s <= 0 or args.pick_service_s <= 0:
        parser.error("arrival interval and pick service time must be positive")
    if not 0 <= args.arrival_jitter_fraction < 1:
        parser.error("arrival jitter fraction must be in [0, 1)")

    config = yaml.safe_load(args.config.read_text())
    args.output.mkdir(parents=True)
    rows = []
    for count in args.counts:
        for seed in args.seeds:
            cartons = inventory(seed, count)
            for capacity in args.buffer_capacities:
                for policy in args.policies:
                    try:
                        result = simulate_conveyor_batch(
                            make_packer(config), cartons, seed=seed, policy=policy,
                            buffer_capacity=capacity,
                            arrival_interval_s=args.arrival_interval_s,
                            pick_service_s=args.pick_service_s,
                            arrival_jitter_fraction=args.arrival_jitter_fraction,
                            lookahead_cartons=args.lookahead_cartons,
                            beam_width=args.beam_width,
                            search_depth=args.search_depth,
                            placement_branches=args.placement_branches,
                            exact_node_limit=args.exact_node_limit,
                            manifest_planner=args.manifest_planner,
                            manifest_beam_width=args.manifest_beam_width,
                            manifest_placement_branches=args.manifest_placement_branches,
                        ).record()
                        row = {
                            "seed": seed, "requested": count, "buffer_capacity": capacity,
                            "policy": policy, "error": "", **{k: v for k, v in result.items() if k != "schedule" and k != "placements"},
                        }
                    except (PalletFull, ValueError) as exc:
                        row = {
                            "seed": seed, "requested": count, "buffer_capacity": capacity,
                            "policy": policy, "committed": 0, "complete": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        result = None
                    rows.append(row)
                    stem = f"n{count}_seed{seed}_buffer{capacity}_{policy}"
                    if result is not None:
                        (args.output / f"{stem}.json").write_text(json.dumps(result, indent=2))
                    print(json.dumps(row), flush=True)

    deadlocks = sum(not row.get("complete", False) for row in rows)
    (args.output / "summary.json").write_text(json.dumps({
        "runs": rows,
        "completed_runs": len(rows) - deadlocks,
        "deadlocked_runs": deadlocks,
        "trial_count": len(rows),
    }, indent=2))
    columns = sorted({key for row in rows for key in row})
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    # A deadlock is a benchmark outcome, not a benchmark runner failure. It is
    # represented in summary.json/CSV and in the individual JSON result files.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
