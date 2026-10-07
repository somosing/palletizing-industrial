"""Seeded mixed-carton load plans with stable, column-wise two-layer stacks."""

import numpy as np


def build_random_load(config, seed=None):
    """Create one random 4–8 carton plan for the configured pallet grid.

    Upper cartons remain inside their parent footprint and are centered over it.
    This yields conservative support without claiming general bin packing.
    """
    rng = np.random.default_rng(seed)
    pallet = config["pallet"]
    process = config["process"]
    slots = int(pallet["rows"] * pallet["columns"])
    min_count = int(process.get("random_boxes_min", slots))
    max_count = int(process.get("random_boxes_max", 2 * slots))
    if min_count < slots or max_count > 2 * slots or max_count < min_count:
        raise ValueError("Random load count must fit one or two cartons per pallet column")
    count = int(rng.integers(min_count, max_count + 1))
    cx, cy = pallet["center_xy"]
    sx, sy = pallet["spacing_xy"]
    xy = [
        np.array(
            [
                cx + (col - (pallet["columns"] - 1) / 2) * sx,
                cy + (row - (pallet["rows"] - 1) / 2) * sy,
            ],
            dtype=float,
        )
        for row in range(pallet["rows"])
        for col in range(pallet["columns"])
    ]
    half_pallet = np.asarray(pallet["dimensions"][:2], float) / 2
    for point in xy:
        if np.any(np.abs(point - [cx, cy]) + 0.16 > half_pallet):
            raise ValueError("Configured pallet grid leaves insufficient carton edge margin")

    # All bases fit inside the 0.36 m grid cell; profiles vary footprint and height.
    profiles = np.asarray(
        [
            [0.30, 0.30, 0.28],
            [0.30, 0.24, 0.24],
            [0.24, 0.30, 0.24],
            [0.28, 0.22, 0.20],
            [0.22, 0.28, 0.20],
            [0.26, 0.26, 0.28],
        ],
        dtype=float,
    )
    plans = []
    stack_tops = np.full(slots, float(pallet["dimensions"][2]))
    base_dims = []
    for slot, center_xy in enumerate(xy):
        dims = profiles[int(rng.integers(len(profiles)))].copy()
        if np.any(dims[:2] > np.asarray(pallet["spacing_xy"]) - 0.02):
            raise ValueError("Carton profile does not fit the configured pallet grid")
        z = stack_tops[slot] + dims[2] / 2
        plans.append(
            {
                "index": len(plans),
                "slot": slot,
                "layer": 0,
                "dimensions": dims,
                "position": np.r_[center_xy, z],
                "yaw": 0.0,
                "support_index": None,
            }
        )
        base_dims.append(dims.copy())
        stack_tops[slot] += dims[2]

    upper_columns = rng.permutation(slots)
    for slot_value in upper_columns[: count - slots]:
        slot = int(slot_value)
        parent = slot
        lower = base_dims[slot]
        dims_xy = lower[:2] * rng.uniform(0.82, 0.94, 2)
        height = float(rng.choice([0.16, 0.20, 0.24]))
        dims = np.r_[dims_xy, height]
        z = stack_tops[slot] + height / 2
        plans.append(
            {
                "index": len(plans),
                "slot": slot,
                "layer": 1,
                "dimensions": dims,
                "position": np.r_[xy[slot], z],
                "yaw": 0.0,
                "support_index": parent,
            }
        )
        stack_tops[slot] += height

    max_top = float(np.max(stack_tops))
    max_carton_height = float(max(item["dimensions"][2] for item in plans))
    robot = config["robot"]
    travel_z = max(
        float(robot["travel_z"]),
        max_top + max_carton_height + float(robot["tool_length"]) + 0.12,
    )
    if travel_z > float(robot.get("max_reach_z", 1.45)):
        raise ValueError(f"Planned stack needs unreachable travel height {travel_z:.3f} m")
    return plans, travel_z
