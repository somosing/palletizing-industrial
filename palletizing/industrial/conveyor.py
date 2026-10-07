"""Discrete-event conveyor queue and finite-buffer pallet selection.

The model treats cartons upstream of the finite buffer as held by conveyor
back-pressure. The robot removes one carton from the buffer at each pick start;
new cartons can then enter the freed slot. Planning sees the buffer and a
configurable short upstream lookahead, but can pick only a carton in the buffer.
It is recomputed after every completed pick (rolling horizon).

This is a scheduling/packing model. It does not simulate conveyor rigid-body
dynamics, barcode sensing, or robot motion; the industrial cell remains a
separate execution backend.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np

from .packing import PalletFull, SupportPacker, optimize_batch_order
from .online_planner import optimize_known_manifest, plan_online_pick, solve_exact_window


@dataclass(frozen=True)
class ConveyorRun:
    """Result from one finite-buffer scheduling simulation."""

    policy: str
    seed: int
    requested_count: int
    buffer_capacity: int
    arrival_interval_s: float
    arrival_jitter_fraction: float
    pick_service_s: float
    lookahead_cartons: int
    schedule: list[dict]
    placements: list[dict]
    makespan_s: float
    upstream_blocked_arrivals: int
    upstream_wait_seconds: float
    max_buffer_occupancy: int
    stack_height_m: float
    layers: int
    volume_utilization: float
    minimum_support_fraction: float
    error: str | None = None
    manifest_order: list[int] | None = None
    manifest_plan_strategy: str | None = None

    def record(self):
        return {
            "policy": self.policy,
            "seed": self.seed,
            "buffer_capacity": self.buffer_capacity,
            "arrival_interval_s": self.arrival_interval_s,
            "arrival_jitter_fraction": self.arrival_jitter_fraction,
            "pick_service_s": self.pick_service_s,
            "lookahead_cartons": self.lookahead_cartons,
            "requested": self.requested_count,
            "committed": len(self.placements),
            "complete": len(self.placements) == self.requested_count,
            "makespan_s": self.makespan_s,
            "throughput_cartons_per_hour": (
                3600.0 * len(self.placements) / self.makespan_s if self.makespan_s > 0 else 0.0
            ),
            "upstream_blocked_arrivals": self.upstream_blocked_arrivals,
            "upstream_wait_seconds": self.upstream_wait_seconds,
            "max_buffer_occupancy": self.max_buffer_occupancy,
            "actual_stack_height_m": self.stack_height_m,
            "actual_layers": self.layers,
            "packing_utilization": self.volume_utilization,
            "minimum_support_fraction": self.minimum_support_fraction,
            "manifest_order": self.manifest_order,
            "manifest_plan_strategy": self.manifest_plan_strategy,
            "requires_upstream_resequencer": self.policy == "manifest-aware",
            "error": self.error,
            "schedule": self.schedule,
            "placements": self.placements,
        }


def _height(packer: SupportPacker) -> float:
    if not packer.placements:
        return 0.0
    return max(p.center[2] + p.dimensions[2] / 2 for p in packer.placements) - packer.base_z


def _can_place(packer: SupportPacker, item: dict) -> bool:
    try:
        return bool(packer.candidates(item["dimensions"], item["mass"], item["capacity"]))
    except (ValueError, TypeError, KeyError):
        return False


def _rolling_horizon_choice(
    packer: SupportPacker, visible: list[int], horizon: list[int], items: list[dict]
) -> int:
    """Choose a buffered carton using the buffer plus short upstream lookahead.

    The objective prioritizes how many horizon cartons can be packed, then
    stack height, layers, minimum support, and utilization. The first carton
    must be in the physical buffer; upstream and upcoming cartons influence
    sequence scoring but cannot be picked before entering the buffer.
    """
    feasible_first = {i for i in visible if _can_place(packer, items[i])}
    best_key = None
    best_first = None
    for first in sorted(feasible_first):
        trial = deepcopy(packer)
        packed = 0
        first_item = items[first]
        try:
            first_placement = trial.propose(
                first_item["dimensions"], first_item["mass"], first_item["capacity"]
            )
        except PalletFull:
            continue
        trial.commit(first_placement)
        packed += 1
        remaining = [index for index in horizon if index != first]
        while remaining:
            options = []
            for index in remaining:
                item = items[index]
                try:
                    placement = trial.propose(item["dimensions"], item["mass"], item["capacity"])
                except PalletFull:
                    continue
                score = (
                    round(placement.center[2] + placement.dimensions[2] / 2, 7),
                    placement.layer,
                    -round(placement.support_fraction, 7),
                    -round(float(np.prod(placement.dimensions)), 7),
                    index,
                )
                options.append((score, index, placement))
            if not options:
                break
            _, selected, placement = min(options, key=lambda option: option[0])
            trial.commit(placement)
            remaining.remove(selected)
            packed += 1
        if packed == 0:
            continue
        min_support = min(p.support_fraction for p in trial.placements)
        layers = max(p.layer for p in trial.placements) + 1
        key = (
            -packed,
            round(_height(trial), 7),
            layers,
            -round(min_support, 7),
            -round(trial.utilization(), 7),
            first,
        )
        if best_key is None or key < best_key:
            best_key, best_first = key, first
    if best_first is None:
        raise PalletFull("No carton currently in the conveyor buffer has a feasible pallet placement")
    return int(best_first)


def _select(
    policy: str, packer: SupportPacker, visible: list[int], horizon: list[int], items: list[dict]
) -> int:
    feasible = [i for i in visible if _can_place(packer, items[i])]
    if not feasible:
        raise PalletFull("No carton currently in the conveyor buffer has a feasible pallet placement")
    if policy == "fifo":
        return min(feasible)
    if policy == "largest-first":
        return max(feasible, key=lambda i: (
            float(items[i]["dimensions"][0]) * float(items[i]["dimensions"][1]),
            float(np.prod(items[i]["dimensions"])),
            -i,
        ))
    if policy == "rolling-horizon":
        return _rolling_horizon_choice(packer, visible, horizon, items)
    raise ValueError(f"Unknown conveyor policy: {policy}")


def _simulate_manifest_aware(
    template: SupportPacker,
    cartons: list[dict],
    arrival_times: list[float],
    *,
    seed: int,
    buffer_capacity: int,
    arrival_interval_s: float,
    arrival_jitter_fraction: float,
    pick_service_s: float,
    manifest_planner: str,
    manifest_beam_width: int,
    manifest_placement_branches: int,
) -> ConveyorRun:
    """Execute a full-manifest packing plan through a finite pick buffer.

    This mode assumes an upstream scan plus a sorter/accumulation lane that can
    select any arrived carton from staging. Only ``buffer_capacity`` cartons can
    occupy the robot pick buffer. The mode makes this hardware assumption
    explicit; it is not achievable on a single FIFO conveyor without a sorter.
    """
    if manifest_planner == "joint-beam":
        beam_plan = None
        try:
            beam_plan = optimize_known_manifest(
                template, cartons, beam_width=manifest_beam_width,
                placement_branches=manifest_placement_branches,
            )
        except PalletFull:
            # Keep a feasible incumbent even when the bounded beam misses one.
            pass
        incumbent = optimize_batch_order(template, cartons, seed=seed)
        if beam_plan is not None:
            beam_score = (
                round(beam_plan.max_height, 6), beam_plan.layers,
                -round(beam_plan.minimum_support_fraction, 6),
                -round(beam_plan.utilization, 6),
            )
            incumbent_score = (
                round(incumbent.max_height, 6), incumbent.layers,
                -round(incumbent.minimum_support_fraction, 6),
                -round(incumbent.utilization, 6),
            )
            if beam_score < incumbent_score:
                beam_plan.search_evaluations += incumbent.search_evaluations
                plan = beam_plan
            else:
                incumbent.search_evaluations += beam_plan.search_evaluations
                incumbent.strategy = "ils_incumbent_after_joint_beam"
                plan = incumbent
        else:
            incumbent.strategy = "ils_incumbent_after_joint_beam_no_complete_beam"
            plan = incumbent
    else:
        plan = optimize_batch_order(template, cartons, seed=seed)
    planned_order = [int(index) for index in plan.order]
    planned_placements = list(plan.placements)
    packer = deepcopy(template)

    now = 0.0
    arrival_index = 0
    arrived: set[int] = set()
    visible: list[int] = []
    buffer_entry_time: dict[int, float] = {}
    started = 0
    in_flight: dict | None = None
    schedule: list[dict] = []
    blocked_arrivals = 0
    upstream_wait = 0.0
    max_occupancy = 0

    def admit_arrivals(at_time: float) -> None:
        nonlocal arrival_index, blocked_arrivals
        while arrival_index < len(cartons) and arrival_times[arrival_index] <= at_time + 1e-9:
            if len(visible) >= buffer_capacity:
                blocked_arrivals += 1
            arrived.add(arrival_index)
            arrival_index += 1

    def refill_buffer(at_time: float) -> None:
        nonlocal max_occupancy, upstream_wait
        while len(visible) < buffer_capacity and started + len(visible) < len(planned_order):
            index = planned_order[started + len(visible)]
            if index not in arrived:
                break
            visible.append(index)
            buffer_entry_time[index] = at_time
            upstream_wait += max(0.0, at_time - arrival_times[index])
        max_occupancy = max(max_occupancy, len(visible))

    while len(packer.placements) < len(cartons):
        if in_flight is not None and in_flight["complete_time_s"] <= now + 1e-9:
            packer.commit(in_flight["placement"])
            in_flight = None
            if len(packer.placements) == len(cartons):
                break

        admit_arrivals(now)
        refill_buffer(now)
        if in_flight is None and visible:
            selected = visible.pop(0)
            pick_position = started
            if selected != planned_order[pick_position]:
                raise RuntimeError("Manifest order and pick-buffer order diverged")
            item = cartons[selected]
            placement = planned_placements[pick_position]
            start = now
            complete = start + float(pick_service_s)
            schedule.append({
                "manifest_index": int(selected),
                "sku": item["sku"],
                "dimensions_m": [float(x) for x in item["dimensions"]],
                "arrival_time_s": float(arrival_times[selected]),
                "buffer_entry_time_s": float(buffer_entry_time[selected]),
                "pick_start_s": float(start),
                "pick_complete_s": float(complete),
                "wait_from_arrival_s": float(start - arrival_times[selected]),
                "visible_manifest_indices": [int(i) for i in [selected] + visible],
                "lookahead_manifest_indices": [],
                "planned_placement": placement.record(),
            })
            started += 1
            in_flight = {"placement": placement, "complete_time_s": complete}
            refill_buffer(now)

        if in_flight is None:
            if arrival_index >= len(cartons):
                raise RuntimeError("Manifest-aware scheduler has no available planned carton")
            now = max(now, float(arrival_times[arrival_index]))
            continue

        next_arrival = arrival_times[arrival_index] if arrival_index < len(cartons) else math.inf
        now = min(float(in_flight["complete_time_s"]), float(next_arrival))

    placements = [placement.record() for placement in packer.placements]
    layers = max((placement.layer for placement in packer.placements), default=-1) + 1
    supports = [placement.support_fraction for placement in packer.placements]
    return ConveyorRun(
        policy="manifest-aware",
        seed=int(seed),
        requested_count=len(cartons),
        buffer_capacity=buffer_capacity,
        arrival_interval_s=float(arrival_interval_s),
        arrival_jitter_fraction=float(arrival_jitter_fraction),
        pick_service_s=float(pick_service_s),
        lookahead_cartons=0,
        schedule=schedule,
        placements=placements,
        makespan_s=float(now),
        upstream_blocked_arrivals=blocked_arrivals,
        upstream_wait_seconds=float(upstream_wait),
        max_buffer_occupancy=max_occupancy,
        stack_height_m=_height(packer),
        layers=layers,
        volume_utilization=packer.utilization(),
        minimum_support_fraction=min(supports, default=1.0),
        manifest_order=planned_order,
        manifest_plan_strategy=plan.strategy,
    )


def simulate_conveyor_batch(
    template: SupportPacker,
    items: list[dict],
    *,
    seed: int,
    policy: str = "rolling-horizon",
    buffer_capacity: int = 3,
    arrival_interval_s: float = 20.0,
    pick_service_s: float = 75.0,
    lookahead_cartons: int = 2,
    arrival_jitter_fraction: float = 0.15,
    beam_width: int = 48,
    search_depth: int = 6,
    placement_branches: int = 3,
    exact_node_limit: int = 100_000,
    manifest_planner: str = "ils",
    manifest_beam_width: int = 24,
    manifest_placement_branches: int = 2,
) -> ConveyorRun:
    """Run one deterministic rolling-horizon conveyor scheduling trial.

    ``items`` are in arrival order. The first carton is available at time zero;
    later intervals vary by the seeded jitter fraction. When the buffer is full,
    arrivals wait upstream without being dropped. The robot is single-server and takes
    ``pick_service_s`` per carton. The planner can inspect ``lookahead_cartons``
    upstream arrivals in addition to cartons already admitted to the buffer,
    but it may select only a carton in the physical buffer. ``seed`` is
    recorded for traceability; item generation is handled by the caller.
    """
    if not items:
        raise ValueError("A conveyor batch must contain at least one carton")
    if policy not in {"fifo", "largest-first", "rolling-horizon", "beam-search", "exact-window", "manifest-aware"}:
        raise ValueError(f"Unknown conveyor policy: {policy}")
    if isinstance(buffer_capacity, bool) or not isinstance(buffer_capacity, int) or not 1 <= buffer_capacity <= 6:
        raise ValueError("buffer_capacity must be an integer in 1..6")
    if isinstance(lookahead_cartons, bool) or not isinstance(lookahead_cartons, int) or not 0 <= lookahead_cartons <= 8:
        raise ValueError("lookahead_cartons must be an integer in 0..8")
    if isinstance(beam_width, bool) or not isinstance(beam_width, int) or not 1 <= beam_width <= 512:
        raise ValueError("beam_width must be an integer in 1..512")
    if isinstance(search_depth, bool) or not isinstance(search_depth, int) or not 1 <= search_depth <= 8:
        raise ValueError("search_depth must be an integer in 1..8")
    if isinstance(placement_branches, bool) or not isinstance(placement_branches, int) or not 1 <= placement_branches <= 16:
        raise ValueError("placement_branches must be an integer in 1..16")
    if isinstance(exact_node_limit, bool) or not isinstance(exact_node_limit, int) or not 1 <= exact_node_limit <= 5_000_000:
        raise ValueError("exact_node_limit must be an integer in 1..5000000")
    if manifest_planner not in {"ils", "joint-beam"}:
        raise ValueError("manifest_planner must be 'ils' or 'joint-beam'")
    if isinstance(manifest_beam_width, bool) or not isinstance(manifest_beam_width, int) or not 1 <= manifest_beam_width <= 512:
        raise ValueError("manifest_beam_width must be an integer in 1..512")
    if isinstance(manifest_placement_branches, bool) or not isinstance(manifest_placement_branches, int) or not 1 <= manifest_placement_branches <= 16:
        raise ValueError("manifest_placement_branches must be an integer in 1..16")
    if not np.isfinite([arrival_interval_s, pick_service_s, arrival_jitter_fraction]).all() or arrival_interval_s <= 0 or pick_service_s <= 0:
        raise ValueError("arrival interval and pick service time must be finite and positive")
    if not 0.0 <= arrival_jitter_fraction < 1.0:
        raise ValueError("arrival_jitter_fraction must be in [0, 1)")

    cartons = [dict(item) for item in items]
    for index, item in enumerate(cartons):
        dims = np.asarray(item.get("dimensions"), dtype=float)
        if dims.shape != (3,) or not np.isfinite(dims).all() or np.any(dims <= 0):
            raise ValueError(f"Invalid dimensions for carton {index}")
        if not np.isfinite([item.get("mass", 1.0), item.get("capacity", 12.0)]).all():
            raise ValueError(f"Invalid mass or crush capacity for carton {index}")
        item.setdefault("mass", 1.0)
        item.setdefault("capacity", 12.0)
        item.setdefault("sku", f"carton-{index:03d}")

    packer = deepcopy(template)
    arrival_rng = np.random.default_rng(seed)
    arrival_times = [0.0]
    for _ in range(1, len(cartons)):
        interval = float(arrival_interval_s) * arrival_rng.uniform(
            1.0 - arrival_jitter_fraction, 1.0 + arrival_jitter_fraction
        )
        arrival_times.append(arrival_times[-1] + float(interval))
    if policy == "manifest-aware":
        return _simulate_manifest_aware(
            template, cartons, arrival_times, seed=seed,
            buffer_capacity=buffer_capacity,
            arrival_interval_s=arrival_interval_s,
            arrival_jitter_fraction=arrival_jitter_fraction,
            pick_service_s=pick_service_s,
            manifest_planner=manifest_planner,
            manifest_beam_width=manifest_beam_width,
            manifest_placement_branches=manifest_placement_branches,
        )
    arrival_index = 0
    visible: list[int] = []
    upstream: list[int] = []
    buffer_entry_time: dict[int, float] = {}
    blocked_arrivals = 0
    upstream_wait = 0.0
    max_occupancy = 0
    schedule: list[dict] = []
    now = 0.0
    in_flight = None
    stop_reason = None

    def admit_arrivals(at_time: float):
        nonlocal arrival_index, blocked_arrivals, max_occupancy, upstream_wait
        while arrival_index < len(cartons) and arrival_times[arrival_index] <= at_time + 1e-9:
            index = arrival_index
            arrival_index += 1
            if len(visible) < buffer_capacity:
                visible.append(index)
                buffer_entry_time[index] = arrival_times[index]
            else:
                upstream.append(index)
                blocked_arrivals += 1
        max_occupancy = max(max_occupancy, len(visible))

    def refill_buffer(at_time: float):
        nonlocal upstream_wait, max_occupancy
        while upstream and len(visible) < buffer_capacity:
            index = upstream.pop(0)
            visible.append(index)
            buffer_entry_time[index] = at_time
            upstream_wait += max(0.0, at_time - arrival_times[index])
        max_occupancy = max(max_occupancy, len(visible))

    while len(packer.placements) < len(cartons):
        if in_flight is not None and in_flight["complete_time_s"] <= now + 1e-9:
            packer.commit(in_flight["placement"])
            in_flight = None
            if len(packer.placements) == len(cartons):
                break

        admit_arrivals(now)
        if in_flight is None and visible:
            upcoming = list(range(arrival_index, min(len(cartons), arrival_index + lookahead_cartons)))
            future_candidates = list(dict.fromkeys(upstream + upcoming))
            future_count = min(lookahead_cartons, max(0, 8 - len(visible)))
            horizon = visible + future_candidates[:future_count]
            planner_info = {}
            planned_placement = None
            try:
                if policy in {"beam-search", "exact-window"}:
                    if policy == "exact-window":
                        online_plan = solve_exact_window(
                            packer, cartons, visible, horizon,
                            search_depth=search_depth,
                            node_limit=exact_node_limit,
                        )
                    else:
                        online_plan = plan_online_pick(
                            packer, cartons, visible, horizon,
                            beam_width=beam_width,
                            search_depth=search_depth,
                            placement_branches=placement_branches,
                        )
                    selected = online_plan.first_pick
                    planned_placement = online_plan.placements[0] if online_plan.placements else None
                    planner_info = {
                        "online_search_mode": online_plan.search_mode,
                        "online_search_depth": online_plan.completed_depth,
                        "online_search_nodes": online_plan.expanded_nodes,
                        "online_search_sequence": list(online_plan.sequence),
                        "online_search_score": list(online_plan.score[:-1]),
                        "online_optimality_certified": online_plan.optimality_certified,
                    }
                else:
                    selected = _select(policy, packer, visible, horizon, cartons)
            except PalletFull as exc:
                stop_reason = f"{exc}; visible={visible}; upstream={upstream}; committed={len(packer.placements)}"
                break
            decision_visible = sorted(visible)
            decision_horizon = sorted(horizon)
            visible.remove(selected)
            placement = planned_placement or packer.propose(
                cartons[selected]["dimensions"], cartons[selected]["mass"], cartons[selected]["capacity"]
            )
            start = now
            complete = start + float(pick_service_s)
            schedule.append({
                "manifest_index": int(selected),
                "sku": cartons[selected]["sku"],
                "dimensions_m": [float(x) for x in cartons[selected]["dimensions"]],
                "arrival_time_s": float(arrival_times[selected]),
                "buffer_entry_time_s": float(buffer_entry_time[selected]),
                "pick_start_s": float(start),
                "pick_complete_s": float(complete),
                "wait_from_arrival_s": float(start - arrival_times[selected]),
                "visible_manifest_indices": [int(i) for i in decision_visible],
                "lookahead_manifest_indices": [int(i) for i in decision_horizon if i not in decision_visible],
                "planner": policy,
                **planner_info,
            })
            in_flight = {"index": selected, "placement": placement, "complete_time_s": complete}
            refill_buffer(now)

        if in_flight is None:
            if arrival_index >= len(cartons):
                if upstream:
                    refill_buffer(now)
                    continue
                stop_reason = (
                    "No pending carton can be selected "
                    f"(committed={len(packer.placements)}, requested={len(cartons)}, "
                    f"started={len(schedule)}, arrived={arrival_index}, "
                    f"buffer={visible}, upstream={upstream}, in_flight={in_flight})"
                )
                break
            now = max(now, arrival_times[arrival_index])
            continue

        next_arrival = arrival_times[arrival_index] if arrival_index < len(cartons) else math.inf
        now = min(float(in_flight["complete_time_s"]), float(next_arrival))

    placements = [placement.record() for placement in packer.placements]
    layers = max((placement.layer for placement in packer.placements), default=-1) + 1
    supports = [placement.support_fraction for placement in packer.placements]
    return ConveyorRun(
        policy=policy,
        seed=int(seed),
        requested_count=len(cartons),
        buffer_capacity=buffer_capacity,
        arrival_interval_s=float(arrival_interval_s),
        arrival_jitter_fraction=float(arrival_jitter_fraction),
        pick_service_s=float(pick_service_s),
        lookahead_cartons=lookahead_cartons,
        schedule=schedule,
        placements=placements,
        makespan_s=float(now),
        upstream_blocked_arrivals=blocked_arrivals,
        upstream_wait_seconds=float(upstream_wait),
        max_buffer_occupancy=max_occupancy,
        stack_height_m=_height(packer),
        layers=layers,
        volume_utilization=packer.utilization(),
        minimum_support_fraction=min(supports, default=1.0),
        error=stop_reason,
    )
