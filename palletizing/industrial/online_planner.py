"""Bounded online lookahead search for mixed-case palletizing.

The planner searches jointly over (a) which currently buffered carton to pick
first and (b) several feasible placement alternatives for every carton in a
short horizon. Future cartons can influence the score, but the first action is
always selected from the physical buffer. This is a beam-search heuristic; it
does not claim a global-optimality certificate.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np

from .packing import BatchPickPlan, PalletFull, Placement, SupportPacker


@dataclass(frozen=True)
class OnlinePlan:
    first_pick: int
    sequence: tuple[int, ...]
    placements: tuple[Placement, ...]
    score: tuple
    expanded_nodes: int
    completed_depth: int
    optimality_certified: bool = False
    search_mode: str = "beam"


@dataclass
class _Node:
    packer: SupportPacker
    remaining: tuple[int, ...]
    sequence: tuple[int, ...]
    first_pick: int | None


def _height(packer: SupportPacker) -> float:
    if not packer.placements:
        return 0.0
    return max(p.center[2] + p.dimensions[2] / 2 for p in packer.placements) - packer.base_z


def _score(node: _Node) -> tuple:
    packer = node.packer
    supports = [p.support_fraction for p in packer.placements]
    layers = max((p.layer for p in packer.placements), default=-1) + 1
    # Lexicographic policy: place the most horizon cartons; then minimize the
    # current stack envelope/layers; then prefer stronger support and density.
    return (
        -len(node.sequence),
        round(_height(packer), 7),
        layers,
        -round(min(supports, default=1.0), 7),
        -round(packer.utilization(), 7),
        node.sequence,
    )


def _diverse_candidates(packer: SupportPacker, item: dict, limit: int):
    """Keep the best feasible candidates while preserving spatial diversity."""
    candidates = packer.candidates(item["dimensions"], item["mass"], item["capacity"])
    selected = []
    seen = set()
    for p in candidates:
        # Preserve alternatives across pallet regions, support parents and yaw,
        # rather than spending the entire beam on nearly identical coordinates.
        key = (
            p.layer,
            int(math.floor((p.center[0] - packer.center[0]) / 0.12)),
            int(math.floor((p.center[1] - packer.center[1]) / 0.12)),
            p.yaw,
            p.support,
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(p)
        if len(selected) >= limit:
            break
    return selected


def _prune(nodes: list[_Node], width: int) -> list[_Node]:
    """Keep strong states and reserve at least one state per first action."""
    nodes.sort(key=_score)
    chosen = []
    first_seen = set()
    for node in nodes:
        if node.first_pick not in first_seen:
            chosen.append(node)
            first_seen.add(node.first_pick)
            if len(chosen) >= width:
                return chosen[:width]
    chosen_ids = {id(node) for node in chosen}
    for node in nodes:
        if len(chosen) >= width:
            break
        if id(node) not in chosen_ids:
            chosen.append(node)
            chosen_ids.add(id(node))
    return chosen


def reconcile_first_placement(
    packer: SupportPacker,
    dimensions,
    mass: float,
    capacity: float,
    planned: Placement | None,
) -> Placement:
    """Resolve the beam's first placement against the final pick measurement.

    Online planning uses the infeed dimensioner estimate. The wrist camera
    measures the selected carton again before placement. Re-run feasibility
    with that final dimension estimate, then preserve the planned support,
    layer, pose and yaw as closely as possible. If perception changed enough
    that the exact planned candidate is no longer valid, return the closest
    feasible candidate; subsequent picks are replanned from the committed
    pallet state. This keeps the executed first action aligned with the state
    that the online beam evaluated whenever the measurement allows it.
    """
    candidates = packer.candidates(dimensions, mass, capacity)
    if not candidates:
        raise PalletFull(
            "Final carton measurement has no support-, stability-, and load-compliant placement"
        )
    if planned is None:
        return candidates[0]

    def score(candidate: Placement) -> tuple:
        center_error = float(np.linalg.norm(
            np.asarray(candidate.center, dtype=float)
            - np.asarray(planned.center, dtype=float)
        ))
        # Keep the intended quarter-turn when possible. A yaw change can swap
        # the footprint axes and invalidate the packing decisions in the beam.
        yaw_error = 0.5 * abs(float(np.angle(np.exp(2j * (candidate.yaw - planned.yaw)))))
        return (
            candidate.support != planned.support,
            candidate.layer != planned.layer,
            round(center_error, 9),
            round(yaw_error, 9),
            -round(candidate.support_fraction, 9),
        )

    return min(candidates, key=score)


def plan_online_pick(
    packer: SupportPacker,
    items: list[dict],
    visible: list[int],
    horizon: list[int],
    *,
    beam_width: int = 48,
    search_depth: int = 6,
    placement_branches: int = 3,
) -> OnlinePlan:
    """Select a currently buffered carton using bounded joint lookahead.

    `horizon` can include already-buffered cartons and cartons with known
    imminent arrival metadata. Only members of `visible` may be the first pick.
    The state transition uses the same support, CoM, crush, payload, clearance,
    and pallet-bound checks as the live packing model.
    """
    if not visible:
        raise PalletFull("Online planner received an empty pick buffer")
    if beam_width < 1 or not 1 <= search_depth <= 8 or placement_branches < 1:
        raise ValueError("Invalid online planner search bounds")
    visible_set = set(visible)
    horizon = tuple(dict.fromkeys(i for i in horizon if 0 <= i < len(items)))
    if not visible_set.intersection(horizon):
        raise PalletFull("No buffered carton is present in the planning horizon")

    beam = [_Node(deepcopy(packer), horizon, (), None)]
    expanded = 0
    best = None
    max_depth = min(search_depth, len(horizon))
    for _depth in range(max_depth):
        children = []
        for node in beam:
            for index in node.remaining:
                if not node.sequence and index not in visible_set:
                    continue
                item = items[index]
                try:
                    candidates = _diverse_candidates(node.packer, item, placement_branches)
                except (ValueError, TypeError, KeyError):
                    candidates = []
                for placement in candidates:
                    trial = deepcopy(node.packer)
                    trial.commit(placement)
                    sequence = node.sequence + (int(index),)
                    child = _Node(
                        trial,
                        tuple(i for i in node.remaining if i != index),
                        sequence,
                        int(index) if node.first_pick is None else node.first_pick,
                    )
                    children.append(child)
                    expanded += 1
        if not children:
            break
        beam = _prune(children, beam_width)
        candidate = min(beam, key=_score)
        if best is None or _score(candidate) < _score(best):
            best = candidate

    if best is None or best.first_pick is None:
        raise PalletFull("No buffered carton has a feasible online placement")
    return OnlinePlan(
        first_pick=best.first_pick,
        sequence=best.sequence,
        placements=tuple(best.packer.placements[len(packer.placements):]),
        score=_score(best),
        expanded_nodes=expanded,
        completed_depth=len(best.sequence),
        optimality_certified=False,
        search_mode="beam",
    )


def solve_exact_window(
    packer: SupportPacker,
    items: list[dict],
    visible: list[int],
    horizon: list[int],
    *,
    search_depth: int = 4,
    node_limit: int = 100_000,
) -> OnlinePlan:
    """Exhaustively optimize a small, finite online window.

    The result is certified only if the entire finite search tree is exhausted.
    It is exact over the finite placement candidates emitted by
    ``SupportPacker.candidates`` and the supplied horizon, not over arbitrary
    continuous placements or cartons that have not arrived / are not forecast.
    ``node_limit`` makes this safe to run as an anytime planner; on timeout it
    returns the best feasible prefix found without claiming an optimality gap.
    """
    if not visible:
        raise PalletFull("Exact window planner received an empty pick buffer")
    if not 1 <= search_depth <= 8 or node_limit < 1:
        raise ValueError("Invalid exact-window search bounds")
    visible_set = set(visible)
    window = tuple(dict.fromkeys(i for i in horizon if 0 <= i < len(items)))
    if not visible_set.intersection(window):
        raise PalletFull("No buffered carton is present in the exact planning window")
    depth_limit = min(search_depth, len(window))
    root = _Node(deepcopy(packer), window, (), None)
    best = root
    stack = [root]
    expanded = 0
    explored = 0
    truncated = False
    while stack:
        if explored >= node_limit:
            truncated = True
            break
        node = stack.pop()
        explored += 1
        if _score(node) < _score(best):
            best = node
        if len(node.sequence) >= depth_limit:
            continue
        children = []
        for index in node.remaining:
            if not node.sequence and index not in visible_set:
                continue
            item = items[index]
            try:
                candidates = node.packer.candidates(
                    item["dimensions"], item["mass"], item["capacity"]
                )
            except (ValueError, TypeError, KeyError):
                candidates = []
            for placement in candidates:
                trial = deepcopy(node.packer)
                trial.commit(placement)
                child = _Node(
                    trial,
                    tuple(i for i in node.remaining if i != index),
                    node.sequence + (int(index),),
                    int(index) if node.first_pick is None else node.first_pick,
                )
                children.append(child)
                if _score(child) < _score(best):
                    best = child
                expanded += 1
        # Deterministic depth-first search. Reversing preserves the candidate
        # generator's preferred ordering as the first explored branch.
        stack.extend(reversed(children))
    if best.first_pick is None:
        raise PalletFull("No buffered carton has a feasible exact-window placement")
    return OnlinePlan(
        first_pick=best.first_pick,
        sequence=best.sequence,
        placements=tuple(best.packer.placements[len(packer.placements):]),
        score=_score(best),
        expanded_nodes=expanded,
        completed_depth=len(best.sequence),
        optimality_certified=not truncated,
        search_mode="exact-window",
    )


def optimize_known_manifest(
    template: SupportPacker,
    items: list[dict],
    *,
    beam_width: int = 24,
    placement_branches: int = 2,
) -> BatchPickPlan:
    """Jointly search a known complete carton set's order and placements.

    This is an anytime-style bounded beam heuristic. It is stronger than
    evaluating one greedy placement for a few order permutations, but does not
    provide a global-optimum certificate. For small cases compare it with
    ``solve_exact_window`` over the whole set to estimate its gap.
    """
    if not items:
        raise ValueError("Cannot optimize an empty carton manifest")
    if not 1 <= beam_width <= 512 or not 1 <= placement_branches <= 16:
        raise ValueError("Invalid manifest beam-search bounds")
    for index, item in enumerate(items):
        d = np.asarray(item.get("dimensions"), dtype=float)
        mass = float(item.get("mass", 1.0))
        capacity = float(item.get("capacity", 12.0))
        if d.shape != (3,) or not np.isfinite(d).all() or np.any(d <= 0):
            raise ValueError(f"Invalid dimensions for manifest item {index}")
        if not np.isfinite([mass, capacity]).all() or mass <= 0 or capacity < 0:
            raise ValueError(f"Invalid mass/capacity for manifest item {index}")

    root = _Node(deepcopy(template), tuple(range(len(items))), (), None)
    beam = [root]
    best = root
    evaluations = 0
    for _ in range(len(items)):
        children = []
        for node in beam:
            for index in node.remaining:
                try:
                    candidates = _diverse_candidates(
                        node.packer, items[index], placement_branches
                    )
                except (ValueError, TypeError, KeyError):
                    candidates = []
                for placement in candidates:
                    trial = deepcopy(node.packer)
                    trial.commit(placement)
                    child = _Node(
                        trial,
                        tuple(i for i in node.remaining if i != index),
                        node.sequence + (int(index),),
                        int(index),
                    )
                    children.append(child)
                    evaluations += 1
                    if _score(child) < _score(best):
                        best = child
        if not children:
            break
        beam = _prune(children, beam_width)

    if len(best.sequence) != len(items):
        raise PalletFull(
            f"Joint manifest search packed {len(best.sequence)}/{len(items)} cartons "
            f"with beam_width={beam_width}, placement_branches={placement_branches}"
        )
    placements = list(best.packer.placements[len(template.placements):])
    height = _height(best.packer)
    layers = max((p.layer for p in placements), default=-1) + 1
    minimum_support = min((p.support_fraction for p in placements), default=1.0)
    return BatchPickPlan(
        order=list(best.sequence),
        placements=placements,
        strategy=f"joint_beam_w{beam_width}_p{placement_branches}",
        layers=layers,
        max_height=float(height),
        utilization=float(best.packer.utilization()),
        minimum_support_fraction=float(minimum_support),
        search_evaluations=evaluations,
    )
