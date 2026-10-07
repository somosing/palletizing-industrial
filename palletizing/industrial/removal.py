"""Precedence planning for removing cartons from a known pallet support graph.

This module computes a geometrically valid top-down removal order. It does not
claim robot reachability: a robot planner must still validate each pick pose,
tool approach, collision path, and suction seal against the live scene.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite
from typing import Callable, Iterable


class RemovalPlanError(ValueError):
    """The supplied support graph cannot be safely depalletized."""


@dataclass(frozen=True)
class RemovalStep:
    sequence: int
    carton_index: int
    layer: int
    support_index: int | None
    reason: str

    def record(self) -> dict:
        return {
            "sequence": self.sequence,
            "carton_index": self.carton_index,
            "layer": self.layer,
            "support_index": self.support_index,
            "reason": self.reason,
        }


def _field(item, name):
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def plan_removal_order(
    placements: Iterable,
    robot_xy=(0.0, 0.0),
    is_reachable: Callable[[object], bool] | None = None,
) -> list[RemovalStep]:
    """Return a top-down order that never removes a carton supporting another.

    Among currently exposed cartons, prefer a reachable carton that is nearer
    the supplied robot XY location; higher cartons break distance ties. A
    caller-provided reachability predicate can filter candidates at each step.
    The palletizer's single-support graph is enough for precedence but not for
    multi-support load transfer or collision-free robot execution.
    """
    items = list(placements)
    if len(robot_xy) != 2:
        raise ValueError("robot_xy must contain x and y")
    rx, ry = float(robot_xy[0]), float(robot_xy[1])
    if not all(isfinite(v) for v in (rx, ry)):
        raise ValueError("robot_xy must be finite")

    by_index = {}
    for item in items:
        index = _field(item, "index")
        support = _field(item, "support")
        center = _field(item, "center")
        layer = _field(item, "layer")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise RemovalPlanError(f"Invalid carton index: {index!r}")
        if index in by_index:
            raise RemovalPlanError(f"Duplicate carton index: {index}")
        if center is None or len(center) < 2 or layer is None:
            raise RemovalPlanError(f"Carton {index} needs center and layer")
        try:
            xy = (float(center[0]), float(center[1]))
            layer_value = int(layer)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RemovalPlanError(f"Carton {index} has invalid center or layer") from exc
        if not all(isfinite(v) for v in xy) or layer_value < 0 or layer_value != layer:
            raise RemovalPlanError(f"Carton {index} has invalid center or layer")
        if support is not None and (isinstance(support, bool) or not isinstance(support, int)):
            raise RemovalPlanError(f"Carton {index} has invalid support index")
        by_index[index] = item

    for index, item in by_index.items():
        support = _field(item, "support")
        if support is not None and support not in by_index:
            raise RemovalPlanError(f"Carton {index} references missing support {support}")
        if support == index:
            raise RemovalPlanError(f"Carton {index} cannot support itself")

    remaining = set(by_index)
    result = []
    while remaining:
        # A carton is exposed when no remaining carton is resting on it.
        exposed = [
            index for index in remaining
            if not any(_field(by_index[child], "support") == index for child in remaining)
        ]
        if not exposed:
            raise RemovalPlanError("Support graph contains a cycle")

        reachable = [i for i in exposed if is_reachable is None or is_reachable(by_index[i])]
        if not reachable:
            raise RemovalPlanError(
                f"No currently exposed carton is reachable; blocked cartons: {sorted(exposed)}"
            )

        def score(index):
            center = _field(by_index[index], "center")
            return (hypot(float(center[0]) - rx, float(center[1]) - ry),
                    -int(_field(by_index[index], "layer")), index)

        selected = min(reachable, key=score)
        item = by_index[selected]
        result.append(RemovalStep(
            sequence=len(result),
            carton_index=selected,
            layer=int(_field(item, "layer")),
            support_index=_field(item, "support"),
            reason="exposed_and_reachable",
        ))
        remaining.remove(selected)

    return result
