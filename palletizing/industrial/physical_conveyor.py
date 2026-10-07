"""Finite, no-drop accumulation buffer used by the GUI cell.

Arrival times are generated before execution, but carton geometry is disclosed
to the planner only when a carton is measured at the infeed and admitted to a
physical buffer slot. A full buffer applies upstream back-pressure.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Admission:
    index: int
    slot: int
    arrival_time_s: float
    admitted_time_s: float


class FiniteConveyorBuffer:
    """FIFO arrivals feeding a finite, selectable staging buffer.

    The accumulation buffer has individually addressable slots, representing
    an indexed conveyor with a diverter/shuttle. The robot may select any item
    already in a slot; upstream items remain queued during back-pressure.
    """

    def __init__(self, arrival_times: list[float], capacity: int):
        if not arrival_times:
            raise ValueError("At least one carton arrival is required")
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        times = [float(value) for value in arrival_times]
        if any(not math.isfinite(value) or value < 0 for value in times):
            raise ValueError("arrival times must be finite and nonnegative")
        if any(a > b for a, b in zip(times, times[1:])):
            raise ValueError("arrival times must be sorted")
        self.arrival_times = times
        self.capacity = capacity
        self._next_arrival = 0
        self._pending: list[int] = []
        self._slots: dict[int, int] = {}
        self._in_transit: set[int] = set()
        self.blocked_arrivals = 0
        self.upstream_wait_seconds = 0.0
        self.max_occupancy = 0
        self.slot_history: list[dict] = []

    @property
    def visible(self) -> list[int]:
        return [self._slots[slot] for slot in sorted(self._slots)
                if self._slots[slot] not in self._in_transit]

    @property
    def occupancy(self) -> int:
        return len(self._slots)

    @property
    def upstream(self) -> list[int]:
        return list(self._pending) + list(range(self._next_arrival, len(self.arrival_times)))

    def slot_for(self, index: int) -> int:
        for slot, carton in self._slots.items():
            if carton == index:
                return slot
        raise KeyError(f"Carton {index} is not in the physical buffer")

    def mark_in_transit(self, index: int) -> None:
        self.slot_for(index)
        self._in_transit.add(index)

    def mark_ready(self, index: int) -> None:
        if index not in self._in_transit:
            raise KeyError(f"Carton {index} is not being transferred into the buffer")
        self._in_transit.remove(index)

    def advance(self, now_s: float) -> list[Admission]:
        """Admit due arrivals to free slots; retain every blocked carton."""
        now = float(now_s)
        if not math.isfinite(now) or now < 0:
            raise ValueError("now_s must be finite and nonnegative")
        admitted = []

        def admit(index: int) -> None:
            slot = next(slot for slot in range(self.capacity) if slot not in self._slots)
            arrival_time = self.arrival_times[index]
            self._slots[slot] = index
            self.upstream_wait_seconds += max(0.0, now - arrival_time)
            admitted.append(Admission(index, slot, arrival_time, now))
            self.slot_history.append({
                "index": index, "slot": slot,
                "arrival_time_s": arrival_time, "admitted_time_s": now,
                "upstream_wait_s": max(0.0, now - arrival_time),
            })

        while (self._next_arrival < len(self.arrival_times)
               and self.arrival_times[self._next_arrival] <= now + 1e-9):
            index = self._next_arrival
            self._next_arrival += 1
            if len(self._slots) < self.capacity and not self._pending:
                admit(index)
            else:
                self._pending.append(index)
                self.blocked_arrivals += 1
        while self._pending and len(self._slots) < self.capacity:
            index = self._pending.pop(0)
            admit(index)
        self.max_occupancy = max(self.max_occupancy, len(self._slots))
        return admitted

    def remove(self, index: int) -> int:
        """Remove a selected carton and return the slot it released."""
        slot = self.slot_for(index)
        del self._slots[slot]
        self._in_transit.discard(index)
        return slot

    def record(self) -> dict:
        return {
            "capacity": self.capacity,
            "current_occupancy": self.occupancy,
            "current_visible_indices": self.visible,
            "current_upstream_indices": self.upstream,
            "max_occupancy": self.max_occupancy,
            "blocked_arrivals": self.blocked_arrivals,
            "upstream_wait_seconds": self.upstream_wait_seconds,
            "admissions": list(self.slot_history),
        }


def make_arrival_times(count: int, interval_s: float, jitter_fraction: float,
                       seed: int) -> list[float]:
    """Create a reproducible stream with positive inter-arrival intervals."""
    if isinstance(count, bool) or int(count) != count or count < 1:
        raise ValueError("count must be a positive integer")
    if not math.isfinite(interval_s) or interval_s <= 0:
        raise ValueError("interval_s must be finite and positive")
    if not math.isfinite(jitter_fraction) or not 0 <= jitter_fraction < 1:
        raise ValueError("jitter_fraction must be in [0, 1)")
    import numpy as np
    rng = np.random.default_rng(seed)
    intervals = interval_s * rng.uniform(1-jitter_fraction, 1+jitter_fraction,
                                         size=max(0, count-1))
    return [0.0, *np.cumsum(intervals).astype(float).tolist()]
