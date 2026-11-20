"""Occupancy intervals for gate positions, weighing stations and the mobile line.

Two different quantities are kept apart:

``inside``
    Vehicles between gate entry and unloading done.  More than one can be
    inside at a time (one unloading, others at the scale or waiting inside the
    gate), so this is reported as a count series, not as a capacity violation.

``position``
    The unloading position itself, anchored on the recorded completion and the
    standard unloading minutes for the gate and vehicle class.  This is the one
    that must not be double booked.

A trip that changed gates appears only at the gate it finally used: its
whole-trip wait must never be counted once at each gate.  Intervals are kept as
explicit [start, end) periods so the duty room can see *when* a resource was
occupied.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

from .model import Dataset
from .timeline import TripRecord


@dataclass(frozen=True)
class Occupancy:
    resource: str
    gate: str
    trip: str
    start: datetime
    end: datetime
    estimated: bool
    evidence: tuple[str, ...]

    @property
    def minutes(self) -> float:
        return (self.end - self.start).total_seconds() / 60


@dataclass
class Overlap:
    resource: str
    gate: str
    start: datetime
    end: datetime
    trips: tuple[str, ...]

    @property
    def minutes(self) -> float:
        return (self.end - self.start).total_seconds() / 60


@dataclass
class CountPeriod:
    gate: str
    start: datetime
    end: datetime
    count: int
    trips: tuple[str, ...]


def position_occupancies(
    dataset: Dataset, records: Iterable[TripRecord]
) -> list[Occupancy]:
    """Unloading-position intervals, anchored on each recorded completion."""
    resources = dataset.resources
    result: list[Occupancy] = []
    for record in records:
        if record.done is None:
            continue
        station = resources.unload[record.final_gate]
        minutes = station.minutes(record.vehicle_class)
        result.append(
            Occupancy(
                resource="position",
                gate=record.final_gate,
                trip=record.trip,
                start=record.done.clock - timedelta(minutes=minutes),
                end=record.done.clock,
                estimated=True,
                evidence=(record.done.event,),
            )
        )
    return sorted(result, key=lambda item: (item.gate, item.start, item.trip))


def scale_occupancies(
    dataset: Dataset, records: Iterable[TripRecord]
) -> list[Occupancy]:
    """Weighing-station intervals, anchored on the recorded weighing event."""
    resources = dataset.resources
    result: list[Occupancy] = []
    for record in records:
        if record.entry is None:
            continue
        station_name = resources.scale_for_gate[record.final_gate]
        station = resources.scale[station_name]
        minutes = station.minutes(record.vehicle_class)
        anchor = record.weigh.clock if record.weigh is not None else record.entry.clock
        result.append(
            Occupancy(
                resource="scale",
                gate=station_name,
                trip=record.trip,
                start=anchor - timedelta(minutes=minutes),
                end=anchor,
                estimated=record.weigh is None,
                evidence=tuple(
                    signal.event
                    for signal in (record.entry, record.weigh)
                    if signal is not None
                ),
            )
        )
    return sorted(result, key=lambda item: (item.gate, item.start, item.trip))


def inside_periods(records: Iterable[TripRecord]) -> list[CountPeriod]:
    """How many vehicles were inside each gate from entry to done/abandon."""
    events: dict[str, list[tuple[datetime, int, str]]] = {}
    for record in records:
        if record.entry is None:
            continue
        end = record.done.clock if record.done is not None else (
            record.abandon.clock if record.abandon is not None else None
        )
        if end is None:
            continue
        events.setdefault(record.entry.gate, []).append((record.entry.clock, 1, record.trip))
        events.setdefault(record.entry.gate, []).append((end, -1, record.trip))
    periods: list[CountPeriod] = []
    for gate, gate_events in events.items():
        gate_events.sort(key=lambda event: (event[0], event[1]))
        active: set[str] = set()
        previous = None
        for moment, delta, trip in gate_events:
            if previous is not None and moment > previous:
                periods.append(
                    CountPeriod(
                        gate=gate,
                        start=previous,
                        end=moment,
                        count=len(active),
                        trips=tuple(sorted(active)),
                    )
                )
            if delta == 1:
                active.add(trip)
            else:
                active.discard(trip)
            previous = moment
    return periods


def port_occupancies(dataset: Dataset) -> list[Occupancy]:
    """Confirmed and suggested mobile-line intervals from ``moves.csv``."""
    result: list[Occupancy] = []
    for move in dataset.moves:
        result.append(
            Occupancy(
                resource="port",
                gate=move.to_gate,
                trip=move.notice,
                start=move.start_at,
                end=move.ready_at,
                estimated=False,
                evidence=(f"{move.notice}:{move.state}",),
            )
        )
    return sorted(result, key=lambda item: (item.start, item.trip))


def overlaps(occupancies: Iterable[Occupancy], capacity: int = 1) -> list[Overlap]:
    grouped: dict[tuple[str, str], list[Occupancy]] = {}
    for item in occupancies:
        grouped.setdefault((item.resource, item.gate), []).append(item)
    result: list[Overlap] = []
    for (resource, gate), items in sorted(grouped.items()):
        items = sorted(items, key=lambda item: (item.start, item.trip))
        for index, item in enumerate(items):
            for other in items[index + 1:]:
                if other.start >= item.end:
                    break
                lo = max(item.start, other.start)
                hi = min(item.end, other.end)
                if hi > lo:
                    result.append(
                        Overlap(
                            resource=resource,
                            gate=gate,
                            start=lo,
                            end=hi,
                            trips=tuple(sorted({item.trip, other.trip})),
                        )
                    )
    return result


def peak_inside(periods: Iterable[CountPeriod]) -> dict[str, int]:
    peak: dict[str, int] = {}
    for period in periods:
        peak[period.gate] = max(peak.get(period.gate, 0), period.count)
    return peak


def busy_minutes(
    occupancies: Iterable[Occupancy], resource: str, gate: str | None = None
) -> float:
    return sum(
        item.minutes
        for item in occupancies
        if item.resource == resource and (gate is None or item.gate == gate)
    )
