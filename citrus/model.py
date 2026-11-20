"""Read and normalise the synthetic receiving records.

Nothing in this module writes to ``data/``. The source quirks that matter
downstream are handled here:

* source clocks are corrected with ``policy.clock_offset_minutes``
  (``unified = source_clock - offset``);
* repeated waiting-belt registrations of the same trip at the same gate are
  marked as duplicates and never become extra vehicles;
* ``arrived_at`` is kept separate from the business event time, so a replay can
  use only what the duty room could actually see at a given moment.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

WAIT = "候车"
CALL = "叫号"
ENTRY = "进门"
WEIGH = "过磅"
CHANGE = "改门"
DONE = "卸完"
ABANDON = "放弃"

QUEUE_KINDS = frozenset({WAIT, CALL, CHANGE})
TERMINAL_KINDS = frozenset({DONE, ABANDON})


class SourceError(ValueError):
    """Raised when a source record cannot be interpreted safely."""


def parse_time(value: str) -> datetime:
    if not value:
        raise SourceError("empty timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise SourceError(f"bad timestamp {value!r}") from exc
    if parsed.tzinfo is None:
        raise SourceError(f"timestamp without timezone: {value!r}")
    return parsed


def _number(value: str, where: str) -> float:
    if value is None or value == "":
        raise SourceError(f"empty number at {where}")
    try:
        return float(value)
    except ValueError as exc:
        raise SourceError(f"bad number {value!r} at {where}") from exc


@dataclass(frozen=True)
class Appointment:
    trip: str
    vehicle: str
    requested_gate: str
    available_gates: tuple[str, ...]
    vehicle_class: str
    promised_complete_by: datetime

    @property
    def is_large(self) -> bool:
        return self.vehicle_class == "大车"

    def can_use(self, gate: str) -> bool:
        return gate in self.available_gates


@dataclass(frozen=True)
class Capacity:
    station: str
    kind: str
    gate: str | None
    small_min: float
    large_min: float
    shared_with: tuple[str, ...]

    def minutes(self, vehicle_class: str) -> float:
        return self.large_min if vehicle_class == "大车" else self.small_min


@dataclass(frozen=True)
class RawSignal:
    event: str
    trip: str
    vehicle: str
    source: str
    kind: str
    gate: str
    source_clock: datetime
    arrived_at: datetime


@dataclass(frozen=True)
class Signal:
    """A signal on the unified clock, with visibility and duplicate flags."""

    event: str
    trip: str
    vehicle: str
    source: str
    kind: str
    gate: str
    clock: datetime
    """Business time after removing the source clock offset."""
    visible_at: datetime
    """Unified time at which the duty room first had this record."""
    duplicate: bool = False
    duplicate_of: str | None = None
    raw_clock: datetime | None = None
    source_offset_minutes: int | None = None

    @property
    def source_clock(self) -> datetime | None:
        """The untouched clock value from the source file."""
        return self.raw_clock


@dataclass(frozen=True)
class Move:
    notice: str
    line: str
    from_gate: str
    to_gate: str
    start_at: datetime
    ready_at: datetime
    state: str
    confirmed_at: datetime | None = None
    source: str | None = None

    @property
    def confirmed(self) -> bool:
        return self.state == "已确认"

    @property
    def downtime_minutes(self) -> float:
        return (self.ready_at - self.start_at).total_seconds() / 60


@dataclass(frozen=True)
class Outage:
    source: str
    start: datetime
    end: datetime

    def covers(self, moment: datetime) -> bool:
        return self.start <= moment <= self.end

    def overlaps(self, start: datetime, end: datetime) -> bool:
        return start <= self.end and end >= self.start


@dataclass(frozen=True)
class Policy:
    clock_offset_minutes: dict[str, int]
    outages: tuple[Outage, ...]
    move_minutes: int
    observe_at: tuple[datetime, ...]
    shift_end: datetime
    warning_wait_minutes: int

    def offset(self, source: str) -> int:
        if source not in self.clock_offset_minutes:
            raise SourceError(f"no clock offset for source {source!r}")
        return self.clock_offset_minutes[source]

    def yard_outages(self) -> tuple[Outage, ...]:
        return tuple(item for item in self.outages if item.source == "yard")


@dataclass(frozen=True)
class Resources:
    """Capacity stations expressed as schedulable resources."""

    unload: dict[str, Capacity]
    scale: dict[str, Capacity]
    scale_for_gate: dict[str, str]
    corridor: Capacity
    port: Capacity

    @property
    def gates(self) -> tuple[str, ...]:
        return tuple(sorted(self.unload))


@dataclass
class Dataset:
    appointments: dict[str, Appointment]
    signals: dict[str, list[Signal]]
    capacity: dict[str, Capacity]
    moves: list[Move]
    policy: Policy
    resources: Resources

    def trips_in_order(self) -> list[str]:
        return sorted(
            self.appointments,
            key=lambda trip: (
                self.appointments[trip].promised_complete_by,
                trip,
            ),
        )

    def signals_for(self, trip: str) -> list[Signal]:
        return self.signals.get(trip, [])

    def visible_signals(self, trip: str, as_of: datetime) -> list[Signal]:
        return [item for item in self.signals_for(trip) if item.visible_at <= as_of]

    def shift_date(self, moment: datetime) -> str:
        return moment.date().isoformat()

    def shift_end_for(self, moment: datetime) -> datetime:
        """The policy shift end on the date of ``moment``."""
        return self.policy.shift_end.replace(
            year=moment.year, month=moment.month, day=moment.day
        )


def _read_csv(name: str) -> list[dict[str, str]]:
    path = DATA_DIR / f"{name}.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise SourceError(f"{path} is empty")
    return rows


def load_policy() -> Policy:
    raw = json.loads((DATA_DIR / "policy.json").read_text(encoding="utf-8"))
    outages = tuple(
        Outage(item["source"], parse_time(item["start"]), parse_time(item["end"]))
        for item in raw["outages"]
    )
    return Policy(
        clock_offset_minutes=dict(raw["clock_offset_minutes"]),
        outages=outages,
        move_minutes=int(raw["move_minutes"]),
        observe_at=tuple(parse_time(value) for value in raw["observe_at"]),
        shift_end=parse_time(raw["shift_end"]),
        warning_wait_minutes=int(raw["warning_wait_minutes"]),
    )


def load_appointments() -> dict[str, Appointment]:
    appointments: dict[str, Appointment] = {}
    for row in _read_csv("appointments"):
        trip = row["trip"]
        if not trip:
            raise SourceError("appointment without trip identity")
        if trip in appointments:
            raise SourceError(f"duplicate appointment for trip {trip}")
        gates = tuple(part for part in row["available_gates"].split(";") if part)
        if row["requested_gate"] not in gates:
            raise SourceError(f"requested gate not in availability for {trip}")
        appointments[trip] = Appointment(
            trip=trip,
            vehicle=row["vehicle"],
            requested_gate=row["requested_gate"],
            available_gates=gates,
            vehicle_class=row["vehicle_class"],
            promised_complete_by=parse_time(row["promised_complete_by"]),
        )
    return appointments


def load_capacity() -> dict[str, Capacity]:
    capacity: dict[str, Capacity] = {}
    for row in _read_csv("capacity"):
        station = row["station"]
        if station in capacity:
            raise SourceError(f"duplicate capacity station {station}")
        capacity[station] = Capacity(
            station=station,
            kind=row["kind"],
            gate=row["gate"] or None,
            small_min=_number(row["small_min"], station),
            large_min=_number(row["large_min"], station),
            shared_with=tuple(
                part for part in row["shared_with"].split(";") if part
            ),
        )
    return capacity


def load_moves() -> list[Move]:
    moves: list[Move] = []
    for row in _read_csv("moves"):
        moves.append(
            Move(
                notice=row["notice"],
                line=row["line"],
                from_gate=row["from_gate"],
                to_gate=row["to_gate"],
                start_at=parse_time(row["start_at"]),
                ready_at=parse_time(row["ready_at"]),
                state=row["state"],
            )
        )
    return moves


def build_resources(capacity: dict[str, Capacity]) -> Resources:
    unload = {
        item.gate: item
        for item in capacity.values()
        if item.kind == "正常卸料" and item.gate
    }
    scales = {
        item.station: item
        for item in capacity.values()
        if item.kind in {"共享称重", "称重"}
    }
    scale_for_gate: dict[str, str] = {}
    for station, item in scales.items():
        if item.gate:
            scale_for_gate[item.gate] = station
        for shared in item.shared_with:
            scale_for_gate[shared] = station
    corridor = next(
        (item for item in capacity.values() if item.kind == "跨门通道"), None
    )
    port = next(
        (item for item in capacity.values() if item.kind == "机动卸料"), None
    )
    if not unload or not scales or corridor is None or port is None:
        raise SourceError("capacity file is missing a required station")
    for gate in unload:
        if gate not in scale_for_gate:
            raise SourceError(f"gate {gate} has no weighing station")
    if set(scale_for_gate) - set(unload):
        raise SourceError("weighing station references an unknown gate")
    return Resources(
        unload=unload,
        scale=scales,
        scale_for_gate=scale_for_gate,
        corridor=corridor,
        port=port,
    )


def _mark_duplicates(raw: list[RawSignal]) -> dict[str, str | None]:
    """Return event id -> first event id for repeated waiting-belt pings.

    A registration is a duplicate only when the same trip already has an
    earlier *waiting* registration at the same gate. The first registration
    counts; the repeat adds neither a vehicle nor a queue position.
    """
    first_for: dict[tuple[str, str], str] = {}
    duplicate_of: dict[str, str | None] = {}
    ordered = sorted(
        raw,
        key=lambda item: (item.source_clock, item.event),
    )
    for item in ordered:
        duplicate_of[item.event] = None
        if item.kind != WAIT:
            continue
        key = (item.trip, item.gate)
        if key in first_for:
            duplicate_of[item.event] = first_for[key]
        else:
            first_for[key] = item.event
    return duplicate_of


def load_signals(policy: Policy | None = None) -> dict[str, list[Signal]]:
    policy = policy or load_policy()
    raw: list[RawSignal] = []
    seen_events: set[str] = set()
    for row in _read_csv("signals"):
        event = row["event"]
        if event in seen_events:
            raise SourceError(f"duplicate signal id {event}")
        seen_events.add(event)
        raw.append(
            RawSignal(
                event=event,
                trip=row["trip"],
                vehicle=row["vehicle"],
                source=row["source"],
                kind=row["kind"],
                gate=row["gate"],
                source_clock=parse_time(row["source_clock"]),
                arrived_at=parse_time(row["arrived_at"]),
            )
        )
    duplicate_of = _mark_duplicates(raw)
    signals: dict[str, list[Signal]] = {}
    for item in raw:
        offset = policy.offset(item.source)
        clock = item.source_clock - timedelta(minutes=offset)
        if item.arrived_at < clock:
            raise SourceError(
                f"visibility precedes business time for event {item.event}"
            )
        first = duplicate_of[item.event]
        signal = Signal(
            event=item.event,
            trip=item.trip,
            vehicle=item.vehicle,
            source=item.source,
            kind=item.kind,
            gate=item.gate,
            clock=clock,
            visible_at=item.arrived_at,
            duplicate=first is not None,
            duplicate_of=first,
            raw_clock=item.source_clock,
            source_offset_minutes=offset,
        )
        signals.setdefault(item.trip, []).append(signal)
    for items in signals.values():
        items.sort(key=lambda signal: (signal.clock, signal.event))
    return signals


def load_dataset() -> Dataset:
    policy = load_policy()
    appointments = load_appointments()
    signals = load_signals(policy)
    capacity = load_capacity()
    moves = load_moves()
    resources = build_resources(capacity)
    unknown = set(signals) - set(appointments)
    if unknown:
        raise SourceError(f"signals without appointment: {sorted(unknown)[:5]}")
    for trip, items in signals.items():
        appointment = appointments[trip]
        for signal in items:
            if not appointment.can_use(signal.gate):
                raise SourceError(
                    f"gate {signal.gate} not available for trip {trip} "
                    f"({signal.event})"
                )
    return Dataset(
        appointments=appointments,
        signals=signals,
        capacity=capacity,
        moves=moves,
        policy=policy,
        resources=resources,
    )
