"""Reconstruct one trip at a time, with explicit censoring.

A trip is identified by its appointment, never by the vehicle plate.  Its
first valid arrival is the first non-duplicate waiting-belt registration.  If
the waiting belt has a known outage the arrival becomes an interval; if no
other source can place the trip, the trip is reported as unconfirmed rather
than quietly dropped from the sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Sequence

from .model import (
    ABANDON,
    CALL,
    CHANGE,
    DONE,
    ENTRY,
    WAIT,
    WEIGH,
    Appointment,
    Dataset,
    Signal,
)

DONE_STATUS = "done"
ABANDONED_STATUS = "abandoned"
INSIDE_STATUS = "inside"
QUEUED_STATUS = "queued"
UNSEEN_STATUS = "unseen"

BASIS_EXACT = "exact"
BASIS_OUTAGE = "outage"
BASIS_UNCONFIRMED = "unconfirmed"


@dataclass(frozen=True)
class GateChange:
    clock: datetime
    from_gate: str
    to_gate: str
    event: str


@dataclass
class TripRecord:
    trip: str
    appointment: Appointment
    signals: list[Signal]
    yard: list[Signal]
    call: Signal | None
    entry: Signal | None
    weigh: Signal | None
    done: Signal | None
    abandon: Signal | None
    change: GateChange | None
    arrival_lo: datetime | None
    arrival_hi: datetime | None
    arrival_basis: str
    arrival_evidence: tuple[str, ...]
    final_gate: str
    status: str
    unresolved: bool

    @property
    def vehicle(self) -> str:
        return self.appointment.vehicle

    @property
    def vehicle_class(self) -> str:
        return self.appointment.vehicle_class

    @property
    def requested_gate(self) -> str:
        return self.appointment.requested_gate

    @property
    def viable(self) -> bool:
        """Whether a quantitative wait can be attached to this trip."""
        return self.arrival_lo is not None

    def terminal_clock(self) -> datetime | None:
        if self.done is not None:
            return self.done.clock
        if self.abandon is not None:
            return self.abandon.clock
        return None

    def wait_interval(self, as_of: datetime | None = None) -> tuple[float, float | None] | None:
        """Wait in minutes as (lower, upper); ``None`` upper means unresolved.

        For a resolved trip the interval comes from arrival uncertainty.  For an
        unresolved trip the lower bound uses the *latest* possible arrival and
        the current observation time, so an unresolved trip can never look like
        a zero-minute wait.
        """
        if not self.viable:
            return None
        terminal = self.terminal_clock()
        if terminal is not None:
            lo = (terminal - self.arrival_hi).total_seconds() / 60
            hi = (terminal - self.arrival_lo).total_seconds() / 60
            return (max(0.0, lo), hi)
        assert as_of is not None, "as_of required for unresolved trips"
        lo = (as_of - self.arrival_hi).total_seconds() / 60
        # Still waiting: the wait is at least this long and has no known end.
        return (max(0.0, lo), None)

    def waiting_before_entry(self, as_of: datetime | None = None) -> tuple[float, float | None] | None:
        """Queue wait before entering the gate (arrival -> entry)."""
        if not self.viable:
            return None
        if self.entry is None:
            if self.terminal_clock() is not None:
                return None
            assert as_of is not None
            lo = (as_of - self.arrival_hi).total_seconds() / 60
            return (max(0.0, lo), None)
        lo = (self.entry.clock - self.arrival_hi).total_seconds() / 60
        hi = (self.entry.clock - self.arrival_lo).total_seconds() / 60
        return (max(0.0, lo), hi)

    def duration_inside(self, as_of: datetime | None = None) -> float | None:
        if self.entry is None:
            return None
        end = self.terminal_clock()
        if end is None:
            if as_of is None:
                return None
            end = as_of
        return max(0.0, (end - self.entry.clock).total_seconds() / 60)

    def promise_met(self) -> bool | None:
        if self.done is None:
            return None
        return self.done.clock <= self.appointment.promised_complete_by

    def changed_gates(self) -> bool:
        return self.change is not None

    def evidence_ids(self) -> list[str]:
        return [signal.event for signal in self.signals]


def _first(signals: Sequence[Signal], kind: str) -> Signal | None:
    for signal in signals:
        if signal.kind == kind:
            return signal
    return None


def _last(signals: Sequence[Signal], kind: str) -> Signal | None:
    found = None
    for signal in signals:
        if signal.kind == kind:
            found = signal
    return found


def _choose_final_gate(
    signals: Sequence[Signal], appointment: Appointment
) -> str:
    for kind in (DONE, ABANDON, WEIGH, ENTRY):
        signal = _first(signals, kind)
        if signal is not None:
            return signal.gate
    signal = _last(signals, CALL)
    if signal is not None:
        return signal.gate
    signal = _last(signals, CHANGE)
    if signal is not None:
        return signal.gate
    return appointment.requested_gate


def _change_record(
    signals: Sequence[Signal], appointment: Appointment
) -> GateChange | None:
    change = _last(signals, CHANGE)
    if change is None:
        return None
    previous = appointment.requested_gate
    for signal in signals:
        if signal.clock >= change.clock:
            break
        if signal.kind in {WAIT, CALL}:
            previous = signal.gate
    if previous == change.gate:
        # A gate-change record to the gate already in use changes nothing.
        return None
    return GateChange(change.clock, previous, change.gate, change.event)


def _arrival_for_outage(
    first_other: datetime | None, dataset: Dataset
) -> tuple[datetime, datetime, tuple[str, ...]] | None:
    """Place a trip whose waiting-belt record is inside a known outage.

    The waiting-belt outage means the arrival happened while the belt was
    down, so the arrival lies in the outage window and no later than the first
    record from another source.  Returns inclusive (lo, hi) and evidence.
    """
    if first_other is None:
        return None
    best = None
    for outage in dataset.policy.yard_outages():
        if outage.start > first_other:
            continue
        hi = min(outage.end, first_other)
        if hi < outage.start:
            continue
        candidate = (outage.start, hi, (f"outage:{outage.start.isoformat()}",))
        if best is None or candidate[0] > best[0]:
            best = candidate
    return best


def build_trip(
    dataset: Dataset,
    trip: str,
    as_of: datetime | None = None,
) -> TripRecord:
    appointment = dataset.appointments[trip]
    signals = dataset.signals_for(trip)
    if as_of is not None:
        signals = [signal for signal in signals if signal.visible_at <= as_of]
    signals = sorted(signals, key=lambda item: (item.clock, item.event))
    yard = [signal for signal in signals if signal.kind == WAIT and not signal.duplicate]
    call = _first(signals, CALL)
    entry = _first(signals, ENTRY)
    weigh = _first(signals, WEIGH)
    done = _first(signals, DONE)
    abandon = _first(signals, ABANDON)
    change = _change_record(signals, appointment)

    first_other = None
    for signal in signals:
        if signal.kind != WAIT:
            first_other = signal.clock
            break

    if yard:
        arrival_lo = arrival_hi = yard[0].clock
        basis = BASIS_EXACT
        evidence = tuple(signal.event for signal in yard)
    else:
        placed = _arrival_for_outage(first_other, dataset)
        if placed is not None:
            arrival_lo, arrival_hi, evidence = placed
            basis = BASIS_OUTAGE
        else:
            arrival_lo = arrival_hi = None
            basis = BASIS_UNCONFIRMED
            evidence = ()

    final_gate = _choose_final_gate(signals, appointment)

    if done is not None:
        status = DONE_STATUS
    elif abandon is not None:
        status = ABANDONED_STATUS
    elif entry is not None:
        status = INSIDE_STATUS
    elif call is not None or change is not None or yard:
        status = QUEUED_STATUS
    else:
        status = UNSEEN_STATUS
    unresolved = status in {UNSEEN_STATUS, QUEUED_STATUS, INSIDE_STATUS}

    return TripRecord(
        trip=trip,
        appointment=appointment,
        signals=signals,
        yard=yard,
        call=call,
        entry=entry,
        weigh=weigh,
        done=done,
        abandon=abandon,
        change=change,
        arrival_lo=arrival_lo,
        arrival_hi=arrival_hi,
        arrival_basis=basis,
        arrival_evidence=evidence,
        final_gate=final_gate,
        status=status,
        unresolved=unresolved,
    )


def build_all(
    dataset: Dataset, as_of: datetime | None = None
) -> dict[str, TripRecord]:
    return {
        trip: build_trip(dataset, trip, as_of=as_of)
        for trip in dataset.appointments
    }


def duplicate_summary(dataset: Dataset) -> dict[str, int]:
    total = sum(len(items) for items in dataset.signals.values())
    duplicates = sum(
        1
        for items in dataset.signals.values()
        for signal in items
        if signal.duplicate
    )
    return {
        "signals": total,
        "duplicate_wait_signals": duplicates,
        "trips": len(dataset.appointments),
        "trips_without_wait": sum(
            1
            for trip in dataset.appointments
            if not any(
                signal.kind == WAIT and not signal.duplicate
                for signal in dataset.signals_for(trip)
            )
        ),
    }


def outage_triage(dataset: Dataset) -> dict[str, list[str]]:
    """Classify trips with no waiting-belt record.

    ``confirmed`` trips can be placed inside a documented outage and remain in
    the sample as interval-censored observations.  ``unconfirmed`` trips are
    listed explicitly and never silently removed.
    """
    confirmed: list[str] = []
    unconfirmed: list[str] = []
    affected_outages: dict[str, list[str]] = {}
    for trip in dataset.appointments:
        if any(
            signal.kind == WAIT and not signal.duplicate
            for signal in dataset.signals_for(trip)
        ):
            continue
        record = build_trip(dataset, trip)
        if record.arrival_basis == BASIS_OUTAGE:
            confirmed.append(trip)
            key = record.arrival_evidence[0] if record.arrival_evidence else "outage"
            affected_outages.setdefault(key, []).append(trip)
        else:
            unconfirmed.append(trip)
    return {
        "confirmed": confirmed,
        "unconfirmed": unconfirmed,
        "outages": {key: sorted(value) for key, value in affected_outages.items()},
    }


@dataclass
class WaitSample:
    records: list[TripRecord] = field(default_factory=list)
    as_of: datetime | None = None

    def add(self, record: TripRecord) -> None:
        self.records.append(record)

    def intervals(self) -> list[tuple[str, float, float | None]]:
        values = []
        for record in self.records:
            interval = record.wait_interval(self.as_of)
            if interval is None:
                continue
            values.append((record.trip, interval[0], interval[1]))
        return values

    def median_bounds(self) -> tuple[float | None, float | None]:
        return median_bounds(self.intervals())

    def over_bounds(self, threshold: float) -> tuple[float | None, float | None]:
        return proportion_over(self.intervals(), threshold)

    def known_share_over(self, threshold: float) -> float | None:
        values = self.intervals()
        if not values:
            return None
        known = [value for value in values if value[2] is not None]
        if not known:
            return None
        return sum(1 for _, lo, _ in known if lo > threshold) / len(known)


def _quantile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("empty sample")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def median_bounds(
    intervals: Sequence[tuple[str, float, float | None]]
) -> tuple[float | None, float | None]:
    """Bounds on the median under interval censoring.

    The lower bound treats every unresolved trip at its current elapsed time;
    the upper bound treats each unresolved trip as infinitely long.  If enough
    trips are unresolved the upper bound is unbounded (``None``).
    """
    if not intervals:
        return (None, None)
    lows = sorted(lo for _, lo, _ in intervals)
    lower = _quantile(lows, 0.5)
    highs = sorted((hi for _, _, hi in intervals if hi is not None))
    unresolved = len(intervals) - len(highs)
    if unresolved > len(intervals) // 2:
        return (lower, None)
    all_highs = sorted(
        [hi for _, _, hi in intervals if hi is not None]
        + [float("inf")] * unresolved
    )
    upper = _quantile(all_highs, 0.5)
    if upper == float("inf"):
        return (lower, None)
    return (lower, upper)


def proportion_over(
    intervals: Sequence[tuple[str, float, float | None]], threshold: float
) -> tuple[float | None, float | None]:
    """Bounds on P(wait > threshold)."""
    if not intervals:
        return (None, None)
    n = len(intervals)
    lower = sum(1 for _, lo, _ in intervals if lo > threshold) / n
    upper = sum(1 for _, _, hi in intervals if hi is None or hi > threshold) / n
    return (lower, upper)


def by_gate(records: Iterable[TripRecord]) -> dict[str, list[TripRecord]]:
    grouped: dict[str, list[TripRecord]] = {}
    for record in records:
        grouped.setdefault(record.final_gate, []).append(record)
    return grouped
