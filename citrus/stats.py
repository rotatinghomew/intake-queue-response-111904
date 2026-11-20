"""Censoring-aware wait statistics and the two reporting measures.

Two measures are never mixed:

``user wait``
    Time from the first valid arrival to the point the driver left the yard,
    either because unloading finished or because the driver gave up.  An
    abandonment is a real wait (the driver did wait that long) and is also a
    failed service outcome.

``promise``
    Whether unloading finished by ``promised_complete_by``.  An abandonment is
    a miss. A trip that is still unresolved is pending, never a zero.

Every distribution is an interval sample.  A trip whose waiting-belt record
falls in a documented outage carries an arrival interval; a trip still in the
yard carries a right-censored wait (upper bound unknown).  Quantiles and
over-threshold shares are therefore reported as lower/upper bounds, together
with the share of the sample that is not yet observable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Sequence

from .timeline import (
    ABANDONED_STATUS,
    DONE_STATUS,
    INSIDE_STATUS,
    QUEUED_STATUS,
    UNSEEN_STATUS,
    TripRecord,
)

Interval = tuple[float, float | None]


@dataclass
class MeasureSummary:
    measure: str
    n_total: int
    n_resolved: int
    n_pending: int
    n_unknown_arrival: int
    n_unseen: int
    median_low: float | None = None
    median_high: float | None = None
    p90_low: float | None = None
    p90_high: float | None = None
    over_low: float | None = None
    over_high: float | None = None
    threshold: float | None = None
    note: str = ""

    @property
    def observable_share(self) -> float:
        if not self.n_total:
            return 0.0
        return self.n_resolved / self.n_total


@dataclass
class ShiftSummary:
    as_of: datetime
    total: int
    served: int
    abandoned: int
    pending: int
    unknown_arrival: int
    served_waits: list[Interval]
    abandoned_waits: list[Interval]
    pending_waits: list[Interval]
    unconfirmed: list[str] = field(default_factory=list)
    user: MeasureSummary | None = None
    promise: MeasureSummary | None = None
    warning_wait_minutes: int = 45

    @property
    def abandonment_rate(self) -> float:
        resolved = self.served + self.abandoned
        if not resolved:
            return 0.0
        return self.abandoned / resolved

    @property
    def unknown_share(self) -> float:
        if not self.total:
            return 0.0
        return self.unknown_arrival / self.total


def _quantile(sorted_values: Sequence[float], fraction: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    if weight == 0:
        return sorted_values[lower]
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def median_bounds(intervals: Sequence[Interval]) -> tuple[float | None, float | None]:
    """Identify bounds on the median of a right-censored interval sample.

    The lower bound uses each trip's lower bound; the upper bound treats every
    unresolved trip as infinite.  If more than half the sample is unresolved,
    the median has no finite upper bound.
    """
    if not intervals:
        return (None, None)
    lows = sorted(lo for lo, _ in intervals)
    lower = _quantile(lows, 0.5)
    finite_highs = [hi for _, hi in intervals if hi is not None]
    unresolved = len(intervals) - len(finite_highs)
    if unresolved > len(intervals) // 2:
        return (lower, None)
    highs = sorted(finite_highs + [float("inf")] * unresolved)
    upper = _quantile(highs, 0.5)
    if upper == float("inf"):
        return (lower, None)
    return (lower, upper)


def quantile_bounds(
    intervals: Sequence[Interval], fraction: float
) -> tuple[float | None, float | None]:
    if not intervals:
        return (None, None)
    lows = sorted(lo for lo, _ in intervals)
    lower = _quantile(lows, fraction)
    finite_highs = [hi for _, hi in intervals if hi is not None]
    unresolved = len(intervals) - len(finite_highs)
    # For an upper quantile the unresolved mass counts toward the top; if too
    # much is unresolved the quantile is not identified.
    if unresolved / len(intervals) > (1 - fraction):
        return (lower, None)
    highs = sorted(finite_highs + [float("inf")] * unresolved)
    upper = _quantile(highs, fraction)
    if upper is None or upper == float("inf"):
        return (lower, None)
    return (lower, upper)


def proportion_over(
    intervals: Sequence[Interval], threshold: float
) -> tuple[float | None, float | None]:
    if not intervals:
        return (None, None)
    n = len(intervals)
    lower = sum(1 for lo, _ in intervals if lo > threshold) / n
    upper = sum(
        1 for _, hi in intervals if hi is None or hi > threshold
    ) / n
    return (lower, upper)


def interval_mean(intervals: Sequence[Interval]) -> tuple[float | None, float | None]:
    if not intervals:
        return (None, None)
    lows = sum(lo for lo, _ in intervals) / len(intervals)
    if any(hi is None for _, hi in intervals):
        return (lows, None)
    highs = sum(hi for _, hi in intervals if hi is not None) / len(intervals)
    return (lows, highs)


def trip_intervals(
    records: Iterable[TripRecord], as_of: datetime
) -> tuple[list[Interval], list[Interval], list[Interval], dict[str, int]]:
    """Split records into served / abandoned / pending wait intervals.

    Pending trips keep an open upper bound.  Trips whose arrival cannot be
    confirmed are counted separately and never become zero-minute waits.
    """
    served: list[Interval] = []
    abandoned: list[Interval] = []
    pending: list[Interval] = []
    counts = {
        "served": 0,
        "abandoned": 0,
        "pending": 0,
        "unknown_arrival": 0,
        "unseen": 0,
        "interval": 0,
    }
    for record in records:
        if record.status == DONE_STATUS:
            interval = record.wait_interval()
            if interval is None:
                counts["unknown_arrival"] += 1
                continue
            served.append(interval)
            counts["served"] += 1
        elif record.status == ABANDONED_STATUS:
            interval = record.wait_interval()
            if interval is None:
                counts["unknown_arrival"] += 1
                continue
            abandoned.append(interval)
            counts["abandoned"] += 1
        elif record.status in {QUEUED_STATUS, INSIDE_STATUS}:
            interval = record.wait_interval(as_of=as_of)
            if interval is None:
                counts["unknown_arrival"] += 1
                continue
            pending.append((interval[0], None))
            counts["pending"] += 1
        elif record.status == UNSEEN_STATUS:
            counts["unseen"] += 1
        if record.arrival_basis == "outage":
            counts["interval"] += 1
    return served, abandoned, pending, counts


def summarize_shift(
    records: Iterable[TripRecord],
    as_of: datetime,
    warning_wait_minutes: int = 45,
) -> ShiftSummary:
    records = list(records)
    served, abandoned, pending, counts = trip_intervals(records, as_of)
    unconfirmed = [
        record.trip
        for record in records
        if record.status in {QUEUED_STATUS, INSIDE_STATUS}
        and record.arrival_lo is None
    ]
    user_intervals = [*served, *abandoned, *pending]
    promise_intervals: list[Interval] = []
    for record in records:
        if record.status == DONE_STATUS and record.done is not None:
            slack = (
                record.appointment.promised_complete_by - record.done.clock
            ).total_seconds() / 60
            # Negative slack = late; map to a zero/one miss indicator.
            promise_intervals.append((0.0 if slack >= 0 else 1.0, 0.0 if slack >= 0 else 1.0))
        elif record.status == ABANDONED_STATUS:
            promise_intervals.append((1.0, 1.0))
        elif record.status in {QUEUED_STATUS, INSIDE_STATUS}:
            # A trip still unresolved after its promised completion time has
            # already missed it; before that time the outcome is genuinely open.
            if as_of > record.appointment.promised_complete_by:
                promise_intervals.append((1.0, 1.0))
            else:
                promise_intervals.append((0.0, 1.0))
    total = len(records)
    user = MeasureSummary(
        measure="user_wait",
        n_total=total,
        n_resolved=counts["served"] + counts["abandoned"],
        n_pending=counts["pending"],
        n_unknown_arrival=counts["unknown_arrival"],
        n_unseen=counts["unseen"],
        median_low=median_bounds(user_intervals)[0],
        median_high=median_bounds(user_intervals)[1],
        p90_low=quantile_bounds(user_intervals, 0.9)[0],
        p90_high=quantile_bounds(user_intervals, 0.9)[1],
        over_low=proportion_over(user_intervals, warning_wait_minutes)[0],
        over_high=proportion_over(user_intervals, warning_wait_minutes)[1],
        threshold=warning_wait_minutes,
        note="abandonment counts at the actual time the driver left",
    )
    promise_over = proportion_over(promise_intervals, 0.5) if promise_intervals else (None, None)
    promise = MeasureSummary(
        measure="promise",
        n_total=total,
        n_resolved=counts["served"] + counts["abandoned"],
        n_pending=counts["pending"],
        n_unknown_arrival=counts["unknown_arrival"],
        n_unseen=counts["unseen"],
        over_low=promise_over[0],
        over_high=promise_over[1],
        threshold=0.5,
        note="share of trips that missed the promised completion time; "
        "0.5 stands for 'late'",
    )
    return ShiftSummary(
        as_of=as_of,
        total=total,
        served=counts["served"],
        abandoned=counts["abandoned"],
        pending=counts["pending"],
        unknown_arrival=counts["unknown_arrival"],
        served_waits=served,
        abandoned_waits=abandoned,
        pending_waits=pending,
        unconfirmed=unconfirmed,
        user=user,
        promise=promise,
        warning_wait_minutes=warning_wait_minutes,
    )


@dataclass
class NaiveComparison:
    entered_only_mean: float | None
    served_only_mean: float | None
    all_terminal_mean: float | None
    abandoned_mean: float | None
    entered_count: int
    served_count: int
    abandoned_count: int
    promise_attainment: float | None
    abandon_rate: float | None
    bias_note: str


def naive_comparison(
    records: Iterable[TripRecord], as_of: datetime | None = None
) -> NaiveComparison:
    """Show how the common shortcuts hide the drivers who gave up."""
    records = list(records)
    served: list[float] = []
    abandoned: list[float] = []
    entered_only: list[float] = []
    met = 0
    done = 0
    for record in records:
        if record.status in {QUEUED_STATUS, INSIDE_STATUS}:
            # Pending trips are excluded from the naive shortcut comparison;
            # they are neither served nor abandoned yet.
            continue
        interval = record.wait_interval(as_of=as_of)
        if interval is None:
            continue
        value = interval[1] if interval[1] is not None else interval[0]
        if record.status == DONE_STATUS:
            served.append(value)
            done += 1
            if record.done is not None and record.done.clock <= record.appointment.promised_complete_by:
                met += 1
            if record.entry is not None:
                entered_only.append(value)
        elif record.status == ABANDONED_STATUS:
            abandoned.append(value)
    all_terminal = served + abandoned
    return NaiveComparison(
        entered_only_mean=(sum(entered_only) / len(entered_only)) if entered_only else None,
        served_only_mean=(sum(served) / len(served)) if served else None,
        all_terminal_mean=(sum(all_terminal) / len(all_terminal)) if all_terminal else None,
        abandoned_mean=(sum(abandoned) / len(abandoned)) if abandoned else None,
        entered_count=len(entered_only),
        served_count=len(served),
        abandoned_count=len(abandoned),
        promise_attainment=(met / done) if done else None,
        abandon_rate=(len(abandoned) / len(all_terminal)) if all_terminal else None,
        bias_note=(
            "entered-only drops drivers who gave up; served-only is the same "
            "survivorship bias with a different name"
        ),
    )


def by_gate_totals(
    records: Iterable[TripRecord], as_of: datetime
) -> dict[str, ShiftSummary]:
    grouped: dict[str, list[TripRecord]] = {}
    for record in records:
        grouped.setdefault(record.final_gate, []).append(record)
    return {
        gate: summarize_shift(items, as_of)
        for gate, items in sorted(grouped.items())
    }
