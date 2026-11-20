"""Queue state, warning evidence and replay safety at an observation time.

Warnings are snapshots: only signals whose ``arrived_at`` is at or before the
observation time may be cited.  A departure recorded later is not allowed to
erase the reason a warning fired earlier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

from .model import (
    ABANDON,
    CALL,
    DONE,
    WAIT,
    Dataset,
)
from .timeline import ABANDONED_STATUS, DONE_STATUS, TripRecord, build_trip

STAGE_WAITING = "waiting"
STAGE_CALLED = "called"
STAGE_REASSIGNED = "reassigned"
STAGE_INSIDE = "inside"

QUEUE_STAGES = frozenset({STAGE_WAITING, STAGE_CALLED, STAGE_REASSIGNED})


@dataclass
class QueueItem:
    trip: str
    gate: str
    vehicle_class: str
    requested_gate: str
    arrival: datetime | None
    arrival_basis: str
    waited_minutes: float | None
    stage: str
    evidence: tuple[str, ...]
    warning: bool
    frozen: bool
    """Inside or already called: the vehicle must not be re-queued elsewhere."""
    projected_wait_minutes: float | None = None
    projected_miss: bool | None = None


@dataclass
class Snapshot:
    as_of: datetime
    gates: dict[str, list[QueueItem]] = field(default_factory=dict)
    inside: dict[str, int] = field(default_factory=dict)
    unseen: list[str] = field(default_factory=list)
    unknown_arrival: list[str] = field(default_factory=list)
    records: dict[str, TripRecord] = field(default_factory=dict)

    @property
    def active_items(self) -> list[QueueItem]:
        return [item for items in self.gates.values() for item in items]

    def queue_depth(self, gate: str) -> int:
        return sum(
            1 for item in self.gates.get(gate, []) if item.stage in QUEUE_STAGES
        )

    @property
    def total_queue(self) -> int:
        return sum(self.queue_depth(gate) for gate in self.gates)

    def warned(self) -> list[QueueItem]:
        return [item for item in self.active_items if item.warning]

    @property
    def unobserved_share(self) -> float:
        total = len(self.records) + len(self.unseen)
        if not total:
            return 0.0
        incomplete = len(self.unseen) + len(self.unknown_arrival)
        return incomplete / total


def _stage(record: TripRecord) -> str:
    if record.entry is not None:
        return STAGE_INSIDE
    if record.call is not None:
        return STAGE_CALLED
    if record.change is not None:
        return STAGE_REASSIGNED
    return STAGE_WAITING


def _standard_remaining(record: TripRecord, dataset: Dataset) -> float:
    """Standard remaining minutes from the current stage to unloading done."""
    resources = dataset.resources
    gate = record.final_gate
    station = resources.unload.get(gate)
    if station is None:
        return 0.0
    unload = station.minutes(record.vehicle_class)
    if record.weigh is not None:
        return unload
    scale_name = resources.scale_for_gate[gate]
    scale = resources.scale[scale_name]
    scale_minutes = scale.minutes(record.vehicle_class)
    if record.entry is not None:
        return scale_minutes + unload
    transfer = resources.corridor.minutes(record.vehicle_class) if record.change else 0.0
    return transfer + scale_minutes + unload


def snapshot(
    dataset: Dataset,
    as_of: datetime,
    warning_wait_minutes: int | None = None,
) -> Snapshot:
    threshold = (
        warning_wait_minutes
        if warning_wait_minutes is not None
        else dataset.policy.warning_wait_minutes
    )
    result = Snapshot(as_of=as_of)
    for trip in dataset.appointments:
        visible = dataset.visible_signals(trip, as_of)
        if not visible:
            result.unseen.append(trip)
            continue
        kinds = {signal.kind for signal in visible}
        if DONE in kinds or ABANDON in kinds:
            continue
        record = build_trip(dataset, trip, as_of=as_of)
        if record.status in {DONE_STATUS, ABANDONED_STATUS}:
            continue
        result.records[trip] = record
        gate = record.final_gate
        stage = _stage(record)
        if stage == STAGE_INSIDE:
            result.inside[gate] = result.inside.get(gate, 0) + 1
        waited = None
        if record.arrival_hi is not None:
            waited = max(0.0, (as_of - record.arrival_hi).total_seconds() / 60)
        else:
            result.unknown_arrival.append(trip)
        projected = None
        projected_miss = None
        if record.arrival_lo is not None:
            projected = (
                waited or 0.0
            ) + _standard_remaining(record, dataset)
            projected_miss = (
                as_of + _minutes(_standard_remaining(record, dataset))
                > record.appointment.promised_complete_by
            )
        warning = False
        if record.arrival_lo is not None and waited is not None:
            warning = waited >= threshold or bool(projected_miss)
        result.gates.setdefault(gate, []).append(
            QueueItem(
                trip=trip,
                gate=gate,
                vehicle_class=record.vehicle_class,
                requested_gate=record.requested_gate,
                arrival=record.arrival_lo,
                arrival_basis=record.arrival_basis,
                waited_minutes=waited,
                stage=stage,
                evidence=tuple(signal.event for signal in visible),
                warning=warning,
                frozen=stage != STAGE_WAITING,
                projected_wait_minutes=projected,
                projected_miss=projected_miss,
            )
        )
    for items in result.gates.values():
        items.sort(
            key=lambda item: (
                item.stage != STAGE_CALLED,
                item.arrival or as_of,
                item.trip,
            )
        )
    return result


def _minutes(value: float):
    from datetime import timedelta

    return timedelta(minutes=value)


@dataclass
class WarningReason:
    gate: str
    trip: str
    stage: str
    waited_minutes: float | None
    projected_wait_minutes: float | None
    arrival_basis: str
    queue_ahead: int
    evidence: tuple[str, ...]
    text: str


def warning_reasons(
    dataset: Dataset, as_of: datetime, warning_wait_minutes: int | None = None
) -> list[WarningReason]:
    state = snapshot(dataset, as_of, warning_wait_minutes)
    reasons: list[WarningReason] = []
    for gate, items in state.gates.items():
        ordered = sorted(items, key=lambda item: (item.arrival or as_of, item.trip))
        for index, item in enumerate(ordered):
            if not item.warning:
                continue
            bits = []
            if item.waited_minutes is not None:
                bits.append(f"已等 {item.waited_minutes:.0f} 分钟")
            if item.projected_miss:
                bits.append("按标准作业时长预计错过预约完成时刻")
            if item.arrival_basis == "outage":
                bits.append("到达只能确认在故障区间内")
            if item.stage in {STAGE_CALLED, STAGE_INSIDE}:
                bits.append("已叫号/已进场，处置中不得改排他队")
            reasons.append(
                WarningReason(
                    gate=gate,
                    trip=item.trip,
                    stage=item.stage,
                    waited_minutes=item.waited_minutes,
                    projected_wait_minutes=item.projected_wait_minutes,
                    arrival_basis=item.arrival_basis,
                    queue_ahead=sum(
                        1
                        for other in items
                        if (other.arrival or as_of) < (item.arrival or as_of)
                        and other.stage in QUEUE_STAGES
                    ),
                    evidence=item.evidence,
                    text="；".join(bits) or "预警",
                )
            )
    return reasons


def evidence_lines(dataset: Dataset, event_ids: Iterable[str]) -> list[str]:
    wanted = set(event_ids)
    lines: list[str] = []
    for trip in dataset.appointments:
        for signal in dataset.signals_for(trip):
            if signal.event not in wanted:
                continue
            lines.append(
                f"{signal.event} | {signal.kind} | {signal.gate} | "
                f"业务时刻 {signal.clock.isoformat()} | "
                f"可见时刻 {signal.visible_at.isoformat()} | {signal.source}"
            )
    return lines


def replay_integrity(
    dataset: Dataset,
    earlier: datetime,
    later: datetime,
    warning_wait_minutes: int | None = None,
) -> list[str]:
    """Report why a later replay may not use later records to clear a warning.

    For every trip warned at ``earlier`` we check whether a terminal record
    existed *and was visible* by then.  A warning that is only explainable with
    a record visible after ``earlier`` is exactly the case the duty room must
    not rewrite in hindsight.
    """
    before = snapshot(dataset, earlier, warning_wait_minutes)
    problems: list[str] = []
    for item in before.warned():
        extra = [
            signal
            for signal in dataset.signals_for(item.trip)
            if signal.kind in {DONE, ABANDON} and signal.visible_at <= earlier
        ]
        if extra:
            continue
        later_terminal = [
            signal
            for signal in dataset.signals_for(item.trip)
            if signal.kind in {DONE, ABANDON} and signal.visible_at <= later
        ]
        if later_terminal:
            first = later_terminal[0]
            problems.append(
                f"{item.trip} 在 {earlier.isoformat()} 已触发 {item.gate} 预警，"
                f"但结束记录 {first.event} 到 {first.visible_at.isoformat()} 才可见；"
                f"回放不得用它撤销预警"
            )
        else:
            problems.append(
                f"{item.trip} 在 {earlier.isoformat()} 已触发 {item.gate} 预警，"
                f"到 {later.isoformat()} 仍无可见结束记录；不得当作零等待"
            )
    return problems


def locate(
    dataset: Dataset,
    as_of: datetime,
    *,
    trip: str | None = None,
    gate: str | None = None,
) -> list[QueueItem]:
    """Find the current queue state for one trip and/or one gate.

    Each item carries the ids of the visible records the warning was based on,
    so a duty officer can retrace exactly what the room could see.
    """
    state = snapshot(dataset, as_of)
    items = state.active_items
    if trip is not None:
        items = [item for item in items if item.trip == trip]
    if gate is not None:
        items = [item for item in items if item.gate == gate]
    return items


def shift_warnings(
    dataset: Dataset, start: datetime, end: datetime
) -> list[tuple[datetime, WarningReason]]:
    """All warnings raised at the policy observation times inside a shift."""
    found: list[tuple[datetime, WarningReason]] = []
    for moment in dataset.policy.observe_at:
        if not (start <= moment <= end):
            continue
        for reason in warning_reasons(dataset, moment):
            found.append((moment, reason))
    return found
