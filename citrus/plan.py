"""Forward-looking options for the mobile unloading line.

Options are deliberately constrained by the business rules:

* a trip may only be served at a gate in its appointment's ``available_gates``;
* a vehicle already called or inside is frozen: it keeps its place and is
  never re-queued behind another gate to improve a statistic;
* the mobile line needs ``policy.move_minutes`` of downtime to relocate and
  serves one gate at a time;
* a confirmed dispatch move is inherited as the starting point; a later
  *suggested* move may be superseded, but the same line is never counted at two
  gates at once;
* an option that looks better only because vehicles fall outside the visible
  ledger is flagged, not rewarded.

The comparison is a deterministic queue simulation seeded from the current
visible state.  It is a simplification, so the comparison also carries a
backtest error; improvements smaller than that error are reported as "not
distinguishable", and the recommendation falls back to holding or to a manual
check instead of inventing a move.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable

from .model import ABANDON, CALL, CHANGE, DONE, ENTRY, WEIGH, Dataset, Move, Signal
from .timeline import ABANDONED_STATUS, DONE_STATUS, build_trip

ACTION_HOLD = "hold"
ACTION_MOVE = "move"
ACTION_MANUAL = "manual"

CALL_LEAD_MINUTES = 2.0
DEFAULT_IMPROVEMENT_MARGIN_MINUTES = 3.0
MIN_BACKTEST_TRIPS = 20


@dataclass
class JobState:
    trip: str
    vehicle_class: str
    gate: str
    requested_gate: str
    arrival: datetime
    arrival_basis: str
    stage: str
    entered: bool
    called: bool
    changed: bool
    release: datetime
    called_at: datetime | None
    unload_ready: datetime | None
    evidence: tuple[str, ...] = ()

    @property
    def frozen(self) -> bool:
        return self.entered or self.called


@dataclass
class Option:
    action: str
    line: str
    gate: str | None
    start_at: datetime | None
    ready_at: datetime | None
    feasible: bool
    rationale: str
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, object] = field(default_factory=dict)
    manual_checks: list[str] = field(default_factory=list)
    indistinguishable_from: str | None = None

    @property
    def improvement(self) -> float | None:
        value = self.metrics.get("median_improvement")
        return float(value) if value is not None else None


@dataclass
class Comparison:
    as_of: datetime
    shift_end: datetime
    options: list[Option]
    recommendation: Option
    backtest_error_minutes: float | None
    backtest_note: str
    inherited_moves: list[str] = field(default_factory=list)
    superseded_moves: list[str] = field(default_factory=list)
    shift_closed: bool = False


# ---------------------------------------------------------------------------
# Visible state


def _trip_gate(appointment, visible: list[Signal]) -> tuple[str, bool]:
    for kind in (DONE, ABANDON, WEIGH, ENTRY, CALL):
        for signal in visible:
            if signal.kind == kind:
                return signal.gate, False
    for signal in reversed(visible):
        if signal.kind == CHANGE:
            return signal.gate, signal.gate != appointment.requested_gate
    return appointment.requested_gate, False


def visible_jobs(
    dataset: Dataset, as_of: datetime, gate_filter: str | None = None
) -> list[JobState]:
    jobs: list[JobState] = []
    for trip, appointment in dataset.appointments.items():
        visible = dataset.visible_signals(trip, as_of)
        if not visible:
            continue
        kinds = {signal.kind for signal in visible}
        if DONE in kinds or ABANDON in kinds:
            continue
        record = build_trip(dataset, trip, as_of)
        if record.status in {DONE_STATUS, ABANDONED_STATUS}:
            continue
        if gate_filter is not None and record.final_gate != gate_filter:
            continue
        _, changed = _trip_gate(appointment, visible)
        entered = record.entry is not None
        called = record.call is not None and not entered
        if record.arrival_hi is not None:
            arrival = record.arrival_hi
            basis = record.arrival_basis
        else:
            arrival = as_of
            basis = "unconfirmed"
        if entered:
            stage = "unload" if record.weigh is not None else "scale"
        elif called:
            stage = "called"
        else:
            stage = "waiting"
        release = as_of if entered or called else max(as_of, arrival)
        jobs.append(
            JobState(
                trip=trip,
                vehicle_class=appointment.vehicle_class,
                gate=record.final_gate,
                requested_gate=appointment.requested_gate,
                arrival=arrival,
                arrival_basis=basis,
                stage=stage,
                entered=entered,
                called=called,
                changed=changed,
                release=release,
                called_at=record.call.clock if record.call else None,
                unload_ready=as_of if record.weigh is not None else None,
                evidence=tuple(signal.event for signal in visible),
            )
        )
    jobs.sort(key=lambda job: (job.release, job.trip))
    return jobs


def _port_location(
    dataset: Dataset, as_of: datetime, moves: list[Move] | None = None
) -> tuple[str | None, datetime]:
    """Where the mobile line is and when it is free, honouring confirmed moves.

    A move that has started but not finished keeps the line busy until ready.
    ``moves`` defaults to the source moves plus confirmed operator decisions.
    """
    if moves is None:
        from .state import effective_moves

        moves = effective_moves(dataset, as_of)
    location: str | None = None
    free = as_of
    for move in sorted(moves, key=lambda item: (item.start_at, item.notice)):
        if move.ready_at <= as_of:
            location, free = move.to_gate, max(free, move.ready_at)
        elif move.start_at <= as_of < move.ready_at:
            location, free = move.to_gate, max(free, move.ready_at)
    if location is None:
        location = dataset.resources.gates[0] if dataset.resources.gates else None
    return location, free


# ---------------------------------------------------------------------------
# Simulation


@dataclass
class SimulationResult:
    done: dict[str, datetime]
    called: dict[str, datetime]
    server: dict[str, str]
    metrics: dict[str, object]
    timeline: list[tuple[datetime, str, str]] = field(default_factory=list)


def _simulate(
    dataset: Dataset,
    as_of: datetime,
    horizon: datetime,
    jobs: list[JobState],
    port_gate: str | None,
    moves: list[Move],
    *,
    ignore_called_order: bool = False,
) -> SimulationResult:
    import heapq
    import itertools

    resources = dataset.resources
    gates = sorted(resources.unload)
    counter = itertools.count()
    heap: list[tuple[datetime, int, int, str, object]] = []

    def push(when: datetime, priority: int, kind: str, payload) -> None:
        heapq.heappush(heap, (when, priority, next(counter), kind, payload))

    unload_free = {gate: as_of for gate in gates}
    scale_free = {station: as_of for station in resources.scale}
    unload_queue: dict[str, list[JobState]] = {gate: [] for gate in gates}
    scale_queue: dict[str, list[JobState]] = {
        station: [] for station in resources.scale
    }
    waiting: dict[str, list[JobState]] = {gate: [] for gate in gates}
    inside = {gate: 0 for gate in gates}
    admitted: set[str] = set()
    done: dict[str, datetime] = {}
    called: dict[str, datetime] = {}
    server: dict[str, str] = {}
    port = {"gate": port_gate, "idle": as_of}
    moving_until = as_of

    # Confirmed moves are inherited in order and cannot overlap: each starts no
    # earlier than the previous relocation finished.
    last_ready = as_of
    for move in sorted(moves, key=lambda item: (item.start_at, item.notice)):
        if move.ready_at <= as_of:
            port["gate"] = move.to_gate
            port["idle"] = max(port["idle"], move.ready_at)
            last_ready = max(last_ready, move.ready_at)
        else:
            start = max(move.start_at, last_ready, as_of)
            ready = max(move.ready_at, start + timedelta(minutes=dataset.policy.move_minutes))
            scheduled = Move(
                notice=move.notice,
                line=move.line,
                from_gate=move.from_gate,
                to_gate=move.to_gate,
                start_at=start,
                ready_at=ready,
                state=move.state,
            )
            last_ready = ready
            if scheduled.start_at > as_of:
                push(scheduled.start_at, 0, "move", scheduled)
    port["idle"] = max(port["idle"], as_of)

    def port_here(gate: str, when: datetime | None = None) -> bool:
        moment = as_of if when is None else when
        return (
            port["gate"] == gate
            and port["idle"] <= moment
            and moving_until <= moment
        )

    def start_unload(when: datetime, gate: str) -> None:
        queue = unload_queue[gate]
        if not queue:
            return
        queue.sort(key=lambda job: (job.unload_ready or job.release, job.trip))
        job = queue[0]
        uses = []
        if unload_free[gate] <= when:
            uses.append(("resident", unload_free[gate]))
        if port_here(gate, when):
            uses.append(("port", port["idle"]))
        if not uses:
            return
        use, _ = min(uses, key=lambda item: item[1])
        queue.pop(0)
        station = resources.port if use == "port" else resources.unload[gate]
        finish = when + timedelta(minutes=station.minutes(job.vehicle_class))
        if use == "port":
            port["idle"] = finish
        else:
            unload_free[gate] = finish
        job.stage = "done"
        done[job.trip] = finish
        server[job.trip] = use
        push(finish, 2, "done", (job, gate))

    def start_scale(when: datetime, station_name: str) -> None:
        queue = scale_queue[station_name]
        if not queue or scale_free[station_name] > when:
            return
        queue.sort(key=lambda job: (job.called_at or job.release, job.trip))
        job = queue.pop(0)
        station = resources.scale[station_name]
        scale_free[station_name] = when + timedelta(minutes=station.minutes(job.vehicle_class))
        job.unload_ready = scale_free[station_name]
        job.stage = "scale_done"
        unload_queue[job.gate].append(job)
        push(scale_free[station_name], 3, "scale_done", (job, station_name))
        start_unload(scale_free[station_name], job.gate)

    def admit(when: datetime, gate: str) -> None:
        if not waiting[gate]:
            return
        if ignore_called_order:
            waiting[gate].sort(key=lambda job: (job.release, job.trip))
        else:
            # Called vehicles keep their call order; others by arrival.
            waiting[gate].sort(
                key=lambda job: (
                    0 if job.called else 1,
                    job.called_at or job.release,
                    job.trip,
                )
            )
        capacity = 1 + (1 if port_here(gate, when) else 0)
        while waiting[gate] and inside[gate] < capacity:
            job = waiting[gate].pop(0)
            inside[gate] += 1
            admitted.add(job.trip)
            job.called_at = when
            called[job.trip] = when
            push(when + timedelta(minutes=CALL_LEAD_MINUTES), 1, "entry", (job, gate))

    for job in jobs:
        if job.entered or job.called:
            inside[job.gate] += 1
            admitted.add(job.trip)
            if job.called_at is not None:
                called[job.trip] = job.called_at
            station_name = resources.scale_for_gate[job.gate]
            if job.unload_ready is not None:
                unload_queue[job.gate].append(job)
            else:
                scale_queue[station_name].append(job)
        else:
            waiting[job.gate].append(job)
    for gate in gates:
        unload_queue[gate].sort(
            key=lambda job: (job.unload_ready or job.release, job.trip)
        )
        start_unload(as_of, gate)
        admit(as_of, gate)

    while heap:
        when, _, _, kind, payload = heapq.heappop(heap)
        if when > horizon:
            break
        if kind == "entry":
            job, gate = payload
            station_name = resources.scale_for_gate[gate]
            scale_queue[station_name].append(job)
            start_scale(when, station_name)
        elif kind == "scale_done":
            _, station_name = payload
            start_scale(when, station_name)
        elif kind == "done":
            job, gate = payload
            inside[gate] = max(0, inside[gate] - 1)
            admit(when, gate)
            start_unload(when, gate)
            start_scale(when, resources.scale_for_gate[gate])
        elif kind == "move":
            move: Move = payload
            if port["idle"] > when:
                push(port["idle"], 0, "move", move)
                continue
            moving_until = when + timedelta(minutes=dataset.policy.move_minutes)
            port["gate"] = move.to_gate
            port["idle"] = moving_until
            push(moving_until, 0, "move_ready", move)
        elif kind == "move_ready":
            for gate in gates:
                start_unload(when, gate)
                admit(when, gate)

    metrics = _metrics(dataset, as_of, jobs, done)
    return SimulationResult(
        done=done, called=called, server=server, metrics=metrics
    )


def _metrics(
    dataset: Dataset,
    as_of: datetime,
    jobs: list[JobState],
    done: dict[str, datetime],
) -> dict[str, object]:
    from .stats import _quantile

    waits: list[float] = []
    missed = 0
    per_gate: dict[str, dict[str, object]] = {}
    for job in jobs:
        bucket = per_gate.setdefault(job.gate, {"waits": [], "pending": 0})
        finish = done.get(job.trip)
        if finish is None:
            bucket["pending"] += 1
            continue
        wait = (finish - job.arrival).total_seconds() / 60
        waits.append(wait)
        bucket["waits"].append(wait)
        if finish > dataset.appointments[job.trip].promised_complete_by:
            missed += 1
    ordered = sorted(waits)
    gate_metrics: dict[str, dict[str, object]] = {}
    for gate, bucket in sorted(per_gate.items()):
        gate_waits = sorted(bucket["waits"])
        gate_metrics[gate] = {
            "served": len(gate_waits),
            "pending": bucket["pending"],
            "median_wait": _quantile(gate_waits, 0.5) if gate_waits else None,
            "p90_wait": _quantile(gate_waits, 0.9) if gate_waits else None,
            "max_wait": gate_waits[-1] if gate_waits else None,
        }
    return {
        "jobs": len(jobs),
        "served": len(waits),
        "unserved": len(jobs) - len(waits),
        "mean_wait": (sum(waits) / len(waits)) if waits else None,
        "median_wait": _quantile(ordered, 0.5) if ordered else None,
        "p90_wait": _quantile(ordered, 0.9) if ordered else None,
        "max_wait": ordered[-1] if ordered else None,
        "missing_promise": missed,
        "missing_promise_share": (missed / len(waits)) if waits else None,
        "per_gate": gate_metrics,
    }


# ---------------------------------------------------------------------------
# Moves and option construction


def _future_moves(
    dataset: Dataset, as_of: datetime, moves: list[Move] | None = None
) -> list[Move]:
    if moves is None:
        from .state import effective_moves

        moves = effective_moves(dataset, as_of)
    return [move for move in moves if move.ready_at > as_of]


def _backtest_day(
    dataset: Dataset, day: str, at: datetime, lookback_days: int = 14
) -> tuple[float | None, str]:
    """Forward-replay earlier days from the same clock time and compare.

    This is a model-fidelity check, not a business claim.  For each earlier
    day the simulation starts at ``at`` (e.g. 06:40), uses only records visible
    by then, and is scored against unloading completions that actually
    happened by the shift end.  The mean absolute error is the size a
    difference between two options has to exceed before the model can
    distinguish them.
    """
    import statistics

    dates = sorted(
        {
            appointment.promised_complete_by.date().isoformat()
            for appointment in dataset.appointments.values()
        }
    )
    prior = [candidate for candidate in dates if candidate < day][-lookback_days:]
    errors: list[float] = []
    used_days = 0
    for candidate in prior:
        moment = at.replace(
            year=int(candidate[:4]),
            month=int(candidate[5:7]),
            day=int(candidate[8:10]),
        )
        shift_end = moment.replace(
            hour=dataset.policy.shift_end.hour,
            minute=dataset.policy.shift_end.minute,
            second=0,
            microsecond=0,
        )
        jobs = visible_jobs(dataset, moment)
        if not jobs:
            continue
        port_gate, _ = _port_location(dataset, moment)
        from .state import effective_moves

        moves = [
            move
            for move in effective_moves(dataset, moment)
            if move.ready_at > moment
        ]
        result = _simulate(dataset, moment, shift_end, jobs, port_gate, moves)
        day_errors = 0
        for job in jobs:
            actual = next(
                (
                    signal.clock
                    for signal in dataset.signals_for(job.trip)
                    if signal.kind == DONE
                ),
                None,
            )
            predicted = result.done.get(job.trip)
            if actual is None or predicted is None or actual > shift_end:
                continue
            errors.append((predicted - actual).total_seconds() / 60)
            day_errors += 1
        if day_errors:
            used_days += 1
    if len(errors) < MIN_BACKTEST_TRIPS:
        return None, (
            f"可用于回放的历史趟次只有 {len(errors)} 趟"
            f"（少于 {MIN_BACKTEST_TRIPS} 趟）；模型误差无法估计，"
            "所有改派差异一律视为不可判定"
        )
    mae = statistics.mean(abs(value) for value in errors)
    bias = statistics.mean(errors)
    return mae, (
        f"用同一模型从相同时点回放前 {used_days} 个完整日："
        f"{len(errors)} 趟可比对，完成时刻平均绝对误差 {mae:.0f} 分钟，"
        f"平均偏差 {bias:+.0f} 分钟；改善小于该误差时视为不可区分"
    )


def compare_options(
    dataset: Dataset,
    as_of: datetime,
    shift_end: datetime | None = None,
    *,
    improvement_margin_minutes: float = DEFAULT_IMPROVEMENT_MARGIN_MINUTES,
) -> Comparison:
    shift_end = shift_end or dataset.policy.shift_end.replace(
        year=as_of.year, month=as_of.month, day=as_of.day
    )
    shift_closed = as_of >= shift_end
    from .state import effective_moves

    jobs = visible_jobs(dataset, as_of)
    all_moves = effective_moves(dataset, as_of)
    port_gate, port_free = _port_location(dataset, as_of, all_moves)
    inherited = [move for move in _future_moves(dataset, as_of, all_moves) if move.confirmed]
    suggested = [move for move in _future_moves(dataset, as_of, all_moves) if not move.confirmed]

    baseline_moves = [*inherited, *suggested]
    baseline = _simulate(dataset, as_of, shift_end, jobs, port_gate, baseline_moves)
    baseline_median = baseline.metrics.get("median_wait")
    options: list[Option] = [
        Option(
            action=ACTION_HOLD,
            line="PORT",
            gate=port_gate,
            start_at=None,
            ready_at=None,
            feasible=True,
            rationale=(
                f"机动线留在{port_gate or '原处'}，不新增转门；"
                + (
                    "已确认的调令按原计划执行"
                    if inherited
                    else "维持当前安排"
                )
            ),
            metrics={**baseline.metrics, "median_improvement": 0.0},
        )
    ]

    if shift_closed:
        manual = Option(
            action=ACTION_MANUAL,
            line="PORT",
            gate=None,
            start_at=None,
            ready_at=None,
            feasible=True,
            rationale=(
                f"{as_of:%H:%M} 已过本班结束 {shift_end:%H:%M}，不再新开转门；"
                "把当前可见状态交给下一班前先人工核对"
            ),
            metrics={**baseline.metrics, "median_improvement": 0.0},
            manual_checks=_manual_checks(dataset, as_of),
        )
        backtest, note = _backtest_day(
            dataset, shift_end.date().isoformat(), as_of
        )
        return Comparison(
            as_of=as_of,
            shift_end=shift_end,
            options=[*options, manual],
            recommendation=manual,
            backtest_error_minutes=backtest,
            backtest_note=note,
            inherited_moves=[move.notice for move in inherited],
            shift_closed=True,
        )

    blocked_by_confirmed = inherited
    for gate in dataset.resources.gates:
        if gate == port_gate:
            continue
        if blocked_by_confirmed:
            labels = "、".join(move.notice for move in blocked_by_confirmed)
            options.append(
                Option(
                    action=ACTION_MOVE,
                    line="PORT",
                    gate=gate,
                    start_at=None,
                    ready_at=None,
                    feasible=False,
                    rationale=f"{gate} 需等已确认调令 {labels} 执行后才可再评估",
                    warnings=["已确认调令优先，不能在同一时段占用机动线"],
                )
            )
            continue
        start_at = max(as_of, port_free)
        ready_at = start_at + timedelta(minutes=dataset.policy.move_minutes)
        if ready_at >= shift_end:
            options.append(
                Option(
                    action=ACTION_MOVE,
                    line="PORT",
                    gate=gate,
                    start_at=start_at,
                    ready_at=ready_at,
                    feasible=False,
                    rationale="转到该门时本班已结束",
                    warnings=["转门停机时间超出本班剩余时间"],
                )
            )
            continue
        proposal = Move(
            notice=f"proposal-{gate}",
            line="PORT",
            from_gate=port_gate or gate,
            to_gate=gate,
            start_at=start_at,
            ready_at=ready_at,
            state="建议",
        )
        # The new suggestion supersedes unstarted suggestions, never a
        # confirmed move and never a move already under way.
        kept_suggestions = [
            move
            for move in suggested
            if move.start_at < as_of
        ]
        result = _simulate(
            dataset,
            as_of,
            shift_end,
            jobs,
            port_gate,
            [*inherited, *kept_suggestions, proposal],
        )
        median = result.metrics.get("median_wait")
        improvement = (
            float(baseline_median) - float(median)
            if baseline_median is not None and median is not None
            else None
        )
        warnings: list[str] = []
        if result.metrics["unserved"] > baseline.metrics["unserved"]:
            warnings.append(
                "班次内未服务完的车增加：等待看起来缩短可能是把车挤出了台账视野"
            )
        if (
            result.metrics["served"] < baseline.metrics["served"]
            and (result.metrics["median_wait"] or 0) < (baseline_median or 0)
        ):
            warnings.append("完成车数减少而中位等待下降，不能据此判定改善")
        options.append(
            Option(
                action=ACTION_MOVE,
                line="PORT",
                gate=gate,
                start_at=start_at,
                ready_at=ready_at,
                feasible=True,
                rationale=(
                    f"把机动线从{port_gate}转到{gate}，"
                    f"{start_at:%H:%M} 出发、{ready_at:%H:%M} 可作业"
                ),
                warnings=warnings,
                metrics={**result.metrics, "median_improvement": improvement},
            )
        )

    manual = Option(
        action=ACTION_MANUAL,
        line="PORT",
        gate=None,
        start_at=None,
        ready_at=None,
        feasible=True,
        rationale=(
            "现有的可见记录还不足以把各改派方案的改善幅度和模型误差分开；"
            "本班不新增未开始的转门，已确认的调令继续执行，"
            "并按清单人工核对后再决定"
        ),
        metrics={**baseline.metrics, "median_improvement": 0.0},
        manual_checks=_manual_checks(dataset, as_of),
    )
    options.append(manual)
    backtest, note = _backtest_day(
            dataset, shift_end.date().isoformat(), as_of
        )
    margin = (
        max(improvement_margin_minutes, backtest)
        if backtest is not None
        else float("inf")
    )
    thin, thin_reasons = _evidence_is_thin(dataset, as_of, jobs)
    manual.rationale = (
        "可见台账不完整：" + "；".join(thin_reasons)
        + "。先人工核对，不新增未开始的转门；已确认的调令继续执行"
    ) if thin else (
        "维持现状与各改派方案的差异小于模型误差，"
        "本班不新增未开始的转门；按清单核对后再决定"
    )
    recommendation = _recommend(options, margin, thin)
    return Comparison(
        as_of=as_of,
        shift_end=shift_end,
        options=options,
        recommendation=recommendation,
        backtest_error_minutes=backtest,
        backtest_note=note,
        inherited_moves=[move.notice for move in inherited],
        superseded_moves=[move.notice for move in suggested],
    )


def _manual_checks(dataset: Dataset, as_of: datetime) -> list[str]:
    return [
        "逐个核对各门当前在场车号、是否已叫号、是否已进场",
        "核对共享称重口与跨门通道是否已经排队",
        "核对司机是否仍可选择其他门（以预约为准）",
        "核对未见到达记录的车是否在候车带断记区间内",
        "核对是否有迟到的离场记录尚未进入台账",
    ]


def _evidence_is_thin(dataset: Dataset, as_of: datetime, jobs: list[JobState]) -> tuple[bool, list[str]]:
    """Whether the visible ledger is too incomplete to prefer any new move."""
    from .queue import snapshot

    state = snapshot(dataset, as_of)
    reasons: list[str] = []
    if state.unobserved_share >= 0.30:
        reasons.append(
            f"完全未见记录的趟次占 {state.unobserved_share:.0%}，"
            "队列深度不可信"
        )
    unknown = [job for job in jobs if job.arrival_basis == "unconfirmed"]
    if unknown:
        reasons.append(f"{len(unknown)} 趟没有可确认的到达时刻")
    if len(jobs) < 3:
        reasons.append(f"当前可见趟次只有 {len(jobs)} 趟")
    return bool(reasons), reasons


def _recommend(
    options: list[Option], margin: float, thin: bool = False
) -> Option:
    hold = next(option for option in options if option.action == ACTION_HOLD)
    manual = next(option for option in options if option.action == ACTION_MANUAL)
    baseline_median = hold.metrics.get("median_wait")
    best: Option | None = None
    for option in options:
        if option.action != ACTION_MOVE or not option.feasible or option.warnings:
            continue
        improvement = option.improvement
        if improvement is None:
            continue
        if improvement <= margin:
            option.indistinguishable_from = "hold"
            continue
        if best is None or (improvement or 0) > (best.improvement or 0):
            best = option
    if best is not None:
        return best
    # Nothing beats holding by more than the model can resolve.  If the ledger
    # itself is incomplete, recommend a manual check rather than inventing a
    # move that only looks good on the visible subset.
    for option in options:
        if option.action == ACTION_MOVE and option.feasible and option.improvement is not None:
            option.indistinguishable_from = "hold"
    has_alternative = any(
        option.action == ACTION_MOVE and option.feasible and not option.warnings
        for option in options
    )
    if thin and has_alternative:
        return manual
    return hold
