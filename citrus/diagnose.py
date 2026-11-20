"""Congestion source diagnosis: is the bottleneck unloading or downstream?

A gate that looks busiest at the entry may actually be limited by a shared
weighing station or the cross-gate corridor.  The duty room needs to know
whether giving it more unloading capacity could help at all, so this module
computes, over the busiest rolling window:

* entry waiting at each gate (arrival -> gate entry);
* unloading demand versus unloading capacity per gate;
* shared weighing demand versus capacity, counting *all* gates that share it;
* cross-gate corridor demand versus capacity.

The binding constraint is reported together with the evidence window, so a
"busiest" gate is not automatically an "unloading-limited" gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable

from .model import CHANGE, DONE, ENTRY, WEIGH, Dataset
from .timeline import TripRecord

BOTTLENECK_UNLOAD = "unload"
BOTTLENECK_WEIGH = "weigh"
BOTTLENECK_CORRIDOR = "corridor"
BOTTLENECK_NONE = "none"

DEFAULT_WINDOW_MINUTES = 60
MIN_DIAGNOSIS_TRIPS = 3


@dataclass
class ResourceLoad:
    resource: str
    label: str
    gates: tuple[str, ...]
    minutes: float
    capacity_minutes: float
    trips: int

    @property
    def utilisation(self) -> float | None:
        if self.capacity_minutes <= 0:
            return None
        return self.minutes / self.capacity_minutes


@dataclass
class WindowLoads:
    start: datetime
    end: datetime
    unload: dict[str, ResourceLoad] = field(default_factory=dict)
    weigh: dict[str, ResourceLoad] = field(default_factory=dict)
    corridor: ResourceLoad | None = None


@dataclass
class GateDiagnosis:
    gate: str
    window_start: datetime
    window_end: datetime
    entry_wait_median: float | None
    entry_wait_p90: float | None
    entry_wait_high: float | None
    queue_now: int
    unload: ResourceLoad | None
    weigh: ResourceLoad | None
    corridor: ResourceLoad | None
    binding: str
    explanation: str
    sample_trips: int = 0
    caveats: list[str] = field(default_factory=list)
    evidence: tuple[str, ...] = ()

    def loads(self) -> list[ResourceLoad]:
        items = [load for load in (self.unload, self.weigh, self.corridor) if load]
        return items


def _marker(record: TripRecord) -> datetime | None:
    if record.entry is not None:
        return record.entry.clock
    if record.arrival_lo is not None:
        return record.arrival_lo
    return None


def _window_loads(
    records: list[TripRecord], dataset: Dataset, start: datetime, end: datetime
) -> WindowLoads:
    minutes = (end - start).total_seconds() / 60
    resources = dataset.resources
    loads = WindowLoads(start=start, end=end)
    for record in records:
        marker = _marker(record)
        if marker is None or not (start <= marker <= end):
            continue
        gate = record.final_gate
        if gate not in resources.unload:
            continue
        class_min = record.vehicle_class
        unload = loads.unload.setdefault(
            gate,
            ResourceLoad(
                f"UN:{gate}",
                f"{gate}卸料位",
                (gate,),
                0.0,
                minutes,
                0,
            ),
        )
        unload.minutes += resources.unload[gate].minutes(class_min)
        unload.trips += 1
        station_name = resources.scale_for_gate[gate]
        station = resources.scale[station_name]
        weigh = loads.weigh.setdefault(
            station_name,
            ResourceLoad(
                f"SC:{station_name}",
                f"{station.kind}{station_name}",
                (),
                0.0,
                minutes,
                0,
            ),
        )
        weigh.minutes += station.minutes(class_min)
        weigh.trips += 1
        weigh.gates = tuple(sorted({*weigh.gates, gate}))
        if record.change is not None:
            corr = resources.corridor
            if loads.corridor is None:
                loads.corridor = ResourceLoad(
                    corr.station, "跨门通道", (gate,), 0.0, minutes, 0
                )
            loads.corridor.minutes += corr.minutes(class_min)
            loads.corridor.trips += 1
            loads.corridor.gates = tuple(
                sorted({*loads.corridor.gates, gate})
            )
    return loads


def _window_records(
    records: Iterable[TripRecord], start: datetime, end: datetime
) -> list[TripRecord]:
    selected = []
    for record in records:
        marker = _marker(record)
        if marker is not None and start <= marker <= end:
            selected.append(record)
    return selected


def _peak_window(
    records: list[TripRecord],
    dataset: Dataset,
    gates: Iterable[str],
    start: datetime,
    end: datetime,
    window_minutes: int,
) -> tuple[datetime, datetime, float]:
    step = timedelta(minutes=5)
    span = timedelta(minutes=window_minutes)
    best = (start, min(start + span, end), -1.0)
    cursor = start
    while cursor < end:
        window_end = min(cursor + span, end)
        loads = _window_loads(records, dataset, cursor, window_end)
        pressure = 0.0
        for load in [*loads.unload.values(), *loads.weigh.values()]:
            pressure = max(pressure, load.utilisation or 0.0)
        if loads.corridor is not None:
            pressure = max(pressure, loads.corridor.utilisation or 0.0)
        if pressure > best[2]:
            best = (cursor, window_end, pressure)
        cursor += step
    return best


def diagnose_gate(
    records: Iterable[TripRecord],
    dataset: Dataset,
    gate: str,
    start: datetime,
    end: datetime,
    window_minutes: int = DEFAULT_WINDOW_MINUTES,
    queue_now: int = 0,
) -> GateDiagnosis:
    records = list(records)
    win_start, win_end, _ = _peak_window(
        records, dataset, [gate], start, end, window_minutes
    )
    selected = _window_records(records, win_start, win_end)
    gate_records = [record for record in selected if record.final_gate == gate]
    entry_waits = sorted(
        max(0.0, (record.entry.clock - record.arrival_hi).total_seconds() / 60)
        for record in gate_records
        if record.entry is not None and record.arrival_hi is not None
    )
    from .stats import _quantile

    loads = _window_loads(selected, dataset, win_start, win_end)
    unload = loads.unload.get(gate)
    weigh = loads.weigh.get(dataset.resources.scale_for_gate[gate])
    corridor = loads.corridor
    candidates = [
        (BOTTLENECK_UNLOAD, unload.utilisation if unload else None),
        (BOTTLENECK_WEIGH, weigh.utilisation if weigh else None),
        (BOTTLENECK_CORRIDOR, corridor.utilisation if corridor else None),
    ]
    binding, best = BOTTLENECK_NONE, None
    for name, value in candidates:
        if value is not None and (best is None or value > best):
            binding, best = name, value
    caveats: list[str] = []
    if len(gate_records) < MIN_DIAGNOSIS_TRIPS:
        binding = BOTTLENECK_NONE
        explanation = (
            f"{gate} 在 {win_start:%H:%M}-{win_end:%H:%M} 只有 {len(gate_records)} 趟"
            f"可见作业记录（少于 {MIN_DIAGNOSIS_TRIPS} 趟），无法把拥堵归因到卸料位、"
            "称重或通道；任何改派都应先人工核对"
        )
        caveats.append("样本不足，不给出瓶颈结论")
    elif best is not None and best < 0.65:
        binding = BOTTLENECK_NONE
        explanation = (
            f"{gate} 在 {win_start:%H:%M}-{win_end:%H:%M} 各服务位利用率都不高"
            f"（最高 {best:.0%}）；看起来最长队更可能来自放行节奏或叫号间隔，"
            "加机动线大概率无效"
        )
    elif binding == BOTTLENECK_WEIGH:
        explanation = (
            f"{gate} 的拥堵来自{weigh.label}（利用率 {best:.0%}），"
            f"该称重同时服务 {'、'.join(weigh.gates)}；加快本门卸料不会解决排队"
        )
    elif binding == BOTTLENECK_CORRIDOR:
        explanation = (
            f"{gate} 的拥堵来自跨门通道（利用率 {best:.0%}）；"
            "再改派更多车只会把队堵在通道上"
        )
    elif binding == BOTTLENECK_UNLOAD and best is not None:
        explanation = (
            f"{gate} 的拥堵来自本门卸料位（利用率 {best:.0%}），"
            "增加卸料能力有机会缩短等待"
        )
    else:
        explanation = (
            f"{gate} 在 {win_start:%H:%M}-{win_end:%H:%M} 没有可比的容量记录，"
            "无法判定瓶颈；应先人工核对"
        )
    if weigh is not None and (weigh.utilisation or 0) >= 0.85:
        caveats.append(
            f"{weigh.label}已达 {(weigh.utilisation or 0):.0%}，"
            "任何改门方案都要先核对称重排队"
        )
    if corridor is not None and (corridor.utilisation or 0) >= 0.75:
        caveats.append("跨门通道接近饱和，转门带来的增量等待必须计入")
    if unload is not None and unload.trips < 3:
        caveats.append("窗口内样本少于 3 车，利用率不稳，只能作为线索")
    if not entry_waits:
        caveats.append("该窗口没有可见的进门记录，入口等待无法估计")
    evidence = tuple(
        signal.event
        for record in gate_records
        for signal in record.signals
        if signal.kind in {ENTRY, WEIGH, DONE, CHANGE}
    )
    return GateDiagnosis(
        gate=gate,
        window_start=win_start,
        window_end=win_end,
        entry_wait_median=_quantile(entry_waits, 0.5) if entry_waits else None,
        entry_wait_p90=_quantile(entry_waits, 0.9) if entry_waits else None,
        entry_wait_high=entry_waits[-1] if entry_waits else None,
        queue_now=queue_now,
        unload=unload,
        weigh=weigh,
        corridor=corridor,
        binding=binding,
        explanation=explanation,
        sample_trips=len(gate_records),
        caveats=caveats,
        evidence=evidence,
    )


def diagnose_all(
    records: Iterable[TripRecord],
    dataset: Dataset,
    start: datetime,
    end: datetime,
    queue_now: dict[str, int] | None = None,
    window_minutes: int = DEFAULT_WINDOW_MINUTES,
) -> list[GateDiagnosis]:
    records = list(records)
    queue_now = queue_now or {}
    gates = sorted(
        {
            record.final_gate
            for record in records
            if record.arrival_lo is not None
        }
    )
    return [
        diagnose_gate(
            records,
            dataset,
            gate,
            start,
            end,
            window_minutes=window_minutes,
            queue_now=queue_now.get(gate, 0),
        )
        for gate in gates
    ]
