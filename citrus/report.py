"""Markdown reports for the duty room.

Every number is accompanied by what it does not know: unresolved trips, arrival
intervals caused by known outages, and the model error that makes two options
indistinguishable.  Nothing is written back into ``data/``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from .diagnose import BOTTLENECK_CORRIDOR, BOTTLENECK_UNLOAD, BOTTLENECK_WEIGH, diagnose_all
from .model import Dataset
from .occupancy import (
    inside_periods,
    overlaps,
    peak_inside,
    position_occupancies,
    scale_occupancies,
)
from .plan import ACTION_HOLD, ACTION_MANUAL, ACTION_MOVE, Comparison, Option, compare_options
from .queue import Snapshot, replay_integrity, snapshot, warning_reasons
from .state import Decision, load_decisions
from .stats import ShiftSummary, by_gate_totals, naive_comparison, summarize_shift
from .timeline import (
    BASIS_OUTAGE,
    TripRecord,
    build_all,
    duplicate_summary,
    outage_triage,
)

REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"


def _fmt(value: float | None, suffix: str = "", digits: int = 0) -> str:
    if value is None:
        return "不可识别"
    return f"{value:.{digits}f}{suffix}"


def _interval(low: float | None, high: float | None, suffix: str = " 分钟") -> str:
    if low is None and high is None:
        return "无可比样本"
    if high is None:
        return f"≥ {_fmt(low)}{suffix}（上界不可识别）"
    if abs(high - low) < 0.5:
        return f"{_fmt(low)}{suffix}"
    return f"{_fmt(low)}–{_fmt(high)}{suffix}"


def records_for_day(dataset: Dataset, as_of: datetime, date: str) -> list[TripRecord]:
    records = build_all(dataset, as_of=as_of)
    return [
        record
        for record in records.values()
        if record.appointment.promised_complete_by.date().isoformat() == date
    ]


def data_quality_section(dataset: Dataset) -> list[str]:
    duplicates = duplicate_summary(dataset)
    triage = outage_triage(dataset)
    lines = [
        "## 资料来源与口径",
        "",
        f"- 趟次 {duplicates['trips']}，信号 {duplicates['signals']} 条；"
        f"重复候车登记 {duplicates['duplicate_wait_signals']} 条，"
        "按同一趟、同一门只计一次，不构成第二辆车。",
        f"- 无有效候车登记的趟次 {duplicates['trips_without_wait']} 趟：",
        f"  - 能在已知候车带断记区间内确认到达的 {len(triage['confirmed'])} 趟："
        + ("、".join(triage["confirmed"]) if triage["confirmed"] else "无")
        + "，按到达区间保留在样本内；",
        f"  - 无法确认的 {len(triage['unconfirmed'])} 趟："
        + ("、".join(triage["unconfirmed"]) if triage["unconfirmed"] else "无")
        + "，单列为“到达不明”，不当作零等待、也不悄悄剔除。",
        "- 来源时钟偏移按 policy.json 还原："
        + "，".join(
            f"{source} {offset:+d} 分钟"
            for source, offset in sorted(dataset.policy.clock_offset_minutes.items())
        )
        + "。",
        "- `arrived_at` 单独用作可见时刻；回放只使用当时已可见的记录。",
        "",
    ]
    return lines


def measure_section(
    summary: ShiftSummary, label: str, naive
) -> list[str]:
    lines = [
        f"### {label}",
        "",
        f"- 观测时点：{summary.as_of.isoformat()}",
        f"- 已卸完 {summary.served} 趟、放弃 {summary.abandoned} 趟、"
        f"仍在排队或场内 {summary.pending} 趟、到达不明 {summary.unknown_arrival} 趟；"
        f"放弃率 {summary.abandonment_rate:.0%}（以已结束趟次为分母）。",
        f"- **用户等待**（首次有效到达 → 卸完或放弃）："
        f"中位 {_interval(summary.user.median_low, summary.user.median_high)}，"
        f"P90 {_interval(summary.user.p90_low, summary.user.p90_high)}；"
        f"超过 {summary.warning_wait_minutes} 分钟的比例 "
        f"{_interval(summary.user.over_low * 100 if summary.user.over_low is not None else None, summary.user.over_high * 100 if summary.user.over_high is not None else None, '%')}",
        f"- **预约完成承诺**（按卸完时刻判定）：未达成比例 "
        f"{_interval(summary.promise.over_low * 100 if summary.promise.over_low is not None else None, summary.promise.over_high * 100 if summary.promise.over_high is not None else None, '%')}；"
        "尚未结束的趟次按待定处理，不记作达成。",
        "",
        "两种口径的差别：放弃的司机在“用户等待”里按实际离开时刻计入，"
        "在“预约完成承诺”里一律算未达成；若只看已进场的车，"
        "放弃者被移出分子分母，平均等待会变好看。",
        "",
        f"- 对照：仅统计已进门车辆的平均整趟等待 "
        f"{_fmt(naive.entered_only_mean, ' 分钟', 1)}（{naive.entered_count} 趟）；"
        f"只统计卸完车辆 {_fmt(naive.served_only_mean, ' 分钟', 1)}（{naive.served_count} 趟）；"
        f"包含放弃的全部已结束趟次 {_fmt(naive.all_terminal_mean, ' 分钟', 1)}；"
        f"放弃者自身平均 {_fmt(naive.abandoned_mean, ' 分钟', 1)}（{naive.abandoned_count} 趟）。",
        f"- 承诺兑现率 {_fmt(naive.promise_attainment * 100 if naive.promise_attainment is not None else None, '%')}。",
        "",
    ]
    return lines


def gate_table(records: list[TripRecord], as_of: datetime) -> list[str]:
    summaries = by_gate_totals(records, as_of)
    lines = [
        "按最终使用的门归集：改门的趟次整趟只算一次，不会在两个门各记一遍。",
        "下表同时给出“仅看已进门”和“含放弃”的平均整趟等待，"
        "两者差距就是掉头司机被移出样本后失真的幅度。",
        "",
        "| 门 | 已卸完 | 放弃 | 排队/场内 | 到达不明 | 用户等待中位 | 超过阈值比例 | 仅看已进门平均 | 含放弃平均 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for gate, summary in summaries.items():
        user = summary.user
        naive = naive_comparison(
            [record for record in records if record.final_gate == gate], as_of
        )
        lines.append(
            f"| {gate} | {summary.served} | {summary.abandoned} | {summary.pending} | "
            f"{summary.unknown_arrival} | "
            f"{_interval(user.median_low, user.median_high)} | "
            f"{_interval(user.over_low * 100 if user.over_low is not None else None, user.over_high * 100 if user.over_high is not None else None, '%')} | "
            f"{_fmt(naive.entered_only_mean, ' 分钟', 1)} | "
            f"{_fmt(naive.all_terminal_mean, ' 分钟', 1)} |"
        )
    lines.append("")
    changed = sum(1 for record in records if record.change is not None)
    if changed:
        lines.append(
            f"本班有 {changed} 趟改门；按预约门看会把它们的等待算到未最终使用的门上，"
            "因此上表按最终门归集。"
        )
        lines.append("")
    return lines


def observation_coverage(dataset: Dataset, as_of: datetime) -> list[str]:
    state = snapshot(dataset, as_of)
    day = as_of.date().isoformat()
    todays = [
        appointment
        for appointment in dataset.appointments.values()
        if appointment.promised_complete_by.date().isoformat() == day
    ]
    unseen = set(state.unseen)
    by_gate: dict[str, int] = {}
    for trip in unseen:
        gate = dataset.appointments[trip].requested_gate
        by_gate[gate] = by_gate.get(gate, 0) + 1
    intervals = [
        trip
        for trip in state.records
        if state.records[trip].arrival_basis == BASIS_OUTAGE
    ]
    lines = [
        "### 观测完整度",
        "",
        f"- 本班预约 {len(todays)} 趟；到 {as_of:%H:%M} 已见到记录 "
        f"{len(todays) - len(unseen)} 趟，完全未见 {len(unseen)} 趟"
        + (
            "（"
            + "，".join(f"{gate} {count} 趟" for gate, count in sorted(by_gate.items()))
            + "）"
            if by_gate
            else ""
        )
        + "。",
        f"- 未完整观测比例 {state.unobserved_share:.0%}："
        "未见记录既可能是还没出发，也可能是候车带漏记；在核对清楚前，"
        "任何“某门最堵”的比较都只代表台账里看得到的部分。",
    ]
    if intervals:
        lines.append(
            f"- 本时点仍在场内且到达只能确认在断记区间内的有 {len(intervals)} 趟："
            + "、".join(sorted(intervals))
            + "；它们按区间计入，不当作零等待。"
        )
    lines.append("")
    return lines


def queue_section(
    dataset: Dataset, state: Snapshot, as_of: datetime
) -> list[str]:
    lines = [
        "## 当前排队与预警",
        "",
        f"- 观测时点 {as_of.isoformat()}："
        f"排队 {state.total_queue} 趟，场内 {sum(state.inside.values())} 趟，"
        f"完全未见记录 {len(state.unseen)} 趟，到达不明 {len(state.unknown_arrival)} 趟；"
        f"未完整观测比例 {state.unobserved_share:.0%}。",
        "",
    ]
    reasons = warning_reasons(dataset, as_of)
    if not reasons:
        lines.append("当前没有越过等待阈值或预计错过承诺的趟次。")
        lines.append("")
        return lines
    lines.extend(
        [
            "| 门 | 趟次 | 阶段 | 已等 | 预计整趟 | 触发理由 | 当时可见依据 |",
            "| --- | --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for reason in reasons:
        evidence = reason.evidence[:3]
        evidence_text = "、".join(evidence) + ("…" if len(reason.evidence) > 3 else "")
        lines.append(
            f"| {reason.gate} | {reason.trip} | {reason.stage} | "
            f"{_fmt(reason.waited_minutes, ' 分钟')} | "
            f"{_fmt(reason.projected_wait_minutes, ' 分钟')} | "
            f"{reason.text} | {evidence_text} |"
        )
    lines.append("")
    return lines


def diagnosis_section(
    dataset: Dataset,
    records: list[TripRecord],
    as_of: datetime,
    start: datetime,
    end: datetime,
) -> list[str]:
    state = snapshot(dataset, as_of)
    queue_now = {gate: state.queue_depth(gate) for gate in state.gates}
    diagnoses = diagnose_all(
        records, dataset, start, min(end, as_of), queue_now=queue_now
    )
    binding_label = {
        BOTTLENECK_UNLOAD: "本门卸料",
        BOTTLENECK_WEIGH: "共享/下游称重",
        BOTTLENECK_CORRIDOR: "跨门通道",
        "none": "无明显容量瓶颈",
    }
    lines = [
        "## 拥堵来源诊断",
        "",
        "以下按最繁忙的连续 60 分钟窗口比较需求与容量；"
        "“看起来最堵”的门不一定受卸料位限制。",
        "",
        "| 门 | 诊断窗口 | 入口等待中位 | 入口等待 P90 | 卸料利用率 | 称重利用率 | 通道利用率 | 判定 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in diagnoses:
        unload = item.unload.utilisation if item.unload else None
        weigh = item.weigh.utilisation if item.weigh else None
        corridor = item.corridor.utilisation if item.corridor else None
        lines.append(
            f"| {item.gate} | {item.window_start:%H:%M}–{item.window_end:%H:%M} | "
            f"{_fmt(item.entry_wait_median, ' 分钟')} | {_fmt(item.entry_wait_p90, ' 分钟')} | "
            f"{_fmt(unload * 100 if unload is not None else None, '%')} | "
            f"{_fmt(weigh * 100 if weigh is not None else None, '%')} | "
            f"{_fmt(corridor * 100 if corridor is not None else None, '%')} | "
            f"{binding_label.get(item.binding, item.binding)} |"
        )
    lines.append("")
    for item in diagnoses:
        lines.append(f"- **{item.gate}**：{item.explanation}")
        for caveat in item.caveats:
            lines.append(f"  - 注意：{caveat}")
    lines.append("")
    return lines


def option_table(comparison: Comparison) -> list[str]:
    lines = [
        "## 机动线处置选项",
        "",
        f"- 视点 {comparison.as_of:%H:%M}，班次结束 {comparison.shift_end:%H:%M}。",
        f"- 模型可辨识度：{comparison.backtest_note}",
        f"- 已继承的确认调令："
        + ("、".join(comparison.inherited_moves) if comparison.inherited_moves else "无")
        + "；本次将替换的未开始建议："
        + ("、".join(comparison.superseded_moves) if comparison.superseded_moves else "无")
        + "。",
        "",
        "| 方案 | 门 | 出发/可作业 | 可行 | 班内服务 | 未服务 | 中位等待 | 改善 | 判定 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for option in comparison.options:
        metrics = option.metrics
        window = (
            f"{option.start_at:%H:%M}/{option.ready_at:%H:%M}"
            if option.start_at and option.ready_at
            else "—"
        )
        verdict = "推荐" if option is comparison.recommendation else ""
        if option.indistinguishable_from:
            verdict = "与维持现状不可区分"
        if not option.feasible:
            verdict = "不可行"
        if option.action == ACTION_MANUAL:
            verdict = "需人工核对" if option is comparison.recommendation else verdict
        lines.append(
            f"| {_action_label(option)} | {option.gate or '—'} | {window} | "
            f"{'是' if option.feasible else '否'} | "
            f"{metrics.get('served', '—')} | {metrics.get('unserved', '—')} | "
            f"{_fmt(metrics.get('median_wait'), ' 分钟')} | "
            f"{_fmt(option.improvement, ' 分钟')} | {verdict} |"
        )
    lines.append("")
    for option in comparison.options:
        if option.warnings:
            lines.append(f"- {_action_label(option)} {option.gate or ''}：" + "；".join(option.warnings))
    lines.append("")
    return lines


def _action_label(option: Option) -> str:
    if option.action == ACTION_HOLD:
        return "维持现状"
    if option.action == ACTION_MOVE:
        return "改派机动线"
    return "人工核对"


def decision_section(comparison: Comparison, decisions: list[Decision]) -> list[str]:
    recommendation = comparison.recommendation
    lines = [
        "## 建议",
        "",
        f"{recommendation.rationale}。",
        "",
    ]
    if recommendation.manual_checks:
        lines.append("人工核对清单：")
        for check in recommendation.manual_checks:
            lines.append(f"- {check}")
        lines.append("")
    if decisions:
        lines.append("已确认处置（后续推演从此处继承）：")
        lines.append("")
        lines.append("| 决定 | 时间 | 动作 | 门 | 出发/可作业 | 操作人 |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for decision in decisions:
            window = (
                f"{decision.start_at:%H:%M}/{decision.ready_at:%H:%M}"
                if decision.start_at and decision.ready_at
                else "—"
            )
            lines.append(
                f"| {decision.decision_id} | {decision.decided_at:%H:%M} | "
                f"{decision.action} | {decision.to_gate or '—'} | {window} | "
                f"{decision.operator} |"
            )
        lines.append("")
    return lines


def occupancy_detail(dataset: Dataset, records: list[TripRecord]) -> list[str]:
    """Per-gate occupancy intervals and overlap periods for the shift."""
    positions = position_occupancies(dataset, records)
    lines = [
        "### 卸料位占用时段",
        "",
        "| 门 | 车号 | 开始 | 结束 | 分钟 | 依据 |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for item in positions:
        lines.append(
            f"| {item.gate} | {item.trip} | {item.start:%H:%M} | {item.end:%H:%M} | "
            f"{item.minutes:.0f} | {'、'.join(item.evidence)} |"
        )
    lines.append("")
    return lines


def replay_section(dataset: Dataset, as_of: datetime) -> list[str]:
    observations = [
        moment for moment in dataset.policy.observe_at if moment <= as_of
    ]
    lines = ["## 回放一致性", ""]
    if len(observations) < 2:
        lines.append("只有一个观测时点，无法比较。")
        lines.append("")
        return lines
    for earlier, later in zip(observations, observations[1:]):
        problems = replay_integrity(dataset, earlier, later)
        lines.append(f"- {earlier:%H:%M} → {later:%H:%M}：")
        if not problems:
            lines.append("  - 没有需要靠后来记录才能成立的预警。")
        for problem in problems:
            lines.append(f"  - {problem}")
    lines.append("")
    return lines


def occupancy_section(dataset: Dataset, records: list[TripRecord]) -> list[str]:
    positions = position_occupancies(dataset, records)
    scales = scale_occupancies(dataset, records)
    inside = inside_periods(records)
    position_overlaps = overlaps(positions, 1)
    scale_overlaps = overlaps(scales, 1)
    lines = [
        "## 门位与通道占用",
        "",
        "门位占用以每次“卸完”倒推该门、该车种的标准卸料分钟得到；"
        "时段保留为 [开始, 结束)，便于查哪一段被占用。",
        "",
        "- 场内同时在门内（进门到卸完/放弃）峰值："
        + (
            "，".join(
                f"{gate} {count} 辆"
                for gate, count in sorted(peak_inside(inside).items())
            )
            if inside
            else "该时点尚无已完成进出的记录"
        )
        + "。",
        f"- 按记录完成的卸料位占用：{len(positions)} 段，"
        f"合计 {sum(item.minutes for item in positions):.0f} 分钟；"
        f"其中相邻两车占用时段重叠 {len(position_overlaps)} 段，"
        f"合计 {sum(item.minutes for item in position_overlaps):.0f} 分钟。"
        "重叠按记录原样保留，不做平滑；它们正是需要人工核对的时段。",
        f"- 称重位共推定占用 {len(scales)} 段，占用时段重叠 {len(scale_overlaps)} 段，"
        f"合计 {sum(item.minutes for item in scale_overlaps):.0f} 分钟；"
        "称重占用由标准分钟数倒推，仅作线索。",
        "",
    ]
    if position_overlaps:
        worst = sorted(position_overlaps, key=lambda item: -item.minutes)[:5]
        lines.append("最长的卸料位重叠：")
        for item in worst:
            lines.append(
                f"- {item.gate} {item.start:%H:%M}–{item.end:%H:%M}"
                f"（{item.minutes:.0f} 分钟，{ '、'.join(item.trips) }）"
            )
        lines.append("")
    return lines


def build_report(
    dataset: Dataset,
    as_of: datetime | None = None,
    *,
    decisions: list[Decision] | None = None,
    day: str | None = None,
) -> str:
    as_of = as_of or dataset.policy.observe_at[0]
    day = day or as_of.date().isoformat()
    records = records_for_day(dataset, as_of, day)
    summary = summarize_shift(records, as_of, dataset.policy.warning_wait_minutes)
    naive = naive_comparison(records, as_of)
    state = snapshot(dataset, as_of)
    comparison = compare_options(dataset, as_of)
    decisions = decisions if decisions is not None else load_decisions()
    shift_start = as_of.replace(hour=5, minute=0, second=0, microsecond=0)
    shift_end = dataset.policy.shift_end.replace(
        year=as_of.year, month=as_of.month, day=as_of.day
    )
    lines = [
        f"# 值班分析：{day} {as_of:%H:%M}（观测时点）",
        "",
        "本报告只读 data/，不改动原始记录；所有结论都标注了证据不足之处。",
        "",
        *data_quality_section(dataset),
        "## 等待与承诺",
        "",
        *measure_section(summary, "全部门合计", naive),
        *gate_table(records, as_of),
        *observation_coverage(dataset, as_of),
        *queue_section(dataset, state, as_of),
        *diagnosis_section(dataset, records, as_of, shift_start, shift_end),
        *option_table(comparison),
        *decision_section(comparison, decisions),
        *occupancy_section(dataset, records),
        *occupancy_detail(dataset, records),
        *replay_section(dataset, as_of),
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_report(text: str, path: Path | None = None) -> Path:
    path = path or REPORT_DIR / "duty_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
