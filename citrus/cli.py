"""Command line entry point for the duty-room analysis."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from .model import SourceError, load_dataset, parse_time
from .plan import ACTION_HOLD, ACTION_MANUAL, ACTION_MOVE, compare_options
from .queue import evidence_lines, locate
from .report import build_report, write_report
from .state import (
    STATE_CONFIRMED,
    Decision,
    save_decision,
)
from .timeline import outage_triage


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="citrus",
        description="值班室分析：等待口径、预警、拥堵来源与机动线处置",
    )
    parser.add_argument(
        "--as-of",
        help="回放观测时点，ISO 时间，默认第一个 observe_at",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    report = sub.add_parser("report", help="生成 Markdown 值班报告")
    report.add_argument("--out", type=Path, help="输出文件")

    summary = sub.add_parser("summary", help="打印等待与承诺摘要")
    summary.add_argument("--day", help="班次日期 YYYY-MM-DD")

    warnings = sub.add_parser("warnings", help="列出当前排队与预警")
    warnings.add_argument("--trip", help="只看某一趟")
    warnings.add_argument("--gate", help="只看某一门")
    warnings.add_argument("--all", action="store_true", help="包括尚未越阈值的趟次")

    options = sub.add_parser("options", help="比较机动线处置选项")
    options.add_argument("--json", action="store_true", help="输出 JSON")

    quality = sub.add_parser("quality", help="查看重复登记与断记确认情况")

    confirm = sub.add_parser("confirm", help="确认一项处置，成为后续推演起点")
    confirm.add_argument("--action", choices=[ACTION_HOLD, ACTION_MOVE, ACTION_MANUAL], required=True)
    confirm.add_argument("--to-gate")
    confirm.add_argument("--from-gate")
    confirm.add_argument("--start-at")
    confirm.add_argument("--ready-at")
    confirm.add_argument("--operator", default="值班员")
    confirm.add_argument("--note", default="")
    confirm.add_argument("--supersede", action="append", default=[])

    return parser


def _as_of(dataset, value: str | None) -> datetime:
    if value is None:
        return dataset.policy.observe_at[0]
    return parse_time(value)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        dataset = load_dataset()
        as_of = _as_of(dataset, args.as_of)
        if args.command == "report":
            text = build_report(dataset, as_of)
            path = write_report(text, args.out)
            print(f"已写入 {path}")
            return 0
        if args.command == "summary":
            text = build_report(dataset, as_of)
            start = text.index("## 等待与承诺")
            end = text.index("## 当前排队与预警")
            print(text[start:end].rstrip())
            return 0
        if args.command == "warnings":
            items = locate(dataset, as_of, trip=args.trip, gate=args.gate)
            if not args.all:
                items = [item for item in items if item.warning]
            for item in items:
                flag = "预警" if item.warning else "正常"
                waited = f"{item.waited_minutes:.0f} 分钟" if item.waited_minutes is not None else "到达不明"
                print(
                    f"{item.gate} {item.trip} {item.stage} {waited} {flag}"
                )
                for line in evidence_lines(dataset, item.evidence):
                    print(f"    {line}")
            return 0
        if args.command == "options":
            comparison = compare_options(dataset, as_of)
            if args.json:
                print(
                    json.dumps(
                        {
                            "as_of": comparison.as_of.isoformat(),
                            "shift_end": comparison.shift_end.isoformat(),
                            "shift_closed": comparison.shift_closed,
                            "backtest_error_minutes": comparison.backtest_error_minutes,
                            "backtest_note": comparison.backtest_note,
                            "inherited_moves": comparison.inherited_moves,
                            "superseded_moves": comparison.superseded_moves,
                            "recommendation": {
                                "action": comparison.recommendation.action,
                                "gate": comparison.recommendation.gate,
                                "rationale": comparison.recommendation.rationale,
                                "manual_checks": comparison.recommendation.manual_checks,
                            },
                            "options": [
                                {
                                    "action": option.action,
                                    "gate": option.gate,
                                    "feasible": option.feasible,
                                    "rationale": option.rationale,
                                    "warnings": option.warnings,
                                    "indistinguishable_from": option.indistinguishable_from,
                                    "metrics": option.metrics,
                                }
                                for option in comparison.options
                            ],
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return 0
            print(f"视点 {comparison.as_of:%H:%M}，班次结束 {comparison.shift_end:%H:%M}")
            print(comparison.backtest_note)
            for option in comparison.options:
                med = option.metrics.get("median_wait")
                print(
                    f"- {option.action:6} {str(option.gate or '—'):3} "
                    f"可行={option.feasible} 中位等待={med if med is not None else '不可识别'} "
                    f"{'；'.join(option.warnings)}"
                    + ("  [推荐]" if option is comparison.recommendation else "")
                    + (
                        "  [与维持现状不可区分]"
                        if option.indistinguishable_from
                        else ""
                    )
                )
            print(f"建议：{comparison.recommendation.rationale}")
            return 0
        if args.command == "quality":
            triage = outage_triage(dataset)
            print(f"可在断记区间确认到达：{len(triage['confirmed'])} 趟")
            for key, trips in triage["outages"].items():
                print(f"  {key}: {'、'.join(trips)}")
            print(f"无法确认到达：{len(triage['unconfirmed'])} 趟")
            for trip in triage["unconfirmed"]:
                print(f"  {trip}")
            return 0
        if args.command == "confirm":
            if args.action == ACTION_MOVE:
                if not (args.to_gate and args.start_at and args.ready_at):
                    parser.error("改派需要 --to-gate、--start-at、--ready-at")
            decision = Decision(
                decision_id=f"D{datetime.now():%Y%m%d%H%M%S}",
                decided_at=as_of,
                action=args.action,
                line="PORT",
                from_gate=args.from_gate,
                to_gate=args.to_gate,
                start_at=parse_time(args.start_at) if args.start_at else None,
                ready_at=parse_time(args.ready_at) if args.ready_at else None,
                state=STATE_CONFIRMED,
                operator=args.operator,
                note=args.note,
                based_on_as_of=as_of,
                based_on_options=args.supersede,
            )
            path = save_decision(decision)
            print(f"已确认 {decision.decision_id}，写入 {path}")
            return 0
        parser.error("unknown command")
    except SourceError as error:
        print(f"资料错误：{error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
