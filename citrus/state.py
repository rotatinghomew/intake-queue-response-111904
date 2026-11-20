"""Operator decisions as durable state, plus replay at a later observation.

A confirmed decision becomes the starting point for every later projection:
its line, gates, start and ready times are inherited exactly, and the option
engine must not count the same mobile line at two gates at once.  When a new
gate-change record arrives mid-shift, unstarted periods may be re-planned;
a move already under way or a gate position already occupied is frozen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .model import Dataset, Move, parse_time

STATE_DIR = Path(__file__).resolve().parents[1] / "state"
DECISIONS_FILE = STATE_DIR / "decisions.json"

STATE_CONFIRMED = "已确认"
STATE_PROPOSED = "建议"


@dataclass
class Decision:
    decision_id: str
    decided_at: datetime
    action: str
    line: str
    from_gate: str | None
    to_gate: str | None
    start_at: datetime | None
    ready_at: datetime | None
    state: str
    operator: str
    note: str
    based_on_as_of: datetime | None = None
    based_on_options: list[str] = field(default_factory=list)

    def to_move(self) -> Move | None:
        if self.action != "move" or self.to_gate is None:
            return None
        assert self.start_at is not None and self.ready_at is not None
        return Move(
            notice=self.decision_id,
            line=self.line,
            from_gate=self.from_gate or "",
            to_gate=self.to_gate,
            start_at=self.start_at,
            ready_at=self.ready_at,
            state=self.state,
            confirmed_at=self.decided_at if self.state == STATE_CONFIRMED else None,
            source="decision",
        )

    def to_json(self) -> dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "decided_at": self.decided_at.isoformat(),
            "action": self.action,
            "line": self.line,
            "from_gate": self.from_gate,
            "to_gate": self.to_gate,
            "start_at": self.start_at.isoformat() if self.start_at else None,
            "ready_at": self.ready_at.isoformat() if self.ready_at else None,
            "state": self.state,
            "operator": self.operator,
            "note": self.note,
            "based_on_as_of": (
                self.based_on_as_of.isoformat() if self.based_on_as_of else None
            ),
            "based_on_options": list(self.based_on_options),
        }

    @classmethod
    def from_json(cls, raw: dict[str, object]) -> "Decision":
        return cls(
            decision_id=str(raw["decision_id"]),
            decided_at=parse_time(str(raw["decided_at"])),
            action=str(raw["action"]),
            line=str(raw["line"]),
            from_gate=raw.get("from_gate"),
            to_gate=raw.get("to_gate"),
            start_at=parse_time(raw["start_at"]) if raw.get("start_at") else None,
            ready_at=parse_time(raw["ready_at"]) if raw.get("ready_at") else None,
            state=str(raw["state"]),
            operator=str(raw.get("operator", "")),
            note=str(raw.get("note", "")),
            based_on_as_of=(
                parse_time(str(raw["based_on_as_of"]))
                if raw.get("based_on_as_of")
                else None
            ),
            based_on_options=list(raw.get("based_on_options") or []),
        )


def load_decisions(path: Path | None = None) -> list[Decision]:
    path = path or DECISIONS_FILE
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Decision.from_json(item) for item in raw.get("decisions", [])]


def save_decision(decision: Decision, path: Path | None = None) -> Path:
    path = path or DECISIONS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_decisions(path)
    updated = [
        item for item in existing if item.decision_id != decision.decision_id
    ]
    updated.append(decision)
    updated.sort(key=lambda item: (item.decided_at, item.decision_id))
    path.write_text(
        json.dumps(
            {"decisions": [item.to_json() for item in updated]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def effective_moves(
    dataset: Dataset, as_of: datetime, decisions: Iterable[Decision] | None = None
) -> list[Move]:
    """Source moves plus confirmed operator decisions visible by ``as_of``.

    A confirmed decision that supersedes a suggested source move removes that
    suggestion; the same mobile line is never counted at two gates at once.
    """
    decisions = [
        decision
        for decision in (decisions if decisions is not None else load_decisions())
        if decision.state == STATE_CONFIRMED and decision.decided_at <= as_of
    ]
    decision_moves = [
        move for move in (decision.to_move() for decision in decisions) if move
    ]
    superseded: set[str] = set()
    for decision in decisions:
        for notice in decision.based_on_options:
            superseded.add(notice)
    moves: list[Move] = []
    for move in dataset.moves:
        if move.notice in superseded and not move.confirmed:
            continue
        moves.append(move)
    for move in decision_moves:
        moves.append(move)
    return sorted(moves, key=lambda item: (item.start_at, item.notice))


def decisions_at(
    decisions: Iterable[Decision], as_of: datetime
) -> list[Decision]:
    return [decision for decision in decisions if decision.decided_at <= as_of]
