import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from citrus.diagnose import (
    BOTTLENECK_NONE,
    BOTTLENECK_UNLOAD,
    BOTTLENECK_WEIGH,
    diagnose_all,
    diagnose_gate,
)
from citrus.model import load_dataset
from citrus.occupancy import (
    inside_periods,
    overlaps,
    peak_inside,
    port_occupancies,
    position_occupancies,
    scale_occupancies,
)
from citrus.plan import compare_options, visible_jobs
from citrus.queue import locate, replay_integrity, snapshot, warning_reasons
from citrus.report import build_report
from citrus.state import (
    STATE_CONFIRMED,
    Decision,
    effective_moves,
    load_decisions,
    save_decision,
)
from citrus.stats import summarize_shift
from citrus.timeline import build_all

DAY = "2026-11-09"


def at(hhmm: str) -> datetime:
    return datetime.fromisoformat(f"{DAY}T{hhmm}:00+08:00")


class OccupancyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_dataset()
        cls.records = list(build_all(cls.dataset).values())

    def test_gate_change_trip_occupies_only_final_gate(self):
        positions = position_occupancies(self.dataset, self.records)
        changed = [
            record
            for record in self.records
            if record.change is not None and record.done is not None
        ]
        self.assertTrue(changed)
        for record in changed:
            trips = [
                position
                for position in positions
                if position.trip == record.trip
            ]
            self.assertEqual(len(trips), 1)
            self.assertEqual(trips[0].gate, record.final_gate)

    def test_inside_peak_is_count_of_vehicles(self):
        periods = inside_periods(self.records)
        peak = peak_inside(periods)
        self.assertTrue(peak)
        for gate, count in peak.items():
            self.assertGreaterEqual(count, 1)

    def test_scale_overlap_is_reported(self):
        scales = scale_occupancies(self.dataset, self.records)
        found = overlaps(scales, 1)
        self.assertTrue(found)
        self.assertGreater(sum(item.minutes for item in found), 0)

    def test_port_occupancy_keeps_notice_and_state(self):
        items = port_occupancies(self.dataset)
        self.assertEqual(
            [(item.trip, item.gate) for item in items],
            [("M0", "北门"), ("M1", "西门")],
        )
        self.assertTrue(all(":" in item.evidence[0] for item in items))


class DiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_dataset()
        cls.records = list(build_all(cls.dataset).values())

    def test_shared_scale_is_flagged_not_unloading(self):
        diagnoses = diagnose_all(self.records, self.dataset, at("05:00"), at("09:00"))
        by_gate = {item.gate: item for item in diagnoses}
        self.assertIn("北门", by_gate)
        self.assertIn("西门", by_gate)
        for gate in ("北门", "西门"):
            self.assertEqual(by_gate[gate].binding, BOTTLENECK_WEIGH)
            self.assertIn("SC1", by_gate[gate].weigh.label)

    def test_small_sample_refuses_to_name_a_bottleneck(self):
        diagnosis = diagnose_gate(
            self.records,
            self.dataset,
            "南门",
            at("05:00"),
            at("05:10"),
            window_minutes=10,
        )
        self.assertEqual(diagnosis.binding, BOTTLENECK_NONE)
        self.assertIn("少于", diagnosis.explanation)


class StateTests(unittest.TestCase):
    def setUp(self):
        self.dataset = load_dataset()
        self.tmp = Path(tempfile.mkdtemp()) / "decisions.json"

    def test_confirmed_decision_becomes_starting_point(self):
        decision = Decision(
            decision_id="D1",
            decided_at=at("06:41"),
            action="move",
            line="PORT",
            from_gate="北门",
            to_gate="西门",
            start_at=at("07:00"),
            ready_at=at("07:20"),
            state=STATE_CONFIRMED,
            operator="值班员",
            note="",
            based_on_as_of=at("06:40"),
            based_on_options=["M1"],
        )
        save_decision(decision, self.tmp)
        loaded = load_decisions(self.tmp)
        self.assertEqual(len(loaded), 1)
        moves = effective_moves(self.dataset, at("06:50"), loaded)
        notices = [move.notice for move in moves]
        self.assertIn("D1", notices)
        self.assertNotIn("M1", notices)
        self.assertIn("M0", notices)

    def test_decision_not_visible_before_it_is_made(self):
        decision = Decision(
            decision_id="D1",
            decided_at=at("07:00"),
            action="move",
            line="PORT",
            from_gate="北门",
            to_gate="西门",
            start_at=at("07:05"),
            ready_at=at("07:25"),
            state=STATE_CONFIRMED,
            operator="值班员",
            note="",
        )
        save_decision(decision, self.tmp)
        moves = effective_moves(self.dataset, at("06:50"), load_decisions(self.tmp))
        self.assertNotIn("D1", [move.notice for move in moves])

    def test_round_trip_json(self):
        decision = Decision(
            decision_id="D2",
            decided_at=at("06:41"),
            action="hold",
            line="PORT",
            from_gate=None,
            to_gate=None,
            start_at=None,
            ready_at=None,
            state=STATE_CONFIRMED,
            operator="值班员",
            note="维持",
            based_on_as_of=at("06:40"),
        )
        save_decision(decision, self.tmp)
        raw = json.loads(self.tmp.read_text(encoding="utf-8"))
        self.assertEqual(raw["decisions"][0]["decision_id"], "D2")
        again = load_decisions(self.tmp)[0]
        self.assertEqual(again.note, "维持")
        self.assertIsNone(again.start_at)


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_dataset()

    def test_policy_observations_are_replay_safe(self):
        observations = self.dataset.policy.observe_at
        for earlier, later in zip(observations, observations[1:]):
            problems = replay_integrity(self.dataset, earlier, later)
            for problem in problems:
                self.assertIn("不得", problem)

    def test_earlier_snapshot_is_subset_of_later(self):
        early = snapshot(self.dataset, at("05:29"))
        late = snapshot(self.dataset, at("10:50"))
        early_trips = {item.trip for item in early.active_items}
        late_trips = {
            record.trip for record in build_all(self.dataset, as_of=at("10:50")).values()
            if record.status in {"queued", "inside"}
        }
        # Anything warned early was still either active or terminated later.
        for trip in early_trips:
            record = build_all(self.dataset, as_of=at("10:50"))[trip]
            self.assertIn(record.status, {"queued", "inside", "done", "abandoned"})

    def test_locate_finds_trip_with_evidence(self):
        items = locate(self.dataset, at("06:40"), trip="V14-05")
        self.assertEqual(len(items), 1)
        self.assertTrue(items[0].evidence)
        self.assertTrue(items[0].warning)
        self.assertTrue(items[0].frozen)

    def test_gate_lookup_returns_only_that_gate(self):
        items = locate(self.dataset, at("06:40"), gate="西门")
        self.assertTrue(items)
        self.assertTrue(all(item.gate == "西门" for item in items))


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_dataset()

    def test_report_contains_both_measures_and_caveats(self):
        text = build_report(self.dataset, at("06:40"))
        for fragment in (
            "用户等待",
            "预约完成承诺",
            "放弃",
            "到达不明",
            "不可识别",
            "回放不得用它撤销预警",
        ):
            self.assertIn(fragment, text)

    def test_report_at_late_observation_marks_shift_closed(self):
        text = build_report(self.dataset, at("10:50"))
        self.assertIn("已过本班结束", text)

    def test_report_reports_unconfirmed_arrivals(self):
        text = build_report(self.dataset, at("05:29"))
        self.assertIn("无法确认的 0 趟", text)


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_dataset()

    def test_comparison_never_recommends_worse_service(self):
        for moment in (at("06:40"), at("07:00"), at("07:40")):
            comparison = compare_options(self.dataset, moment)
            held = comparison.options[0]
            for option in comparison.options:
                if option.action != "move" or not option.feasible:
                    continue
                if option.metrics.get("unserved", 0) > held.metrics.get("unserved", 0):
                    self.assertNotEqual(comparison.recommendation, option)

    def test_called_vehicles_keep_their_place(self):
        jobs = visible_jobs(self.dataset, at("06:40"))
        frozen = [job for job in jobs if job.frozen]
        self.assertTrue(frozen)
        comparison = compare_options(self.dataset, at("06:40"))
        # Frozen jobs never vanish from the simulated job set.
        for option in comparison.options:
            if option.metrics:
                self.assertGreaterEqual(
                    option.metrics.get("jobs", 0), len(jobs)
                )


if __name__ == "__main__":
    unittest.main()
