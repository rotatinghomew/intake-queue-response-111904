import unittest
from datetime import datetime, timedelta
from pathlib import Path

from citrus.model import (
    ABANDON,
    CALL,
    CHANGE,
    DONE,
    ENTRY,
    WAIT,
    WEIGH,
    Appointment,
    Capacity,
    Dataset,
    Move,
    Outage,
    Policy,
    Resources,
    Signal,
    build_resources,
    load_capacity,
    load_dataset,
)
from citrus.plan import compare_options, visible_jobs
from citrus.queue import replay_integrity, snapshot, warning_reasons
from citrus.stats import (
    median_bounds,
    naive_comparison,
    proportion_over,
    summarize_shift,
)
from citrus.timeline import (
    BASIS_EXACT,
    BASIS_OUTAGE,
    BASIS_UNCONFIRMED,
    build_all,
    build_trip,
    duplicate_summary,
    outage_triage,
)

TZ = datetime.fromisoformat("2026-11-09T00:00:00+08:00").tzinfo


def at(hhmm: str, day: str = "2026-11-09") -> datetime:
    return datetime.fromisoformat(f"{day}T{hhmm}:00+08:00")


def make_signal(
    trip: str,
    kind: str,
    gate: str,
    clock: str,
    visible: str | None = None,
    source: str = "gate",
    event: str | None = None,
    duplicate: bool = False,
) -> Signal:
    return Signal(
        event=event or f"{trip}:{kind}",
        trip=trip,
        vehicle="车",
        source=source,
        kind=kind,
        gate=gate,
        clock=at(clock),
        visible_at=at(visible or clock),
        duplicate=duplicate,
    )


def make_dataset(
    signals: dict[str, list[Signal]],
    appointments: dict[str, Appointment] | None = None,
    policy: Policy | None = None,
    moves: list[Move] | None = None,
) -> Dataset:
    appointments = appointments or {}
    policy = policy or Policy(
        clock_offset_minutes={"yard": 0, "gate": 0, "scale": 0, "dispatch": 0},
        outages=(),
        move_minutes=20,
        observe_at=(at("06:00"),),
        shift_end=at("09:00"),
        warning_wait_minutes=45,
    )
    capacity = load_capacity()
    resources = build_resources(capacity)
    return Dataset(
        appointments=appointments,
        signals=signals,
        capacity=capacity,
        moves=moves or [],
        policy=policy,
        resources=resources,
    )


def appointment(
    trip: str,
    promised: str = "07:00",
    gate: str = "北门",
    available: tuple[str, ...] = ("北门", "西门"),
    vehicle_class: str = "小车",
) -> Appointment:
    return Appointment(
        trip=trip,
        vehicle="车",
        requested_gate=gate,
        available_gates=available,
        vehicle_class=vehicle_class,
        promised_complete_by=at(promised),
    )


class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_dataset()

    def test_duplicate_wait_signals_do_not_add_vehicles(self):
        info = duplicate_summary(self.dataset)
        self.assertEqual(info["trips"], 540)
        self.assertGreater(info["duplicate_wait_signals"], 0)
        # Every trip still has exactly one counted arrival.
        self.assertEqual(
            sum(
                1
                for trip in self.dataset.appointments
                if any(
                    signal.kind == WAIT and not signal.duplicate
                    for signal in self.dataset.signals_for(trip)
                )
            ),
            530,
        )

    def test_clock_offsets_applied(self):
        for trip in self.dataset.appointments:
            for signal in self.dataset.signals_for(trip):
                offset = self.dataset.policy.clock_offset_minutes[signal.source]
                self.assertEqual(
                    signal.clock, signal.source_clock - timedelta(minutes=offset)
                )

    def test_visibility_never_precedes_business_time(self):
        for trip in self.dataset.appointments:
            for signal in self.dataset.signals_for(trip):
                self.assertGreaterEqual(signal.visible_at, signal.clock)

    def test_outage_trips_are_interval_not_dropped(self):
        triage = outage_triage(self.dataset)
        self.assertEqual(len(triage["confirmed"]), 10)
        self.assertEqual(triage["unconfirmed"], [])
        for trip in triage["confirmed"]:
            record = build_trip(self.dataset, trip)
            self.assertEqual(record.arrival_basis, BASIS_OUTAGE)
            self.assertLess(record.arrival_lo, record.arrival_hi)


class TimelineTests(unittest.TestCase):
    def test_unconfirmed_arrival_is_never_zero(self):
        signals = {
            "T1": [
                make_signal("T1", CALL, "北门", "05:30", source="dispatch"),
                make_signal("T1", ENTRY, "北门", "05:32"),
            ]
        }
        dataset = make_dataset(signals, {"T1": appointment("T1")})
        record = build_trip(dataset, "T1")
        self.assertEqual(record.arrival_basis, BASIS_UNCONFIRMED)
        self.assertIsNone(record.arrival_lo)
        self.assertIsNone(record.wait_interval())

    def test_outage_interval_requires_other_source(self):
        signals = {
            "T1": [
                make_signal("T1", ENTRY, "北门", "05:50"),
            ]
        }
        policy = Policy(
            clock_offset_minutes={"yard": 0, "gate": 0, "scale": 0, "dispatch": 0},
            outages=(Outage("yard", at("05:35"), at("06:10")),),
            move_minutes=20,
            observe_at=(at("06:20"),),
            shift_end=at("09:00"),
            warning_wait_minutes=45,
        )
        dataset = make_dataset(
            signals, {"T1": appointment("T1")}, policy=policy
        )
        record = build_trip(dataset, "T1")
        self.assertEqual(record.arrival_basis, BASIS_OUTAGE)
        self.assertEqual(record.arrival_lo, at("05:35"))
        self.assertEqual(record.arrival_hi, at("05:50"))

    def test_gate_change_is_counted_once_at_final_gate(self):
        signals = {
            "T1": [
                make_signal("T1", WAIT, "北门", "05:10", source="yard"),
                make_signal("T1", CHANGE, "西门", "05:20", source="dispatch"),
                make_signal("T1", ENTRY, "西门", "05:30"),
                make_signal("T1", DONE, "西门", "06:00"),
            ]
        }
        dataset = make_dataset(signals, {"T1": appointment("T1")})
        record = build_trip(dataset, "T1")
        self.assertEqual(record.final_gate, "西门")
        self.assertEqual(record.wait_interval(), (50.0, 50.0))

    def test_gate_change_record_to_same_gate_changes_nothing(self):
        signals = {
            "T1": [
                make_signal("T1", WAIT, "西门", "05:10", source="yard"),
                make_signal("T1", CHANGE, "西门", "05:20", source="dispatch"),
                make_signal("T1", DONE, "西门", "06:00"),
            ]
        }
        dataset = make_dataset(signals, {"T1": appointment("T1", gate="西门")})
        record = build_trip(dataset, "T1")
        self.assertIsNone(record.change)
        self.assertEqual(record.final_gate, "西门")

    def test_pending_trip_wait_is_open_ended(self):
        signals = {
            "T1": [make_signal("T1", WAIT, "北门", "05:10", source="yard")]
        }
        dataset = make_dataset(signals, {"T1": appointment("T1")})
        record = build_trip(dataset, "T1", as_of=at("06:00"))
        low, high = record.wait_interval(as_of=at("06:00"))
        self.assertEqual(low, 50.0)
        self.assertIsNone(high)

    def test_duplicate_registration_does_not_move_arrival(self):
        first = make_signal("T1", WAIT, "北门", "05:10", source="yard", event="e1")
        second = make_signal(
            "T1", WAIT, "北门", "05:40", source="yard", event="e2", duplicate=True
        )
        dataset = make_dataset(
            {"T1": [first, second, make_signal("T1", DONE, "北门", "06:00")]},
            {"T1": appointment("T1")},
        )
        record = build_trip(dataset, "T1")
        self.assertEqual(record.arrival_lo, at("05:10"))
        self.assertEqual(record.wait_interval(), (50.0, 50.0))


class MeasureTests(unittest.TestCase):
    def test_median_bounds_unresolved_mass(self):
        # One unresolved trip among three still lets the median be identified:
        # the worst case places it at the top of the sample.
        low, high = median_bounds([(10.0, 10.0), (20.0, 20.0), (30.0, None)])
        self.assertEqual(low, 20.0)
        self.assertEqual(high, 20.0)
        # Two unresolved trips make the median's upper bound unidentifiable.
        low, high = median_bounds([(10.0, 10.0), (20.0, None), (30.0, None)])
        self.assertEqual(low, 20.0)
        self.assertIsNone(high)

    def test_proportion_over_bounds(self):
        intervals = [(10.0, 10.0), (50.0, 50.0), (20.0, None)]
        low, high = proportion_over(intervals, 45)
        self.assertEqual(low, 1 / 3)
        self.assertEqual(high, 2 / 3)

    def test_abandonment_is_a_real_wait_and_a_missed_promise(self):
        signals = {
            "T1": [
                make_signal("T1", WAIT, "北门", "05:10", source="yard"),
                make_signal("T1", DONE, "北门", "05:50"),
            ],
            "T2": [
                make_signal("T2", WAIT, "北门", "05:10", source="yard"),
                make_signal("T2", ABANDON, "北门", "05:35"),
            ],
        }
        appointments = {
            "T1": appointment("T1", promised="05:40"),
            "T2": appointment("T2", promised="05:40"),
        }
        dataset = make_dataset(signals, appointments)
        records = list(build_all(dataset, as_of=at("06:00")).values())
        summary = summarize_shift(records, at("06:00"), 30)
        self.assertEqual(summary.served, 1)
        self.assertEqual(summary.abandoned, 1)
        self.assertEqual(summary.user.median_low, 32.5)
        self.assertEqual(summary.promise.over_low, 1.0)

    def test_naive_shortcut_hides_abandonment(self):
        signals = {
            "T1": [
                make_signal("T1", WAIT, "北门", "05:10", source="yard"),
                make_signal("T1", ENTRY, "北门", "05:20"),
                make_signal("T1", DONE, "北门", "06:00"),
            ],
            "T2": [
                make_signal("T2", WAIT, "北门", "05:10", source="yard"),
                make_signal("T2", ABANDON, "北门", "05:30"),
            ],
        }
        dataset = make_dataset(
            signals, {"T1": appointment("T1", "07:00"), "T2": appointment("T2", "07:00")}
        )
        records = list(build_all(dataset, as_of=at("07:00")).values())
        naive = naive_comparison(records, at("07:00"))
        self.assertEqual(naive.entered_count, 1)
        self.assertEqual(naive.abandoned_count, 1)
        self.assertGreater(naive.entered_only_mean, naive.all_terminal_mean)


class QueueTests(unittest.TestCase):
    def test_warning_uses_only_records_visible_now(self):
        signals = {
            "T1": [
                make_signal("T1", WAIT, "北门", "05:10", source="yard"),
                make_signal("T1", DONE, "北门", "06:50", visible="07:05"),
            ]
        }
        dataset = make_dataset(signals, {"T1": appointment("T1")})
        reasons = warning_reasons(dataset, at("06:30"))
        self.assertEqual(len(reasons), 1)
        self.assertEqual(reasons[0].trip, "T1")
        self.assertNotIn("T1:DONE", reasons[0].evidence)

    def test_late_departure_does_not_clear_warning_in_replay(self):
        signals = {
            "T1": [
                make_signal("T1", WAIT, "北门", "05:10", source="yard"),
                make_signal("T1", DONE, "北门", "06:35", visible="07:10"),
            ]
        }
        dataset = make_dataset(signals, {"T1": appointment("T1")})
        problems = replay_integrity(dataset, at("06:30"), at("07:20"))
        self.assertEqual(len(problems), 1)
        self.assertIn("T1", problems[0])

    def test_unseen_trips_are_counted_not_dropped(self):
        signals = {
            "T1": [make_signal("T1", WAIT, "北门", "05:10", source="yard")]
        }
        dataset = make_dataset(
            signals, {"T1": appointment("T1"), "T2": appointment("T2")}
        )
        state = snapshot(dataset, at("06:00"))
        self.assertEqual(state.unseen, ["T2"])

    def test_inside_vehicle_is_frozen(self):
        signals = {
            "T1": [
                make_signal("T1", WAIT, "北门", "05:10", source="yard"),
                make_signal("T1", ENTRY, "北门", "05:30"),
            ]
        }
        dataset = make_dataset(signals, {"T1": appointment("T1")})
        state = snapshot(dataset, at("06:00"))
        item = state.active_items[0]
        self.assertTrue(item.frozen)
        self.assertEqual(state.inside["北门"], 1)


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_dataset()

    def test_confirmed_move_blocks_same_line_elsewhere(self):
        comparison = compare_options(self.dataset, at("05:29"))
        moves = [option for option in comparison.options if option.action == "move"]
        self.assertTrue(moves)
        self.assertTrue(all(not option.feasible for option in moves))
        self.assertIn("M0", comparison.inherited_moves)

    def test_option_never_uses_unavailable_gate(self):
        comparison = compare_options(self.dataset, at("06:40"))
        for option in comparison.options:
            if option.action != "move" or option.gate is None:
                continue
            allowed = {
                appointment.available_gates
                for appointment in self.dataset.appointments.values()
            }
            self.assertTrue(
                any(option.gate in gates for gates in allowed),
                option.gate,
            )

    def test_small_improvement_is_not_recommended(self):
        comparison = compare_options(self.dataset, at("06:40"))
        for option in comparison.options:
            if option.action == "move" and option.feasible:
                self.assertEqual(option.indistinguishable_from, "hold")
        self.assertNotEqual(comparison.recommendation.action, "move")

    def test_after_shift_no_new_move(self):
        comparison = compare_options(self.dataset, at("10:50"))
        self.assertTrue(comparison.shift_closed)
        self.assertEqual(comparison.recommendation.action, "manual")
        self.assertFalse(
            any(
                option.action == "move" and option.feasible
                for option in comparison.options
            )
        )

    def test_visible_jobs_exclude_terminal_trips(self):
        jobs = visible_jobs(self.dataset, at("10:50"))
        self.assertTrue(all(job.stage != "done" for job in jobs))
        self.assertEqual(
            sorted(job.trip for job in jobs),
            [
                "V14-05",
                "V14-10",
                "V14-15",
                "V14-20",
                "V14-25",
                "V14-30",
                "V14-35",
            ],
        )


if __name__ == "__main__":
    unittest.main()
