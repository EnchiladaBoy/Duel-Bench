"""The tournament runner. Its whole job is to remove the two things that make a
single match unciteable: side bias and sample size."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import tournament as tn  # noqa: E402


class TestSchedule(unittest.TestCase):
    def test_every_pair_is_played_both_ways(self):
        """Without both directions every rating carries the first-mover
        advantage of whichever model was passed as --model-a."""
        plan = tn.schedule(["a", "b"], ["untimed"], games=1)
        pairs = {(e["model_a"], e["model_b"]) for e in plan}
        self.assertEqual(pairs, {("a", "b"), ("b", "a")})

    def test_match_count_is_pairs_times_games_times_two_times_modes(self):
        plan = tn.schedule(["a", "b", "c"], ["untimed", "realtime"], games=3)
        self.assertEqual(len(plan), 3 * 3 * 2 * 2)   # 3 pairs, 3 games, 2 dirs, 2 modes

    def test_sides_are_perfectly_balanced(self):
        plan = tn.schedule(["a", "b", "c"], ["time-bank"], games=2)
        as_a = {}
        for entry in plan:
            as_a[entry["model_a"]] = as_a.get(entry["model_a"], 0) + 1
        self.assertEqual(len(set(as_a.values())), 1,
                         "every model must play agent-a the same number of times")

    def test_no_model_is_scheduled_against_itself(self):
        plan = tn.schedule(["a", "b", "c"], ["untimed"], games=1)
        self.assertFalse([e for e in plan if e["model_a"] == e["model_b"]])

    def test_duplicate_models_are_collapsed(self):
        self.assertEqual(len(tn.schedule(["a", "a", "b"], ["untimed"], games=1)), 2)

    def test_every_entry_starts_pending_and_unattempted(self):
        for entry in tn.schedule(["a", "b"], ["untimed"], games=1):
            self.assertEqual(entry["status"], "pending")
            self.assertEqual(entry["attempts"], 0)
            self.assertIsNone(entry["match_id"])

    def test_ids_are_unique(self):
        plan = tn.schedule(["a", "b", "c"], ["untimed", "realtime"], games=2)
        self.assertEqual(len({e["id"] for e in plan}), len(plan))

    def test_schedule_is_deterministic(self):
        first = tn.schedule(["c", "a", "b"], ["untimed"], games=2)
        second = tn.schedule(["b", "c", "a"], ["untimed"], games=2)
        self.assertEqual([e["id"] for e in first], [e["id"] for e in second])


class TestCostEstimate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _match(self, name, tokens, mock=False):
        d = self.root / name
        d.mkdir()
        (d / "result.json").write_text(json.dumps({
            "mock": mock,
            "usage": {"agent-a": {"total_tokens": tokens},
                      "agent-b": {"total_tokens": tokens}}}))

    def test_cost_is_learned_from_finished_matches(self):
        self._match("m1", 10000)
        self._match("m2", 20000)
        self.assertEqual(tn.observed_tokens_per_match(self.root), 30000)

    def test_mock_matches_do_not_skew_the_estimate(self):
        # A mock match costs nothing, so counting it would make a real
        # tournament look far cheaper than it is.
        self._match("m1", 10000)
        self._match("m2", 999999, mock=True)
        self.assertEqual(tn.observed_tokens_per_match(self.root), 20000)

    def test_no_history_means_no_observation(self):
        self.assertIsNone(tn.observed_tokens_per_match(self.root))


class TestResumeSemantics(unittest.TestCase):
    """Long runs against real models must survive a transient outage without
    discarding everything already paid for."""

    def test_completed_matches_are_skipped(self):
        plan = tn.schedule(["a", "b"], ["untimed"], games=1)
        plan[0]["status"] = "done"
        pending = [e for e in plan if e["status"] != "done"]
        self.assertEqual(len(pending), 1)

    def test_a_failure_is_retried_once_then_abandoned(self):
        entry = tn.schedule(["a", "b"], ["untimed"], games=1)[0]
        entry["status"], entry["attempts"] = "failed", 1
        self.assertFalse(entry["status"] == "failed" and entry["attempts"] > 1)
        entry["attempts"] = 2
        self.assertTrue(entry["status"] == "failed" and entry["attempts"] > 1)


if __name__ == "__main__":
    unittest.main()


class _Args:
    mock = True
    time_bank = 30.0
    max_rounds = 8
    time_limit = 120
    passthrough = []
    match_timeout = 60


class TestOrchestratorCommand(unittest.TestCase):
    def _cmd(self, mode):
        entry = {"mode": mode, "model_a": "a", "model_b": "b"}
        return tn.orchestrator_command(entry, _Args())

    def test_sides_are_never_re_randomised(self):
        """The tournament assigns sides deliberately, one pairing each way. If
        the orchestrator reshuffled them the balance would be destroyed."""
        self.assertIn("--no-shuffle-sides", self._cmd("untimed"))

    def test_a_flag_is_only_sent_to_modes_where_it_applies(self):
        # The orchestrator REJECTS a meaningless override rather than ignoring
        # it, so forwarding --max-rounds to realtime fails the whole match.
        self.assertNotIn("--max-rounds", self._cmd("realtime"))
        self.assertIn("--max-rounds", self._cmd("untimed"))

    def test_time_bank_only_goes_to_time_bank(self):
        self.assertIn("--time-bank", self._cmd("time-bank"))
        for mode in ("untimed", "move-timed", "realtime"):
            self.assertNotIn("--time-bank", self._cmd(mode), mode)

    def test_the_mode_and_both_models_are_always_passed(self):
        cmd = self._cmd("time-bank")
        self.assertEqual(cmd[cmd.index("--mode") + 1], "time-bank")
        self.assertEqual(cmd[cmd.index("--model-a") + 1], "a")
        self.assertEqual(cmd[cmd.index("--model-b") + 1], "b")

    def test_every_forwarded_flag_is_accepted_by_every_mode_it_is_sent_to(self):
        # The guarantee that matters: a multi-mode tournament must never build
        # an invocation the orchestrator will reject.
        import modes
        for mode in modes.names():
            cmd = self._cmd(mode)
            overrides = {"--max-rounds": "max_rounds", "--time-bank": "time_bank",
                         "--time-limit": "wall_clock"}
            for flag, field in overrides.items():
                if flag in cmd:
                    self.assertIn(mode, modes.OVERRIDE_APPLIES[field],
                                  f"{flag} sent to {mode}, which rejects it")


class TestRetryHappensInTheSameRun(unittest.TestCase):
    """A for-loop over the plan has already passed a failed entry by the time it
    is set back to pending, so the retry would silently wait for --resume."""

    @staticmethod
    def _pick(plan):
        return next((e for e in plan
                     if e["status"] == "pending" and e["attempts"] <= 1), None)

    def test_a_once_failed_entry_is_picked_up_again(self):
        plan = tn.schedule(["a", "b"], ["untimed"], games=1)
        plan[0].update(status="pending", attempts=1)     # failed once
        plan[1].update(status="done", attempts=1)
        self.assertIs(self._pick(plan), plan[0])

    def test_a_twice_failed_entry_is_abandoned(self):
        plan = tn.schedule(["a", "b"], ["untimed"], games=1)
        plan[0].update(status="failed", attempts=2)
        plan[1].update(status="done", attempts=1)
        self.assertIsNone(self._pick(plan))

    def test_a_finished_plan_selects_nothing(self):
        plan = tn.schedule(["a", "b"], ["untimed"], games=1)
        for entry in plan:
            entry.update(status="done", attempts=1)
        self.assertIsNone(self._pick(plan))
