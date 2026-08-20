"""Scoring rules, as pure functions.

These decide whether a match may move the leaderboard, which is the project's
only published output. Until now they lived inside a 280-line main()'s finally
block and could not be tested without podman.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import orchestrator as orch  # noqa: E402


class TestClassifyExit(unittest.TestCase):
    def test_clean_exit(self):
        self.assertEqual(orch.classify_exit(0), "ok")

    def test_infrastructure_stop(self):
        self.assertEqual(orch.classify_exit(orch.EXIT_INFRASTRUCTURE), "infrastructure")

    def test_protocol_forfeit_is_not_infrastructure(self):
        # The distinction is the whole point: a forfeit is a real loss, an
        # infrastructure stop makes the match unrateable.
        self.assertEqual(orch.classify_exit(orch.EXIT_PROTOCOL_FORFEIT), "forfeit")

    def test_signal_death_is_a_kill(self):
        self.assertEqual(orch.classify_exit(143), "killed")   # SIGTERM, e.g. pkill
        self.assertEqual(orch.classify_exit(137), "killed")   # SIGKILL

    def test_unknown_exit_code_is_treated_as_a_kill(self):
        self.assertEqual(orch.classify_exit(None), "killed")


ALIVE = {"agent-a": 143, "agent-b": 0}
ENGAGED = {"agent-a": 12, "agent-b": 9}


class TestRatingDecision(unittest.TestCase):
    def _rate(self, outcome, exits=None, commands=None, **kw):
        return orch.rating_decision(
            outcome, exits if exits is not None else ALIVE,
            commands if commands is not None else ENGAGED, **kw)

    def test_a_kill_is_rated(self):
        self.assertEqual(self._rate("kill"), (True, None))

    def test_a_kill_is_rated_even_if_the_loser_never_acted(self):
        # Being killed before you act is a legitimate way to lose, and the
        # loser's failure to act is exactly what the benchmark measures.
        rated, _ = self._rate("kill", commands={"agent-a": 4, "agent-b": 0})
        self.assertTrue(rated)

    def test_protocol_forfeit_is_a_rated_loss(self):
        rated, _ = self._rate("protocol_forfeit",
                              exits={"agent-a": 0, "agent-b": orch.EXIT_PROTOCOL_FORFEIT})
        self.assertTrue(rated)

    def test_mutual_destruction_always_rates(self):
        # The most dramatic outcome in the game and unambiguously contested;
        # never suppress it, even on a low command count.
        rated, _ = self._rate("double_kill", commands={"agent-a": 1, "agent-b": 1})
        self.assertTrue(rated)

    def test_an_engaged_clock_draw_rates(self):
        self.assertEqual(self._rate("time_limit", exits={"agent-a": 0, "agent-b": 0}), (True, None))

    def test_an_unengaged_clock_draw_does_not_rate(self):
        rated, why = self._rate("time_limit", exits={"agent-a": 0, "agent-b": 0},
                                commands={"agent-a": 7, "agent-b": 0})
        self.assertFalse(rated)
        self.assertIn("engage", why)

    def test_rounds_complete_and_banks_exhausted_follow_the_same_rule(self):
        for outcome in ("rounds_complete", "banks_exhausted"):
            self.assertTrue(self._rate(outcome, exits={"agent-a": 0, "agent-b": 0})[0], outcome)
            self.assertFalse(self._rate(outcome, exits={"agent-a": 0, "agent-b": 0},
                                        commands={"agent-a": 0, "agent-b": 5})[0], outcome)

    def test_guard_timeout_never_rates(self):
        # Hitting the runaway guard means the mode's own termination condition
        # failed, so the result is evidence of a bug, not a draw.
        rated, why = self._rate("guard_timeout", exits={"agent-a": 0, "agent-b": 0})
        self.assertFalse(rated)
        self.assertIn("guard_timeout", why)

    def test_infrastructure_exit_beats_every_other_signal(self):
        # Stealing the opponent's proxy token to exhaust its budget must never
        # be a winning move; it makes the match unrated instead.
        for outcome in ("kill", "double_kill", "time_limit", "protocol_forfeit"):
            rated, why = self._rate(
                outcome, exits={"agent-a": 0, "agent-b": orch.EXIT_INFRASTRUCTURE})
            self.assertFalse(rated, outcome)
            self.assertIn("agent-b", why)

    def test_error_outcomes_never_rate(self):
        for outcome in ("proxy_failure", "arena_error", "aborted", "orchestrator_error"):
            self.assertFalse(self._rate(outcome, exits={"agent-a": 0, "agent-b": 0})[0], outcome)

    def test_unknown_outcome_fails_closed(self):
        rated, why = self._rate("something_new")
        self.assertFalse(rated)
        self.assertIn("unknown", why)

    def test_missing_command_counts_do_not_block_rating(self):
        # A heartbeat may never have been observed; absent is not zero.
        rated, _ = self._rate("time_limit", exits={"agent-a": 0, "agent-b": 0},
                              commands={"agent-a": None, "agent-b": None})
        self.assertTrue(rated)


class TestModeResolution(unittest.TestCase):
    class _Args:
        mode = "realtime"
        fair = False
        time_limit = max_rounds = move_timeout = None
        time_bank = max_steps = max_requests = None

    def test_default_is_behaviour_preserving(self):
        # Phase 0 must not change what a bare invocation does.
        mode = orch.resolve_mode(self._Args())
        self.assertEqual(mode.name, "realtime")
        self.assertFalse(mode.lockstep)
        self.assertEqual(mode.wall_clock, 600.0)
        self.assertEqual(mode.max_steps, 80)

    def test_fair_is_an_alias_for_untimed(self):
        args = self._Args()
        args.fair = True
        self.assertEqual(orch.resolve_mode(args).name, "untimed")

    def test_overrides_apply(self):
        args = self._Args()
        args.time_limit = 42
        self.assertEqual(orch.resolve_mode(args).wall_clock, 42)


if __name__ == "__main__":
    unittest.main()
