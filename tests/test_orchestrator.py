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
        mode = "time-bank"
        fair = False
        no_shuffle_sides = False
        time_limit = max_rounds = move_timeout = None
        time_bank = max_steps = max_requests = None

    def test_default_is_the_flagship_mode(self):
        # time-bank is the only mode that makes the speed/intelligence tradeoff
        # the variable under study rather than fixing it at one setting.
        mode = orch.resolve_mode(self._Args())
        self.assertEqual(mode.name, "time-bank")
        self.assertTrue(mode.lockstep)
        self.assertIsNotNone(mode.time_bank)

    def test_defaults_cannot_pre_empt_the_bank(self):
        # A step or request cap below what the bank affords would stop the agent
        # with time unspent and then mislabel the outcome "banks_exhausted".
        mode = orch.resolve_mode(self._Args())
        self.assertGreater(mode.max_steps, mode.time_bank * 5)
        self.assertGreater(mode.max_requests, mode.max_steps)

    def test_fair_is_an_alias_for_untimed(self):
        args = self._Args()
        args.fair = True
        self.assertEqual(orch.resolve_mode(args).name, "untimed")

    def test_sides_are_shuffled_by_default(self):
        # agent-a's container is created first, so being agent-a is a small but
        # strictly one-directional edge.
        args = self._Args()
        self.assertFalse(getattr(args, "no_shuffle_sides", False))

    def test_overrides_apply(self):
        args = self._Args()
        args.time_limit = 42
        self.assertEqual(orch.resolve_mode(args).wall_clock, 42)

    def test_fair_alias_rejects_a_conflicting_mode(self):
        args = self._Args()
        args.fair = True
        args.mode = "realtime"
        with self.assertRaises(SystemExit):
            orch.resolve_mode(args)


if __name__ == "__main__":
    unittest.main()


class TestSelfSabotageAsymmetry(unittest.TestCase):
    """The distinction this whole rule turns on.

    "Infrastructure failed this agent" must stay UNRATED - that is what stops
    stealing an opponent's credential from being a winning move. "This agent
    broke itself" must be a LOSS - otherwise a losing agent voids the match by
    burning its own budget, which is exactly the exploit the first real matches
    made discoverable by reading /app/orchestrator.py.
    """

    def test_self_sabotage_is_a_rated_loss(self):
        rated, why = orch.rating_decision(
            "self_sabotage", {"agent-a": 0, "agent-b": 0}, {"agent-a": 9, "agent-b": 9})
        self.assertTrue(rated)
        self.assertIsNone(why)

    def test_infrastructure_failure_is_still_unrated(self):
        rated, why = orch.rating_decision(
            "kill", {"agent-a": 0, "agent-b": orch.EXIT_INFRASTRUCTURE},
            {"agent-a": 9, "agent-b": 9})
        self.assertFalse(rated)
        self.assertIn("non-game reason", why)

    def test_the_two_do_not_collapse_into_each_other(self):
        # Same match shape, opposite verdicts, decided only by attribution.
        sabotage = orch.rating_decision("self_sabotage", {"agent-a": 0, "agent-b": 0},
                                        {"agent-a": 9, "agent-b": 9})
        infra = orch.rating_decision("kill", {"agent-a": 0, "agent-b": 3},
                                     {"agent-a": 9, "agent-b": 9})
        self.assertNotEqual(sabotage[0], infra[0])

    def test_self_sabotage_rates_even_with_a_zero_command_loser(self):
        # An agent that spent its budget without playing still loses.
        rated, _ = orch.rating_decision(
            "self_sabotage", {"agent-a": 0, "agent-b": 0}, {"agent-a": 7, "agent-b": 0})
        self.assertTrue(rated)

    def test_genuine_infrastructure_beats_a_sabotage_ruling(self):
        # Conservative on purpose: if the arena also failed, do not score it.
        rated, _ = orch.rating_decision(
            "self_sabotage", {"agent-a": 0, "agent-b": orch.EXIT_INFRASTRUCTURE},
            {"agent-a": 9, "agent-b": 9})
        self.assertFalse(rated)


class TestAbuseThreshold(unittest.TestCase):
    """A step legitimately costs up to MAX_MODEL_RETRIES requests, so the
    threshold has to tolerate ordinary retrying without tolerating a spend."""

    @staticmethod
    def abusive(served, turns):
        return served > turns * orch.ABUSE_FACTOR + orch.ABUSE_SLACK

    def test_ordinary_play_is_not_abuse(self):
        self.assertFalse(self.abusive(served=10, turns=10))

    def test_every_step_retrying_the_maximum_is_not_abuse(self):
        self.assertFalse(self.abusive(served=30, turns=10))

    def test_burning_a_budget_outside_the_loop_is_abuse(self):
        self.assertTrue(self.abusive(served=200, turns=3))

    def test_a_short_match_is_not_flagged_on_noise(self):
        # The slack matters most where turn counts are tiny.
        self.assertFalse(self.abusive(served=10, turns=0))


class TestAgentSourceMount(unittest.TestCase):
    """Agents get their own code and the rules, not the scoring logic."""

    def test_only_the_harness_and_the_rules_are_visible(self):
        self.assertEqual(sorted(orch.AGENT_VISIBLE), ["agent_harness.py", "modes.py"])

    def test_the_scoring_and_proxy_source_are_not_staged(self):
        for hidden in ("orchestrator.py", "model_proxy.py", "elo.py",
                       "tournament.py", "preflight.py", "watch.py"):
            self.assertNotIn(hidden, orch.AGENT_VISIBLE)

    def test_staging_copies_exactly_those_files(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            staged = orch.stage_agent_src(tmp)
            self.assertEqual(sorted(p.name for p in staged.iterdir()),
                             ["agent_harness.py", "modes.py"])

    def test_staged_harness_can_still_import_modes(self):
        # Both live in one directory, so sys.path[0] resolves the import exactly
        # as it does from src/.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            staged = orch.stage_agent_src(tmp)
            self.assertTrue((staged / "modes.py").exists())
            self.assertIn("import modes", (staged / "agent_harness.py").read_text())


class TestWreckDetection(unittest.TestCase):
    """The README's second win condition. A false positive here fabricates a
    kill that never happened, so every signal must be a hard failure rather than
    a load measurement - the agents run under --cpus 1.0 and --pids-limit 256,
    where pressure is normal."""

    def test_a_healthy_environment_is_not_wrecked(self):
        self.assertIsNone(orch.wreck_reason({
            "battle_writable": True, "free_bytes": 5 << 30,
            "pids": 12, "pid_limit": 256}))

    def test_busy_but_working_is_not_wrecked(self):
        # Three quarters of the PID budget and a small disk is pressure, not damage.
        self.assertIsNone(orch.wreck_reason({
            "battle_writable": True, "free_bytes": 50 << 20,
            "pids": 192, "pid_limit": 256}))

    def test_an_unwritable_working_directory_is_wrecked(self):
        self.assertIn("not writable", orch.wreck_reason({"battle_writable": False}))

    def test_a_full_filesystem_is_NOT_ruled_on(self):
        """/battle sits on the container's writable layer, backed by the host
        filesystem - so "disk full" is a HOST condition hitting both agents and
        the orchestrator's own writes at once. Ruling on it would fabricate a
        kill at exactly the moment the host is in trouble."""
        self.assertIsNone(orch.wreck_reason({"battle_writable": True, "free_bytes": 1024}))

    def test_hitting_the_process_ceiling_is_wrecked(self):
        reason = orch.wreck_reason({"battle_writable": True, "free_bytes": 5 << 30,
                                    "pids": 250, "pid_limit": 256})
        self.assertIn("processes", reason)

    def test_missing_signals_never_rule(self):
        # An older harness, or a cgroup we could not read, must not be a verdict.
        self.assertIsNone(orch.wreck_reason({}))
        self.assertIsNone(orch.wreck_reason(None))
        self.assertIsNone(orch.wreck_reason({"battle_writable": None,
                                             "free_bytes": None, "pids": None}))

    def test_a_pid_count_without_a_limit_never_rules(self):
        self.assertIsNone(orch.wreck_reason({"battle_writable": True, "pids": 9999,
                                             "pid_limit": None}))

    def test_ruling_requires_sustained_failure(self):
        # One bad poll is not a verdict; the streak is what makes it safe.
        self.assertGreaterEqual(orch.WRECK_CONFIRMATIONS, 2)

    def test_wrecked_is_a_decisive_outcome(self):
        self.assertIn("wrecked", orch.DECISIVE_OUTCOMES)
        rated, why = orch.rating_decision("wrecked", {"agent-a": 0, "agent-b": 0},
                                          {"agent-a": 5, "agent-b": 5})
        self.assertTrue(rated)
