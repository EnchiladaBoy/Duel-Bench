"""The mode table is the single source of truth for the rules. If it can drift
between the three components, every downstream guarantee is void."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import modes  # noqa: E402


class TestTable(unittest.TestCase):
    def test_all_four_modes_exist(self):
        self.assertEqual(modes.names(), ["move-timed", "realtime", "time-bank", "untimed"])

    def test_env_round_trip_is_exact(self):
        # Any lossy field here means the proxy and the harness could disagree
        # with the orchestrator about the rules mid-match.
        for name in modes.names():
            mode = modes.resolve(name)
            self.assertEqual(modes.from_env(modes.to_env(mode)), mode, name)

    def test_round_trip_preserves_overrides(self):
        mode = modes.resolve("time-bank", time_bank=42.0, max_requests=7)
        restored = modes.from_env(modes.to_env(mode))
        self.assertEqual(restored.time_bank, 42.0)
        self.assertEqual(restored.max_requests, 7)

    def test_time_bank_is_lockstep(self):
        # With lockstep off the mode collapses into realtime-with-a-cap and the
        # intended dynamic (each agent charged its OWN thinking time) is lost.
        self.assertTrue(modes.resolve("time-bank").lockstep)

    def test_time_bank_request_budget_cannot_pre_empt_the_bank(self):
        # A fast model with a 300s bank at ~0.5s/move wants ~600 moves. If the
        # request budget were the usual 200 it would be hit first, and budget
        # exhaustion is fatal - so every time-bank match would score unrated.
        bank = modes.resolve("time-bank")
        self.assertGreater(bank.max_requests, 600)
        self.assertGreater(bank.max_steps, 300)

    def test_move_timed_tolerates_more_misses_than_untimed(self):
        # A mode whose premise is "miss the deadline, lose the round" must not
        # eject an agent from the game entirely after two misses.
        self.assertGreater(modes.resolve("move-timed").max_missed_rounds,
                           modes.resolve("untimed").max_missed_rounds)

    def test_every_mode_has_a_wall_clock(self):
        # The rule in realtime, a runaway guard everywhere else - but always
        # present, so monitor() needs no per-mode branch.
        for name in modes.names():
            self.assertIsInstance(modes.resolve(name).wall_clock, float)

    def test_only_realtime_terminates_on_the_clock(self):
        for name in modes.names():
            mode = modes.resolve(name)
            expected = "wall_clock" if name == "realtime" else mode.termination
            self.assertEqual(mode.termination, expected)
        self.assertEqual(modes.resolve("realtime").termination, "wall_clock")


class TestOverrideValidation(unittest.TestCase):
    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(modes.ModeError):
            modes.resolve("blitz")

    def test_meaningless_override_is_rejected_not_ignored(self):
        # Silently ignoring a flag is how benchmark configs rot.
        with self.assertRaises(modes.ModeError):
            modes.resolve("realtime", time_bank=300.0)
        with self.assertRaises(modes.ModeError):
            modes.resolve("time-bank", move_deadline=10.0)

    def test_none_overrides_are_ignored(self):
        self.assertEqual(modes.resolve("untimed", time_bank=None),
                         modes.resolve("untimed"))


class TestPromptNote(unittest.TestCase):
    def test_every_mode_describes_itself_to_the_model(self):
        # An agent not told which mode it is in cannot play that mode's
        # strategy, and the benchmark would measure nothing about the tradeoff.
        for name in modes.names():
            note = modes.prompt_note(modes.resolve(name))
            self.assertIn(name, note)
            self.assertTrue(note.endswith("\n"))

    def test_time_bank_prompt_explains_the_budget(self):
        note = modes.prompt_note(modes.resolve("time-bank"))
        self.assertIn("300", note)
        self.assertIn("free", note)   # waiting for the opponent is free


if __name__ == "__main__":
    unittest.main()


class TestBudgetInvariant(unittest.TestCase):
    """The harness retries a failed step up to MAX_MODEL_RETRIES times and every
    attempt spends a request. If the request budget can bind before the step cap,
    an unlucky-but-honest agent is stopped by a safety net rather than by the
    mode's own terminator - and the budget stops being evidence of anything."""

    def test_every_shipped_mode_satisfies_it(self):
        # realtime (200 vs 240) and time-bank (2500 vs 6000) both violated this
        # before the check existed.
        for name in modes.names():
            mode = modes.resolve(name)
            self.assertFalse(modes.budget_binds_early(mode), name)

    def test_the_table_is_validated_at_import(self):
        bad = modes.MODES["realtime"]._replace(max_requests=1)
        with self.assertRaises(modes.ModeError):
            modes.check_budget_invariant(bad, shipped=True)

    def test_a_deliberate_override_warns_rather_than_raising(self):
        # A test wanting a three-request budget is legitimate; an operator
        # overriding both knobs is making a choice.
        mode = modes.resolve("untimed", max_requests=3)
        self.assertEqual(mode.max_requests, 3)
