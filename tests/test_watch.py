"""The match viewer rebuilds its whole picture from the event stream, so the
stream has to be sufficient on its own - that is what makes replay and a future
UI possible without touching the arena."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import watch  # noqa: E402


def ev(event, t=0.0, src="proxy", **payload):
    return {"v": 1, "seq": 0, "t": t, "src": src, "event": event, **payload}


class TestStateReconstruction(unittest.TestCase):
    def setUp(self):
        self.state = watch.MatchState()

    def feed(self, *events):
        for event in events:
            self.state.apply(event)
        return self.state

    def test_match_start_establishes_the_board(self):
        s = self.feed(ev("match_start", src="orchestrator", mode="time-bank",
                         model_a="x/one", model_b="y/two",
                         mode_config={"time_bank": 300.0}))
        self.assertEqual(s.mode, "time-bank")
        self.assertEqual(s.models["agent-a"], "x/one")
        self.assertEqual(s.bank_granted, 300.0)

    def test_commands_are_counted_per_agent(self):
        s = self.feed(ev("command_start", src="agent-a", step=1, command="ps aux"),
                      ev("command_start", src="agent-a", step=2, command="whoami"),
                      ev("command_start", src="agent-b", step=1, command="ls"))
        self.assertEqual(s.agents["agent-a"]["commands"], 2)
        self.assertEqual(s.agents["agent-b"]["commands"], 1)
        self.assertEqual(s.agents["agent-a"]["last"], "whoami")

    def test_bank_tracks_move_start_and_thinking_ticks(self):
        # A clock UI decrements locally between these and snaps to each update.
        s = self.feed(ev("move_start", agent="agent-a", bank_remaining=120.0),
                      ev("thinking", agent="agent-a", elapsed=3.0, bank_remaining=117.0))
        self.assertEqual(s.agents["agent-a"]["bank"], 117.0)
        self.assertEqual(s.agents["agent-a"]["thinking"], 3.0)

    def test_completion_clears_the_thinking_indicator(self):
        s = self.feed(ev("move_start", agent="agent-a", bank_remaining=10.0),
                      ev("thinking", agent="agent-a", elapsed=2.0),
                      ev("completion", agent="agent-a", bank_remaining=8.0))
        self.assertIsNone(s.agents["agent-a"]["thinking"])
        self.assertEqual(s.agents["agent-a"]["bank"], 8.0)

    def test_exhaustion_zeroes_the_clock(self):
        s = self.feed(ev("move_start", agent="agent-a", bank_remaining=0.4),
                      ev("bank_exhausted", agent="agent-a"))
        self.assertEqual(s.agents["agent-a"]["bank"], 0.0)

    def test_forfeits_and_passes_are_distinguished(self):
        s = self.feed(ev("move_forfeit", agent="agent-a"),
                      ev("pass", src="agent-b"))
        self.assertEqual(s.agents["agent-a"]["forfeits"], 1)
        self.assertEqual(s.agents["agent-b"]["passes"], 1)

    def test_agent_down_is_reflected(self):
        s = self.feed(ev("agent_down", src="orchestrator", agent="agent-b", how="exited"))
        self.assertFalse(s.agents["agent-b"]["alive"])

    def test_a_snapshot_alone_reconstructs_the_scoreboard(self):
        # This is what lets a viewer join mid-match without replaying the stream.
        s = self.feed(ev("snapshot", src="orchestrator", round=7,
                         banks={"agent-a": 12.5, "agent-b": 200.0},
                         agents={"agent-a": {"alive": True, "commands_run": 9},
                                 "agent-b": {"alive": False, "commands_run": 4}}))
        self.assertEqual(s.round, 7)
        self.assertEqual(s.agents["agent-a"]["bank"], 12.5)
        self.assertEqual(s.agents["agent-a"]["commands"], 9)
        self.assertFalse(s.agents["agent-b"]["alive"])

    def test_snapshot_never_lowers_a_command_count(self):
        # Snapshots come from a heartbeat that can be a poll stale.
        s = self.feed(ev("command_start", src="agent-a", step=1, command="x"),
                      ev("command_start", src="agent-a", step=2, command="y"),
                      ev("snapshot", src="orchestrator",
                         agents={"agent-a": {"alive": True, "commands_run": 1}}))
        self.assertEqual(s.agents["agent-a"]["commands"], 2)

    def test_match_end_stops_the_follower(self):
        s = self.feed(ev("match_end", src="orchestrator", winner="agent-a",
                         outcome="kill", rated=True, duration=12.0))
        self.assertIsNotNone(s.finished)
        self.assertTrue(s.rated)

    def test_elapsed_never_runs_backwards(self):
        s = self.feed(ev("a", t=5.0), ev("b", t=2.0))
        self.assertEqual(s.elapsed, 5.0)

    def test_unknown_events_are_ignored_not_fatal(self):
        # The stream will grow new event types; an old viewer must not crash.
        self.feed(ev("something_invented_later", agent="agent-a", payload={"x": 1}))

    def test_feed_is_bounded(self):
        for i in range(500):
            self.state.apply(ev("command_start", src="agent-a", step=i, command=f"c{i}"))
        self.assertLessEqual(len(self.state.feed), 200)


class TestBar(unittest.TestCase):
    def test_full_and_empty_clamp(self):
        self.assertIn("█", watch.bar(1.0))
        self.assertNotIn("░", watch.bar(1.0))
        self.assertNotIn("█", watch.bar(0.0))

    def test_out_of_range_does_not_break_the_layout(self):
        for fraction in (-5.0, 0.5, 99.0):
            plain = (watch.bar(fraction)
                     .replace(watch.GREEN, "").replace(watch.YELLOW, "")
                     .replace(watch.RED, "").replace(watch.GREY, "")
                     .replace(watch.RESET, ""))
            self.assertEqual(len(plain), 22)


if __name__ == "__main__":
    unittest.main()
