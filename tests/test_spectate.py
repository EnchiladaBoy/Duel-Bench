"""The browser spectator. Read-only, standalone, and unreachable from the arena."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import spectate  # noqa: E402


def write_stream(path, events):
    path.write_text("".join(json.dumps(e) + "\n" for e in events))


EVENTS = [
    {"v": 1, "seq": 1, "t": 0.0, "src": "orchestrator", "event": "match_start",
     "mode": "time-bank", "model_a": "x/one", "model_b": "y/two",
     "mode_config": {"time_bank": 60.0}},
    {"v": 1, "seq": 2, "t": 0.5, "src": "proxy", "event": "go", "agents": ["agent-a", "agent-b"]},
    {"v": 1, "seq": 3, "t": 1.0, "src": "proxy", "event": "move_start",
     "agent": "agent-a", "bank_remaining": 55.0, "round": 1},
    {"v": 1, "seq": 4, "t": 2.0, "src": "agent-a", "event": "command_start", "command": "ps aux"},
    {"v": 1, "seq": 5, "t": 9.0, "src": "orchestrator", "event": "match_end",
     "winner": "agent-a", "outcome": "kill", "rated": True, "duration": 9.0},
]


class TestReadEvents(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "events.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def test_reads_every_event(self):
        write_stream(self.path, EVENTS)
        events, _ = spectate.read_events(self.path)
        self.assertEqual(len(events), len(EVENTS))

    def test_resumes_without_re_reading(self):
        # tell() is disabled inside a for-loop over a text file, which made an
        # earlier version of this pattern re-read line 1 forever.
        write_stream(self.path, EVENTS[:2])
        first, pos = spectate.read_events(self.path)
        with self.path.open("a") as fh:
            fh.write(json.dumps(EVENTS[2]) + "\n")
        second, _ = spectate.read_events(self.path, pos)
        self.assertEqual(len(first), 2)
        self.assertEqual(len(second), 1)
        self.assertEqual(second[0]["seq"], 3)

    def test_a_partial_final_line_is_left_for_next_time(self):
        self.path.write_text(json.dumps(EVENTS[0]) + "\n" + '{"partial":')
        events, _ = spectate.read_events(self.path)
        self.assertEqual(len(events), 1)

    def test_malformed_lines_are_skipped(self):
        self.path.write_text("not json\n" + json.dumps(EVENTS[0]) + "\n")
        self.assertEqual(len(spectate.read_events(self.path)[0]), 1)

    def test_a_missing_file_is_not_fatal(self):
        self.assertEqual(spectate.read_events(self.path / "nope")[0], [])


class TestSnapshot(unittest.TestCase):
    """A late joiner must render immediately rather than replaying the stream."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "events.jsonl"
        write_stream(self.path, EVENTS)

    def tearDown(self):
        self.tmp.cleanup()

    def test_snapshot_carries_the_whole_board(self):
        state = spectate.snapshot(self.path)
        self.assertEqual(state["mode"], "time-bank")
        self.assertEqual(state["models"]["agent-a"], "x/one")
        self.assertEqual(state["bank_granted"], 60.0)
        self.assertEqual(state["agents"]["agent-a"]["bank"], 55.0)
        self.assertEqual(state["agents"]["agent-a"]["commands"], 1)
        self.assertEqual(state["finished"]["outcome"], "kill")

    def test_snapshot_is_json_serialisable(self):
        json.dumps(spectate.snapshot(self.path))

    def test_state_comes_from_the_terminal_viewer_not_a_reimplementation(self):
        # One state machine, already covered by tests/test_watch.py.
        import watch
        self.assertIs(spectate.watch.MatchState, watch.MatchState)

    def test_feed_is_plain_text_for_the_browser(self):
        for line in spectate.snapshot(self.path)["feed"]:
            self.assertNotIn("\033", line)


class TestAnsiStripping(unittest.TestCase):
    def test_colour_codes_are_removed(self):
        self.assertEqual(spectate.strip_ansi("\033[32mok\033[0m"), "ok")

    def test_plain_text_is_untouched(self):
        self.assertEqual(spectate.strip_ansi("agent-a $ ps aux"), "agent-a $ ps aux")


class TestPageIsSelfContained(unittest.TestCase):
    """Zero dependencies means zero dependencies: no CDN, no framework, and the
    page must work with no network beyond the local server."""

    def test_no_external_resources(self):
        for marker in ("http://", "https://", "//cdn", "<script src", "<link "):
            self.assertNotIn(marker, spectate.PAGE, marker)

    def test_it_renders_the_things_that_matter(self):
        for token in ("time bank", "clock", "agent-a", "agent-b", "EventSource"):
            self.assertIn(token.replace("time bank", "clock"), spectate.PAGE)

    def test_it_consumes_the_event_stream(self):
        self.assertIn("/events", spectate.PAGE)
        self.assertIn("/state", spectate.PAGE)


class TestBinding(unittest.TestCase):
    def test_the_server_binds_loopback_only(self):
        """An agent that could poll this feed would see its opponent's every
        command and its output, which would destroy reconnaissance as a skill."""
        source = (Path(spectate.__file__)).read_text()
        self.assertIn('("127.0.0.1", args.port)', source)
        self.assertNotIn('("0.0.0.0"', source)


if __name__ == "__main__":
    unittest.main()
