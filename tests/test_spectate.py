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

    def test_first_paint_needs_no_round_trip(self):
        """The state is INLINED into the page, not fetched. Fetching raced the
        first paint, so a viewer opening mid-match saw an empty board until the
        stream caught up - verified by screenshotting the rendered page."""
        self.assertIn("/*BOOTSTRAP*/null", spectate.PAGE)
        self.assertNotIn('fetch("/state")', spectate.PAGE)

    def test_the_inlined_state_cannot_close_the_script_element(self):
        # Match text is agent-controlled and lands in this JSON.
        import inspect
        self.assertIn('replace("</"', inspect.getsource(spectate.Spectator))


class TestBinding(unittest.TestCase):
    def test_the_server_binds_loopback_only(self):
        """An agent that could poll this feed would see its opponent's every
        command and its output, which would destroy reconnaissance as a skill."""
        source = (Path(spectate.__file__)).read_text()
        self.assertIn('("127.0.0.1", args.port)', source)
        self.assertNotIn('("0.0.0.0"', source)


if __name__ == "__main__":
    unittest.main()


class TestForgedEventsNeverReachTheBrowser(unittest.TestCase):
    """The Python reducer refuses arena-only events from the wrong source, but
    the browser applies the SSE stream directly - so a forged match_end would
    have ended the spectator's match even though the terminal viewer was immune.
    Filtered server-side so the rule lives in one place, not also in JavaScript.
    """

    @staticmethod
    def allowed(event, src):
        import watch
        return watch.MatchState.trusted(event, src)

    def test_an_agent_cannot_end_the_browser_match(self):
        self.assertFalse(self.allowed("match_end", "agent-a"))

    def test_an_agent_cannot_forge_a_kill_or_a_scoreboard(self):
        for event in ("agent_down", "snapshot", "arena_ready"):
            self.assertFalse(self.allowed(event, "agent-b"), event)

    def test_an_agent_cannot_forge_proxy_state(self):
        for event in ("go", "move_start", "bank_exhausted", "barrier_release"):
            self.assertFalse(self.allowed(event, "agent-a"), event)

    def test_the_arena_is_still_believed(self):
        self.assertTrue(self.allowed("match_end", "orchestrator"))
        self.assertTrue(self.allowed("move_start", "proxy"))

    def test_agents_are_still_believed_about_their_own_actions(self):
        # What an agent DID is exactly what it is entitled to report, and the
        # feed would be empty without it.
        for event in ("command_start", "command_result", "pass", "idle"):
            self.assertTrue(self.allowed(event, "agent-a"), event)

    def test_the_filter_is_applied_on_the_wire(self):
        source = Path(spectate.__file__).read_text()
        self.assertIn("watch.MatchState.trusted", source)

    def test_the_rule_is_not_duplicated_in_javascript(self):
        # Two implementations of the rules is exactly the drift that a single
        # reducer exists to prevent.
        self.assertNotIn("ARENA_ONLY", spectate.PAGE)


class TestFeedTimestamps(unittest.TestCase):
    """Feed entries used to be bare strings, so a viewer that joined late - or
    replayed - showed every line at 0.0s, because the timestamp only ever
    existed in the renderer's local variable."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "events.jsonl"
        write_stream(self.path, EVENTS)

    def tearDown(self):
        self.tmp.cleanup()

    def test_each_entry_carries_its_match_time(self):
        feed = spectate.snapshot(self.path)["feed"]
        self.assertTrue(feed)
        for entry in feed:
            self.assertIn("t", entry)
            self.assertIn("text", entry)

    def test_times_are_not_all_zero(self):
        times = [e["t"] for e in spectate.snapshot(self.path)["feed"]]
        self.assertGreater(max(times), 0.0)

    def test_times_do_not_run_backwards(self):
        times = [e["t"] for e in spectate.snapshot(self.path)["feed"]]
        self.assertEqual(times, sorted(times))


class TestNoDuplicatedFeed(unittest.TestCase):
    """The page is served with its state inlined, and the stream then replays
    from the beginning - so without a resume point the viewer sees the entire
    match twice. Found by looking at the rendered page: the second copy was
    recognisable because the JavaScript adds emoji the Python feed does not."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "events.jsonl"
        write_stream(self.path, EVENTS)

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_snapshot_says_where_it_ends(self):
        self.assertEqual(spectate.snapshot(self.path)["last_seq"],
                         max(e["seq"] for e in EVENTS))

    def test_the_page_resumes_the_stream_after_the_inlined_state(self):
        self.assertIn('"/events?from="', spectate.PAGE)

    def test_an_empty_stream_still_reports_a_resume_point(self):
        empty = Path(self.tmp.name) / "empty.jsonl"
        empty.write_text("")
        self.assertEqual(spectate.snapshot(empty)["last_seq"], 0)

    def test_the_bootstrap_feed_has_no_repeats(self):
        import collections
        feed = [e["text"] for e in spectate.snapshot(self.path)["feed"]]
        self.assertEqual([t for t, n in collections.Counter(feed).items() if n > 1], [])
