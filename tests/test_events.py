"""The live event stream: one ordered host-side timeline merged from three
sources whose clocks are unrelated."""
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import orchestrator as orch  # noqa: E402


class TestEventStream(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "events.jsonl"
        self.stream = orch.EventStream(self.path)

    def tearDown(self):
        self.stream.close(linger=0.5)
        self.tmp.cleanup()

    def _records(self):
        self.stream.close(linger=1.0)
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]

    def test_seq_is_strictly_increasing_across_sources(self):
        for i in range(20):
            self.stream.emit("agent-a" if i % 2 else "proxy", "tick", i=i)
        seqs = [r["seq"] for r in self._records()]
        self.assertEqual(seqs, sorted(seqs))
        self.assertEqual(len(set(seqs)), len(seqs))

    def test_every_record_carries_the_envelope(self):
        self.stream.emit("proxy", "hello", extra=1)
        record = self._records()[0]
        for key in ("v", "seq", "t", "src", "event"):
            self.assertIn(key, record)
        self.assertEqual(record["src"], "proxy")
        self.assertEqual(record["event"], "hello")

    def test_t_is_relative_to_match_start(self):
        self.stream.emit("proxy", "a")
        time.sleep(0.05)
        self.stream.emit("proxy", "b")
        a, b = self._records()
        self.assertLess(a["t"], b["t"])
        self.assertLess(a["t"], 1.0)

    def test_ingest_normalises_the_harness_kind_field(self):
        # The harness calls its type "kind", the proxy calls it "event".
        self.stream._ingest("agent-a", json.dumps({"kind": "command_start", "step": 3}))
        record = self._records()[0]
        self.assertEqual(record["event"], "command_start")
        self.assertEqual(record["step"], 3)

    def test_source_clock_is_kept_but_never_used_for_ordering(self):
        # Three containers, three unrelated wall clocks. The source timestamp is
        # preserved for skew analysis only.
        self.stream._ingest("proxy", json.dumps({"event": "x", "ts": 1234.5}))
        record = self._records()[0]
        self.assertEqual(record["src_ts"], 1234.5)
        self.assertNotEqual(record["t"], 1234.5)

    def test_a_source_cannot_overwrite_the_envelope(self):
        # An agent controls the bytes on its own stdout, so it must not be able
        # to forge its src, its ordering, or another agent's identity.
        self.stream._ingest("agent-a", json.dumps(
            {"kind": "x", "src": "agent-b", "seq": 999999, "t": -1, "v": 99}))
        record = self._records()[0]
        self.assertEqual(record["src"], "agent-a")
        self.assertEqual(record["seq"], 1)
        self.assertGreaterEqual(record["t"], 0)

    def test_malformed_lines_are_skipped_not_fatal(self):
        self.stream._ingest("agent-a", "not json at all")
        self.stream._ingest("agent-a", "")
        self.stream._ingest("agent-a", json.dumps({"kind": "ok"}))
        records = self._records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["event"], "ok")

    def test_overflow_is_counted_not_silently_lost(self):
        self.stream.queue.maxsize = 1
        self.stream._stop.set()      # stall the writer so the queue fills
        time.sleep(0.3)
        for _ in range(50):
            self.stream.emit("proxy", "flood")
        self.assertGreater(self.stream.dropped, 0)


class TestFileTailer(unittest.TestCase):
    """The proxy log is tailed live from its host bind mount."""

    def test_tailer_advances_past_each_line(self):
        tmp = tempfile.TemporaryDirectory()
        source = Path(tmp.name) / "proxy.jsonl"
        source.write_text(json.dumps({"event": "proxy_start"}) + "\n")
        stream = orch.EventStream(Path(tmp.name) / "events.jsonl")
        stream.follow_file(source, "proxy")
        time.sleep(0.8)
        with source.open("a") as fh:
            fh.write(json.dumps({"event": "completion", "agent": "agent-a"}) + "\n")
            fh.flush()
        time.sleep(0.8)
        stream.close(linger=1.0)
        events = [json.loads(l)["event"]
                  for l in (Path(tmp.name) / "events.jsonl").read_text().splitlines()]
        # tell() is disabled inside a for-loop over a text file, which made an
        # earlier version re-read line 1 on every pass.
        self.assertEqual(events.count("proxy_start"), 1)
        self.assertEqual(events.count("completion"), 1)
        tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
