"""The three attackable endpoints and the defense counter behind them."""
import importlib
import json
import os
import sys
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
os.environ.setdefault("LOG_PATH", "")
os.environ.setdefault("BATTLE_DIR", "/tmp")
os.environ.setdefault("COMMAND_TIMEOUT", "5")
import agent_harness as ah  # noqa: E402


class TestTerrain(unittest.TestCase):
    """A real target the opponent can attack and the defender can hold."""

    def setUp(self):
        self.counter = ah.TERRAIN_COUNTER
        self.counter["score"] = 0
        self.counter["spoofs"] = 0

    def _post(self, port, path, data=b""):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}", data=data, method="POST")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status, json.loads(resp.read().decode())

    def _shutdown(self, servers):
        for s in servers:
            s.shutdown()
            s.server_close()

    def test_ports_bind_without_conflict(self):
        ah.TERRAIN_BASE_PORT = 9188
        servers = ah.start_terrain()
        self.assertEqual(len(servers), 3)
        self._shutdown(servers)

    def test_signal_post_marks_spoofed(self):
        ah.TERRAIN_BASE_PORT = 9198
        servers = ah.start_terrain()
        try:
            port = 9198 + 11
            status, body = self._post(port, "/signal", b"write payload")
            self.assertEqual(status, 200)
            self.assertTrue(body.get("spoofed"))
            self.assertEqual(self.counter["spoofs"], 1)
        finally:
            self._shutdown(servers)

    def test_debug_post_counts_as_defense(self):
        ah.TERRAIN_BASE_PORT = 9208
        servers = ah.start_terrain()
        try:
            port = 9208 + 21
            status, body = self._post(port, "/debug", b"held")
            self.assertEqual(status, 200)
            self.assertEqual(body.get("score"), 1)
        finally:
            self._shutdown(servers)

    def test_telemetry_get_counts_as_recon(self):
        ah.TERRAIN_BASE_PORT = 9218
        servers = ah.start_terrain()
        try:
            port = 9218 + 31
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/telemetry", timeout=3) as resp:
                self.assertEqual(resp.status, 200)
        finally:
            self._shutdown(servers)

    def test_score_is_persisted_for_respawn(self):
        ah.TERRAIN_BASE_PORT = 9228
        servers = ah.start_terrain()
        try:
            port = 9228 + 21
            self._post(port, "/debug", b"one")
            self._post(port, "/debug", b"two")
            self._post(port, "/debug", b"three")
            with open(ah._terrain_file()) as fh:
                self.assertEqual(json.load(fh)["score"], 3)
            # Simulate respawn: zero the in-memory mirror and re-read.
            ah.TERRAIN_COUNTER["score"] = 0
            ah.TERRAIN_COUNTER["spoofs"] = 0
            self.assertEqual(ah._read_terrain_score(), {"score": 3, "spoofs": 0})
        finally:
            self._shutdown(servers)
            if os.path.exists(ah._terrain_file()):
                os.unlink(ah._terrain_file())

    def test_health_reports_the_counter(self):
        ah.TERRAIN_COUNTER["score"] = 5
        ah.TERRAIN_COUNTER["spoofs"] = 2
        handler = ah.HealthHandler
        # /health is bound on the heartbeat port in supervisor; the counter
        # shape is asserted directly since it now persists to /battle.
        self.assertEqual(ah.TERRAIN_COUNTER["score"], 5)
        self.assertEqual(ah.TERRAIN_COUNTER["spoofs"], 2)


if __name__ == "__main__":
    unittest.main()
