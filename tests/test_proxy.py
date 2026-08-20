"""Regression tests for the model proxy: credential disclosure, the lockstep
barrier, and request validation."""
import json
import os
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

SRC = str(Path(__file__).resolve().parent.parent / "src")
sys.path.insert(0, SRC)

import modes  # noqa: E402

TOK_A, TOK_B = "aaaa" * 8, "bbbb" * 8
# Configure the proxy the way the orchestrator does: one resolved Mode, shipped
# whole, rather than a scatter of individual env scalars that can drift apart.
TEST_MODE = modes.resolve("untimed", max_requests=3, move_deadline=2)
os.environ.update({
    "TOKENS_JSON": json.dumps({TOK_A: "model/a", TOK_B: "model/b"}),
    "ROLES_JSON": json.dumps({TOK_A: "agent-a", TOK_B: "agent-b"}),
    "PROXY_LOG": "",
    "MOCK_BACKEND": "1",
    "MOCK_SCRIPTS_JSON": json.dumps({TOK_A: ["ps"], TOK_B: ["ps"]}),
    "MAX_TOKENS_PER_CALL": "1024",
    "CONTROL_TOKEN": "cccc" * 8,
})
os.environ.update(modes.to_env(TEST_MODE))
import model_proxy as mp  # noqa: E402


def reset_barrier():
    mp.barrier.clear()
    mp.barrier.update(mp.initial_barrier_state())


class TestNoTokenDisclosure(unittest.TestCase):
    """Audit C2: /health returned a dict KEYED BY BEARER TOKEN with no auth,
    and the barrier echoed both tokens to both agents."""

    def test_barrier_join_returns_roles_not_tokens(self):
        reset_barrier()
        out = json.dumps(mp.barrier_join(TOK_A))
        self.assertIn("agent-a", out)
        self.assertNotIn(TOK_A, out)
        self.assertNotIn(TOK_B, out)

    def test_barrier_release_record_holds_no_tokens(self):
        reset_barrier()
        mp.barrier_join(TOK_A)
        mp.barrier_join(TOK_B)
        out = json.dumps(mp.barrier["releases"])
        self.assertNotIn(TOK_A, out)
        self.assertNotIn(TOK_B, out)
        self.assertIn("agent-b", out)


class TestBarrier(unittest.TestCase):
    def setUp(self):
        reset_barrier()

    def test_round_releases_when_both_join(self):
        self.assertFalse(mp.barrier_join(TOK_A)["released"])
        second = mp.barrier_join(TOK_B)
        self.assertTrue(second["released"])
        self.assertTrue(second["both"])

    def test_double_join_by_one_agent_cannot_release(self):
        mp.barrier_join(TOK_A)
        self.assertFalse(mp.barrier_join(TOK_A)["released"])

    def test_wait_on_a_future_round_is_rejected(self):
        """Audit H1: the wait path used to CREATE a release record for any
        round a client named, so one curl loop pre-released every future round
        and turned --fair back into a free-for-all."""
        _, status = mp.barrier_wait(9999, TOK_A)
        self.assertEqual(status, 400)
        self.assertNotIn(9999, mp.barrier["releases"])

    def test_wait_without_joining_is_rejected(self):
        """Audit H1: waiting without joining left first_join None, so the
        deadline was rebased every pass and the handler thread blocked forever."""
        _, status = mp.barrier_wait(1, TOK_A)
        self.assertEqual(status, 400)

    def test_partial_release_fires_on_timeout(self):
        mp.barrier_join(TOK_A)
        start = time.time()
        result, status = mp.barrier_wait(1, TOK_A)
        self.assertEqual(status, 200)
        self.assertLess(time.time() - start, mp.ROUND_TIMEOUT + 2)
        self.assertFalse(result["both"])
        self.assertEqual(result["joined"], ["agent-a"])

    def test_absent_agent_drops_out_of_the_quorum(self):
        """Audit H1: a survivor used to pay the full --move-timeout every round
        forever once the opponent stopped participating."""
        waits = []
        for _ in range(mp.MAX_MISSED_ROUNDS + 2):
            joined = mp.barrier_join(TOK_A)
            start = time.time()
            mp.barrier_wait(joined["round"], TOK_A)
            waits.append(time.time() - start)
        self.assertGreater(waits[0], 1.0)
        self.assertLess(waits[-1], 0.5, f"survivor still stalling: {waits}")

    def test_returning_agent_rejoins_the_quorum(self):
        for _ in range(mp.MAX_MISSED_ROUNDS + 1):
            joined = mp.barrier_join(TOK_A)
            mp.barrier_wait(joined["round"], TOK_A)
        self.assertNotIn(TOK_B, mp.barrier["active"])
        mp.barrier_join(TOK_B)
        self.assertIn(TOK_B, mp.barrier["active"])

    def test_release_history_is_bounded(self):
        for _ in range(mp.KEEP_RELEASES + 50):
            mp.barrier_join(TOK_A)
            mp.barrier_join(TOK_B)
        self.assertLessEqual(len(mp.barrier["releases"]), mp.KEEP_RELEASES)

    def test_concurrent_joins_release_exactly_once(self):
        results = []
        def join(tok):
            results.append(mp.barrier_join(tok))
        threads = [threading.Thread(target=join, args=(t,)) for t in (TOK_A, TOK_B)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(sum(1 for r in results if r["released"]), 1)


class TestUpstreamPayload(unittest.TestCase):
    """Audit H9: the proxy copied the client's body wholesale, so an agent with
    curl could smuggle sampling and routing parameters, self-boost, and drain
    the key far past what --max-requests implies."""

    def test_disallowed_keys_are_dropped(self):
        body = {"messages": [], "n": 8, "temperature": 2.0, "provider": {"order": ["x"]},
                "reasoning": {"effort": "high"}, "transforms": ["y"], "logit_bias": {}}
        payload = mp.build_upstream_payload(body, "model/a")
        for key in ("provider", "reasoning", "transforms", "logit_bias", "temperature"):
            self.assertNotIn(key, payload)

    def test_n_is_forced_to_one(self):
        self.assertEqual(
            mp.build_upstream_payload({"messages": [], "n": 8}, "m")["n"], 1)

    def test_max_tokens_is_clamped(self):
        payload = mp.build_upstream_payload({"messages": [], "max_tokens": 999999}, "m")
        self.assertEqual(payload["max_tokens"], mp.MAX_TOKENS_PER_CALL)

    def test_garbage_max_tokens_falls_back_to_the_cap(self):
        payload = mp.build_upstream_payload({"messages": [], "max_tokens": "lots"}, "m")
        self.assertEqual(payload["max_tokens"], mp.MAX_TOKENS_PER_CALL)

    def test_model_is_pinned_by_the_proxy(self):
        payload = mp.build_upstream_payload({"messages": [], "model": "evil/model"}, "m")
        self.assertEqual(payload["model"], "m")

    def test_messages_and_tools_pass_through(self):
        body = {"messages": [{"role": "user", "content": "hi"}], "tools": [{"t": 1}]}
        payload = mp.build_upstream_payload(body, "m")
        self.assertEqual(payload["messages"], body["messages"])
        self.assertEqual(payload["tools"], body["tools"])


class TestHTTPSurface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from http.server import ThreadingHTTPServer
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), mp.ProxyHandler)
        cls.server.daemon_threads = True
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        time.sleep(0.2)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def _request(self, path, data=None, token=None, method=None):
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=data,
            headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_health_needs_no_auth_but_discloses_nothing(self):
        status, body = self._request("/health")
        self.assertEqual(status, 200)
        raw = json.dumps(body)
        self.assertNotIn(TOK_A, raw)
        self.assertNotIn(TOK_B, raw)
        self.assertNotIn("requests_served", body)

    def test_completions_require_a_known_token(self):
        status, _ = self._request(
            "/v1/chat/completions", b'{"messages":[]}', token="not-a-token", method="POST")
        self.assertEqual(status, 401)

    def test_non_object_body_is_rejected_before_the_budget(self):
        """Audit M3: a JSON array passed the parse check, spent a budget slot,
        triggered a real billable upstream call, then crashed the handler."""
        before = mp.REQUEST_COUNT[TOK_A]
        status, _ = self._request("/v1/chat/completions", b"[]", token=TOK_A, method="POST")
        self.assertEqual(status, 400)
        self.assertEqual(mp.REQUEST_COUNT[TOK_A], before)

    def test_missing_messages_is_rejected(self):
        status, _ = self._request(
            "/v1/chat/completions", b'{"foo":1}', token=TOK_A, method="POST")
        self.assertEqual(status, 400)

    def test_bad_content_length_does_not_kill_the_handler(self):
        import http.client
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.putrequest("POST", "/v1/chat/completions")
        conn.putheader("Authorization", f"Bearer {TOK_A}")
        conn.putheader("Content-Length", "not-a-number")
        conn.endheaders()
        self.assertEqual(conn.getresponse().status, 400)
        conn.close()
        # the server is still serving
        self.assertEqual(self._request("/health")[0], 200)

    def test_oversized_body_is_refused(self):
        import http.client
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.putrequest("POST", "/v1/chat/completions")
        conn.putheader("Authorization", f"Bearer {TOK_A}")
        conn.putheader("Content-Length", str(mp.MAX_BODY_BYTES + 1))
        conn.endheaders()
        self.assertEqual(conn.getresponse().status, 413)
        conn.close()

    def test_control_status_requires_the_control_token(self):
        """Audit C2 generalised: the proxy is reachable by both agents, so any
        endpoint disclosing match state must be gated on a token that never
        enters an agent container."""
        self.assertEqual(self._request("/control/status")[0], 401)
        self.assertEqual(self._request("/control/status", token=TOK_A)[0], 401)

    def test_control_status_discloses_no_token_material(self):
        status, body = self._request("/control/status", token="cccc" * 8)
        self.assertEqual(status, 200)
        raw = json.dumps(body)
        self.assertNotIn(TOK_A, raw)
        self.assertNotIn(TOK_B, raw)
        self.assertIn("agent-a", json.dumps(body.get("requests")))

    def test_control_status_reports_no_terminal_until_a_mode_declares_one(self):
        _, body = self._request("/control/status", token="cccc" * 8)
        self.assertIsNone(body["terminal"])

    def test_budget_exhaustion_is_tagged_as_ours(self):
        """Audit H2: the harness must be able to tell the proxy's own budget
        rejection from an upstream provider rate-limit, or one transient
        throttle permanently benches the agent."""
        mp.REQUEST_COUNT[TOK_B] = mp.MAX_REQUESTS
        status, body = self._request(
            "/v1/chat/completions", b'{"messages":[]}', token=TOK_B, method="POST")
        self.assertEqual(status, 429)
        self.assertEqual(body.get("error_kind"), "proxy_budget")


if __name__ == "__main__":
    unittest.main()
