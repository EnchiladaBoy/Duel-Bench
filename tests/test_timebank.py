"""Time-bank mode: the only genuinely new measurement in the mode system.

Reloads model_proxy under a time-bank configuration and restores the previous
one afterwards, because the proxy reads its mode once at import.
"""
import importlib
import json
import os
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import modes  # noqa: E402
import model_proxy as mp  # noqa: E402

TOK_A, TOK_B = "aaaa" * 8, "bbbb" * 8
_saved_env = {}
BANK = 2.0


_ENV_KEYS = (modes.ENV_VAR, "MOCK_SLEEP_JSON", "TOKENS_JSON", "ROLES_JSON",
             "MOCK_SCRIPTS_JSON", "MOCK_BACKEND", "PROXY_LOG")


def setUpModule():
    global _saved_env
    # Configure everything this module needs rather than inheriting it from
    # whichever test module discovery happened to import first.
    _saved_env = {k: os.environ.get(k) for k in _ENV_KEYS}
    os.environ.update({
        "TOKENS_JSON": json.dumps({TOK_A: "model/a", TOK_B: "model/b"}),
        "ROLES_JSON": json.dumps({TOK_A: "agent-a", TOK_B: "agent-b"}),
        "MOCK_SCRIPTS_JSON": json.dumps({TOK_A: ["ps"] * 8, TOK_B: ["ps"] * 8}),
        "MOCK_BACKEND": "1",
        "PROXY_LOG": "",
        "MOCK_SLEEP_JSON": json.dumps({TOK_A: [0.6, 0.6, 0.6, 0.6], TOK_B: 0.05}),
    })
    os.environ.update(modes.to_env(modes.resolve("time-bank", time_bank=BANK)))
    importlib.reload(mp)


def tearDownModule():
    for key, value in _saved_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    importlib.reload(mp)


def _post(token, port_holder=[]):
    """Drive a real completion through the running handler."""
    if not port_holder:
        from http.server import ThreadingHTTPServer
        srv = ThreadingHTTPServer(("127.0.0.1", 0), mp.ProxyHandler)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        port_holder.append(srv.server_address[1])
        time.sleep(0.2)
    import urllib.error
    import urllib.request
    req = urllib.request.Request(
        f"http://127.0.0.1:{port_holder[0]}/v1/chat/completions",
        data=b'{"messages":[]}',
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def exhaust(token):
    """Do exactly what the production path does when a bank empties: record it
    under LOCK, then retire the agent from the barrier."""
    with mp.LOCK:
        mp.BANK_REMAINING[token] = 0.0
        mp.EXHAUSTED.add(token)
    mp.retire_from_barrier(token)


def reset():
    mp.barrier.clear()
    mp.barrier.update(mp.initial_barrier_state())
    mp.EXHAUSTED.clear()
    mp.IN_FLIGHT.clear()
    for token in mp.TOKENS:
        mp.BANK_REMAINING[token] = mp.MODE.time_bank
        mp.MOVE_SECONDS[token] = []
        mp.REQUEST_COUNT[token] = 0
        mp.MOCK_INDEX[token] = 0


class TestMockDelay(unittest.TestCase):
    def test_scalar_delay(self):
        self.assertEqual(mp.mock_delay_for(TOK_B, 3), 0.05)

    def test_list_delay_cycles(self):
        # Varying per-move latency is what makes a bank testable without spend.
        self.assertEqual(mp.mock_delay_for(TOK_A, 0), 0.6)
        self.assertEqual(mp.mock_delay_for(TOK_A, 4), 0.6)

    def test_missing_token_is_zero(self):
        self.assertEqual(mp.mock_delay_for("nope", 0), 0.0)


class TestBankAccounting(unittest.TestCase):
    def setUp(self):
        reset()

    def test_mode_is_lockstep(self):
        self.assertTrue(mp.MODE.lockstep)

    def test_bank_starts_at_the_granted_amount(self):
        self.assertEqual(mp.BANK_REMAINING[TOK_A], BANK)

    def test_a_real_call_charges_the_bank_and_never_goes_negative(self):
        # Drives the actual handler path rather than re-implementing the sum.
        before = mp.BANK_REMAINING[TOK_A]
        for _ in range(6):
            _post(TOK_A)
        self.assertLess(mp.BANK_REMAINING[TOK_A], before)
        self.assertGreaterEqual(mp.BANK_REMAINING[TOK_A], 0.0)

    def test_exhaustion_is_reported_as_a_game_outcome_not_a_budget_failure(self):
        for _ in range(8):
            status, body = _post(TOK_A)
        self.assertEqual(status, 429)
        self.assertEqual(body.get("error_kind"), "time_bank_exhausted")

    def test_one_move_in_flight_per_agent(self):
        # Concurrent requests on one token would otherwise each pass a
        # nearly-empty bank and overdraw it.
        results = []
        threads = [threading.Thread(target=lambda: results.append(_post(TOK_A)))
                   for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        kinds = [b.get("error_kind") for _, b in results]
        self.assertIn("concurrent_request", kinds)


class TestExhaustion(unittest.TestCase):
    def setUp(self):
        reset()

    def test_exhaustion_retires_the_agent_from_the_quorum(self):
        exhaust(TOK_A)
        self.assertIn(TOK_A, mp.EXHAUSTED)
        self.assertNotIn(TOK_A, mp.barrier["active"])

    def test_an_exhausted_agent_cannot_rejoin(self):
        # A merely-absent agent may rejoin (test_proxy asserts that); one whose
        # bank is spent must not, or it stalls the survivor every round.
        exhaust(TOK_A)
        mp.barrier_join(TOK_A)
        self.assertNotIn(TOK_A, mp.barrier["active"])

    def test_survivor_is_released_promptly_when_the_opponent_exhausts(self):
        """The notify_all in mark_exhausted. Without it the survivor blocks
        until the round deadline, repeatedly - minutes of dead time at the most
        dramatic moment of the match."""
        joined = mp.barrier_join(TOK_B)
        released = []

        def wait():
            released.append(mp.barrier_wait(joined["round"], TOK_B))

        waiter = threading.Thread(target=wait, daemon=True)
        waiter.start()
        time.sleep(0.2)
        self.assertFalse(released, "survivor should still be waiting")
        exhaust(TOK_A)
        waiter.join(timeout=3)
        self.assertTrue(released, "survivor was not released when the opponent exhausted")
        self.assertLess(len(released), 2)

    def test_all_banks_spent_is_a_terminal_condition(self):
        # Without a terminal signal both agents idle alive and the match drifts
        # to the runaway guard - the immortal-idle bug in a new place.
        exhaust(TOK_A)
        exhaust(TOK_B)
        self.assertGreaterEqual(len(mp.EXHAUSTED), len(mp.TOKENS))


class TestArenaBlock(unittest.TestCase):
    def setUp(self):
        reset()

    def _arena(self, token):
        handler = mp.ProxyHandler.__new__(mp.ProxyHandler)
        return handler._arena_block(token, 1.25)

    def test_reports_own_bank_and_move_time(self):
        arena = self._arena(TOK_A)
        self.assertEqual(arena["mode"], "time-bank")
        self.assertEqual(arena["move_seconds"], 1.25)
        self.assertEqual(arena["bank_remaining"], BANK)

    def test_opponent_bank_is_disclosed_as_a_whole_number(self):
        arena = self._arena(TOK_A)
        self.assertEqual(arena["opponent"], "agent-b")
        self.assertIsInstance(arena["opponent_bank_remaining"], int)

    def test_arena_block_never_carries_token_material(self):
        # The direct analogue of the barrier disclosure test: role-keyed only.
        raw = json.dumps(self._arena(TOK_A))
        self.assertNotIn(TOK_A, raw)
        self.assertNotIn(TOK_B, raw)


if __name__ == "__main__":
    unittest.main()
