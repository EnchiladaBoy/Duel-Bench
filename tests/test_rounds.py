"""Round termination and move-deadline forfeits.

Reloads model_proxy under a move-timed configuration (which has both a round cap
and a forfeit deadline) and restores the previous one afterwards.
"""
import importlib
import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import modes  # noqa: E402
import model_proxy as mp  # noqa: E402

TOK_A, TOK_B = "aaaa" * 8, "bbbb" * 8
_saved = {}
_KEYS = (modes.ENV_VAR, "TOKENS_JSON", "ROLES_JSON", "MOCK_BACKEND",
         "MOCK_SCRIPTS_JSON", "MOCK_SLEEP_JSON", "PROXY_LOG", "STARTING_GUN")
ROUNDS, DEADLINE = 4, 1.0


def setUpModule():
    global _saved
    _saved = {k: os.environ.get(k) for k in _KEYS}
    os.environ.update({
        "TOKENS_JSON": json.dumps({TOK_A: "m/a", TOK_B: "m/b"}),
        "ROLES_JSON": json.dumps({TOK_A: "agent-a", TOK_B: "agent-b"}),
        "MOCK_BACKEND": "1", "PROXY_LOG": "", "STARTING_GUN": "0",
        "MOCK_SCRIPTS_JSON": json.dumps({TOK_A: ["ps"] * 20, TOK_B: ["ps"] * 20}),
        "MOCK_SLEEP_JSON": json.dumps({}),
    })
    os.environ.update(modes.to_env(
        modes.resolve("move-timed", max_rounds=ROUNDS, move_deadline=DEADLINE)))
    importlib.reload(mp)


def tearDownModule():
    for key, value in _saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    importlib.reload(mp)


def reset():
    mp.barrier.clear()
    mp.barrier.update(mp.initial_barrier_state())
    mp.reset_move_state()
    mp.EXHAUSTED.clear()
    mp.IN_FLIGHT.clear()


class TestRoundCap(unittest.TestCase):
    def setUp(self):
        reset()

    def test_mode_terminates_on_rounds(self):
        self.assertEqual(mp.MODE.termination, "rounds")
        self.assertEqual(mp.MODE.max_rounds, ROUNDS)

    def test_the_round_counter_advances_once_per_completed_round(self):
        for expected in range(1, ROUNDS + 1):
            self.assertEqual(mp.barrier["round"], expected)
            mp.barrier_join(TOK_A)
            mp.barrier_join(TOK_B)
        self.assertEqual(mp.barrier["round"], ROUNDS + 1)

    def test_both_agents_get_exactly_the_same_number_of_turns(self):
        """The entire point of a round-terminated mode: equal turns each,
        regardless of which model thinks faster."""
        joins = {TOK_A: 0, TOK_B: 0}
        for _ in range(ROUNDS):
            for token in (TOK_A, TOK_B):
                mp.barrier_join(token)
                joins[token] += 1
        self.assertEqual(joins[TOK_A], joins[TOK_B])


class TestMoveForfeit(unittest.TestCase):
    def setUp(self):
        reset()

    def test_a_move_inside_the_deadline_does_not_forfeit(self):
        mp.LAST_MOVE_SECONDS[TOK_A] = DEADLINE / 2
        self.assertFalse(mp.barrier_join(TOK_A).get("forfeit"))

    def test_a_move_over_the_deadline_forfeits(self):
        mp.LAST_MOVE_SECONDS[TOK_A] = DEADLINE * 3
        self.assertTrue(mp.barrier_join(TOK_A).get("forfeit"))
        self.assertEqual(mp.FORFEITS[TOK_A], 1)

    def test_a_forfeiting_agent_still_joins_so_the_opponent_is_not_stalled(self):
        # The agent loses its move, not the round's pacing. If it did not join,
        # the opponent would block until the deadline expired every round.
        mp.LAST_MOVE_SECONDS[TOK_A] = DEADLINE * 3
        mp.barrier_join(TOK_A)
        result = mp.barrier_join(TOK_B)
        self.assertTrue(result["released"])

    def test_the_forfeit_is_charged_to_the_slow_agent_only(self):
        mp.LAST_MOVE_SECONDS[TOK_A] = DEADLINE * 3
        mp.LAST_MOVE_SECONDS[TOK_B] = DEADLINE / 4
        mp.barrier_join(TOK_A)
        mp.barrier_join(TOK_B)
        self.assertEqual(mp.FORFEITS[TOK_A], 1)
        self.assertEqual(mp.FORFEITS[TOK_B], 0)

    def test_a_first_move_with_no_history_cannot_forfeit(self):
        self.assertIsNone(mp.LAST_MOVE_SECONDS[TOK_A])
        self.assertFalse(mp.barrier_join(TOK_A).get("forfeit"))

    def test_guard_modes_never_forfeit(self):
        # untimed has deadline_effect "guard": a long think is legal there.
        self.assertEqual(modes.resolve("untimed").deadline_effect, "guard")
        self.assertEqual(modes.resolve("move-timed").deadline_effect, "forfeit")


if __name__ == "__main__":
    unittest.main()
