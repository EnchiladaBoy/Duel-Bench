"""Regression tests for the leaderboard. Every test here maps to an audit
finding; the scoring is the project's only published output, so it is the part
that most needs executable evidence."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import elo  # noqa: E402


def write_match(root, name, **fields):
    data = {"model_a": "a/one", "model_b": "b/two", "winner": "agent-a",
            "mock": False, "pid_shared": True, "network_internal": True,
            "mode": "untimed"}
    data.update(fields)
    d = Path(root) / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps(data))


class TestExpectedScore(unittest.TestCase):
    def test_equal_ratings_is_even(self):
        self.assertAlmostEqual(elo.expected_score(1500, 1500), 0.5)

    def test_higher_rating_favoured(self):
        self.assertGreater(elo.expected_score(1700, 1500), 0.5)

    def test_symmetry(self):
        self.assertAlmostEqual(
            elo.expected_score(1600, 1400) + elo.expected_score(1400, 1600), 1.0)


class TestSelfPlay(unittest.TestCase):
    """Audit H6: the loser update used to read a stale rating and overwrite the
    winner update, so a model that beat itself dropped to 1484."""

    def test_self_play_is_a_no_op(self):
        results = [{"model_a": "m", "model_b": "m", "winner": "agent-a"}]
        ratings, games = elo.rate(results, k=32.0)
        self.assertAlmostEqual(ratings["m"], 1500.0)
        self.assertEqual(games["m"], 2)

    def test_self_play_draw_is_also_a_no_op(self):
        results = [{"model_a": "m", "model_b": "m", "winner": "draw"}]
        ratings, _ = elo.rate(results, k=32.0)
        self.assertAlmostEqual(ratings["m"], 1500.0)

    def test_normal_match_is_zero_sum(self):
        results = [{"model_a": "x", "model_b": "y", "winner": "agent-a"}]
        ratings, _ = elo.rate(results, k=32.0)
        self.assertAlmostEqual(ratings["x"] + ratings["y"], 3000.0)
        self.assertAlmostEqual(ratings["x"], 1516.0)


class TestEligibility(unittest.TestCase):
    """Audit C1: elo.py rated anything it was handed, so the shipped
    leaderboard was topped by a hardcoded shell script."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def _load(self, **kw):
        return elo.load_results(self.root, **kw)

    def test_mock_matches_are_excluded(self):
        write_match(self.root, "m1", mock=True)
        rated, skipped = self._load()
        self.assertEqual(rated, [])
        self.assertIn("mock", skipped[0][1])

    def test_mock_can_be_opted_into(self):
        write_match(self.root, "m1", mock=True)
        rated, _ = self._load(include_mock=True)
        self.assertEqual(len(rated), 1)

    def test_unshared_pid_namespace_is_excluded(self):
        write_match(self.root, "m1", pid_shared=False)
        rated, skipped = self._load()
        self.assertEqual(rated, [])
        self.assertIn("PID namespace", skipped[0][1])

    def test_non_internal_network_is_excluded(self):
        write_match(self.root, "m1", network_internal=False)
        self.assertEqual(self._load()[0], [])

    def test_orchestrator_unrated_flag_is_honoured(self):
        write_match(self.root, "m1", rated=False, unrated_reason="no commands run")
        rated, skipped = self._load()
        self.assertEqual(rated, [])
        self.assertIn("no commands run", skipped[0][1])

    def test_error_and_aborted_outcomes_are_excluded(self):
        write_match(self.root, "m1", winner="error")
        write_match(self.root, "m2", winner="aborted")
        self.assertEqual(self._load()[0], [])

    def test_missing_arena_fields_do_not_exclude_older_results(self):
        # Absent != degraded: results written before these fields existed
        # should still rate, PROVIDED they say which mode they were played in.
        d = Path(self.root) / "old"
        d.mkdir()
        (d / "result.json").write_text(json.dumps(
            {"model_a": "a", "model_b": "b", "winner": "draw", "mode": "realtime"}))
        self.assertEqual(len(self._load()[0]), 1)

    def test_a_missing_mode_IS_exclusionary(self):
        """Deliberately unlike the arena fields above. An absent arena field
        means 'not recorded, probably fine'; an absent mode means we cannot know
        which leaderboard the result belongs on, and guessing would recreate the
        pooling that per-mode rating exists to prevent."""
        d = Path(self.root) / "premodes"
        d.mkdir()
        (d / "result.json").write_text(json.dumps(
            {"model_a": "a", "model_b": "b", "winner": "draw"}))
        rated, skipped = self._load()
        self.assertEqual(rated, [])
        self.assertIn("no mode recorded", skipped[0][1])
        # ...but recoverable on request, into their own pool.
        self.assertEqual(len(self._load(include_legacy=True)[0]), 1)

    def test_a_clean_match_is_rated(self):
        write_match(self.root, "m1")
        self.assertEqual(len(self._load()[0]), 1)


class TestOrdering(unittest.TestCase):
    """Audit M7/L4: ELO is order-dependent, so ordering must come from a
    recorded timestamp rather than from filesystem layout."""

    def test_utc_timestamp_beats_path_order(self):
        items = [
            ("zzz/result.json", {"started_at_utc": "2026-01-01T00:00:00+00:00"}),
            ("aaa/result.json", {"started_at_utc": "2026-06-01T00:00:00+00:00"}),
        ]
        ordered = sorted(items, key=elo.sort_key)
        self.assertEqual(ordered[0][0], "zzz/result.json")

    def test_falls_back_to_path_when_undated(self):
        items = [("b/result.json", {}), ("a/result.json", {})]
        self.assertEqual(sorted(items, key=elo.sort_key)[0][0], "a/result.json")


if __name__ == "__main__":
    unittest.main()


class TestModePartitioning(unittest.TestCase):
    """Modes impose different rules, so their ratings are not comparable. A
    single pooled number would be meaningless."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_modes_are_never_pooled(self):
        write_match(self.root, "m1", mode="untimed", winner="agent-a")
        write_match(self.root, "m2", mode="realtime", winner="agent-b")
        pools = elo.partition(elo.load_results(self.root)[0])
        self.assertEqual(sorted(pools), ["realtime", "untimed"])

    def test_a_model_playing_two_modes_gets_two_independent_ratings(self):
        write_match(self.root, "m1", mode="untimed", winner="agent-a")
        write_match(self.root, "m2", mode="realtime", winner="agent-b")
        pools = elo.partition(elo.load_results(self.root)[0])
        untimed, _ = elo.rate(pools["untimed"], k=32.0)
        realtime, _ = elo.rate(pools["realtime"], k=32.0)
        self.assertGreater(untimed["a/one"], 1500)
        self.assertLess(realtime["a/one"], 1500)

    def test_pool_of_falls_back_to_legacy(self):
        self.assertEqual(elo.pool_of({}), elo.LEGACY_POOL)
        self.assertEqual(elo.pool_of({"mode": "time-bank"}), "time-bank")

    def test_token_usage_is_reported_per_model(self):
        write_match(self.root, "m1", usage={"agent-a": {"total_tokens": 400},
                                            "agent-b": {"total_tokens": 200}})
        usage = elo.usage_per_model(elo.load_results(self.root)[0])
        self.assertEqual(usage["a/one"], 400)
        self.assertEqual(usage["b/two"], 200)
