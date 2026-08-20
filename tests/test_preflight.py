"""Pre-match checks. The pure parts are tested here with no network; the live
checks are exercised by running them, since they cost nothing."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import preflight as pf  # noqa: E402

CATALOG = {"data": [
    {"id": "cheap/one", "pricing": {"prompt": "0.0000001", "completion": "0.0000002"},
     "context_length": 128000, "top_provider": {"max_completion_tokens": 8192},
     "supported_parameters": ["tools", "temperature"]},
    {"id": "notools/two", "pricing": {"prompt": "0.0000005", "completion": "0.0000005"},
     "context_length": 32768, "top_provider": {"max_completion_tokens": 4096},
     "supported_parameters": ["temperature"]},
    {"id": "variable/three", "pricing": {"prompt": "-1", "completion": "-1"},
     "context_length": 8192, "top_provider": {}, "supported_parameters": ["tools"]},
    {"pricing": {}},                      # malformed: no id
]}


class TestCatalog(unittest.TestCase):
    def setUp(self):
        self.catalog = pf.parse_catalog(CATALOG)

    def test_malformed_entries_are_dropped(self):
        self.assertEqual(len(self.catalog), 3)

    def test_tool_support_is_read_from_supported_parameters(self):
        by_id = {m["id"]: m for m in self.catalog}
        self.assertTrue(pf.supports_tools(by_id["cheap/one"]))
        self.assertFalse(pf.supports_tools(by_id["notools/two"]))

    def test_variable_pricing_has_no_price(self):
        # -1 means the price is not fixed; treating it as free would make every
        # cost estimate meaningless.
        by_id = {m["id"]: m for m in self.catalog}
        self.assertIsNone(pf.price_per_mtok(by_id["variable/three"]))

    def test_price_is_per_million_combined(self):
        by_id = {m["id"]: m for m in self.catalog}
        self.assertAlmostEqual(pf.price_per_mtok(by_id["cheap/one"]), 0.3)


class TestIdVerification(unittest.TestCase):
    """A near miss is a 404 discovered by paying for the attempt, so matching is
    exact - no normalising, no fuzzy matching."""

    def setUp(self):
        self.catalog = pf.parse_catalog(CATALOG)

    def test_exact_ids_are_found(self):
        present, missing = pf.verify_ids(self.catalog, ["cheap/one"])
        self.assertEqual((present, missing), (["cheap/one"], []))

    def test_a_near_miss_is_missing_not_matched(self):
        for near in ("cheap/One", "cheap/one ", "cheap/on", "~cheap/one"):
            self.assertEqual(pf.verify_ids(self.catalog, [near])[1], [near], near)

    def test_mixed_input_is_split(self):
        present, missing = pf.verify_ids(self.catalog, ["cheap/one", "ghost/model"])
        self.assertEqual(present, ["cheap/one"])
        self.assertEqual(missing, ["ghost/model"])


def completion(**overrides):
    body = {
        "choices": [{"finish_reason": "tool_calls", "message": {
            "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "run_bash", "arguments": '{"command": "echo ok"}'}}]}}],
        "usage": {"total_tokens": 120, "cost": 0.00004},
    }
    body.update(overrides)
    return body


class TestCompletionAssessment(unittest.TestCase):
    """Each problem here corresponds to a failure that used to be discovered
    only by burning a whole match."""

    def test_a_good_completion_has_no_problems(self):
        self.assertEqual(pf.assess_completion(completion()), [])

    def test_a_200_with_no_choices_is_caught(self):
        problems = pf.assess_completion({"error": {"code": 502}})
        self.assertTrue(any("no choices" in p for p in problems))

    def test_truncation_is_caught(self):
        body = completion()
        body["choices"][0]["finish_reason"] = "length"
        self.assertTrue(any("truncated" in p for p in pf.assess_completion(body)))

    def test_multiple_tool_calls_are_caught(self):
        body = completion()
        calls = body["choices"][0]["message"]["tool_calls"]
        body["choices"][0]["message"]["tool_calls"] = calls * 2
        self.assertTrue(any("2 tool calls" in p for p in pf.assess_completion(body)))

    def test_list_content_is_caught(self):
        body = completion()
        body["choices"][0]["message"]["content"] = [{"type": "text", "text": "hi"}]
        self.assertTrue(any("list of parts" in p for p in pf.assess_completion(body)))

    def test_unparseable_arguments_are_caught(self):
        body = completion()
        body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = '{"comm'
        self.assertTrue(any("did not parse" in p for p in pf.assess_completion(body)))

    def test_a_wrong_tool_name_is_caught(self):
        body = completion()
        body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] = "browse"
        self.assertTrue(any("not run_bash" in p for p in pf.assess_completion(body)))

    def test_missing_usage_is_caught(self):
        # Without total_tokens the budget never advances, so spend is unbounded.
        body = completion(usage={"cost": 0.1})
        self.assertTrue(any("total_tokens" in p for p in pf.assess_completion(body)))

    def test_missing_cost_is_caught(self):
        body = completion(usage={"total_tokens": 10})
        self.assertTrue(any("cost" in p for p in pf.assess_completion(body)))

    def test_no_tool_call_at_all_is_caught(self):
        body = completion()
        body["choices"][0]["message"] = {"content": "I would run echo ok"}
        self.assertTrue(any("did not use run_bash" in p for p in pf.assess_completion(body)))


if __name__ == "__main__":
    unittest.main()
