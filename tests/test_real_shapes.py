"""Response shapes that only a REAL provider produces.

The mock backend emits one syntactically perfect run_bash tool call, with valid
JSON arguments, a string content, finish_reason "tool_calls" and zeroed usage,
every single time. Every case here is something it structurally cannot produce -
which is why 189 passing tests said nothing about whether a real match works.
"""
import json
import os
import sys
import unittest
from unittest import mock
from pathlib import Path

SRC = str(Path(__file__).resolve().parent.parent / "src")
sys.path.insert(0, SRC)
os.environ.setdefault("LOG_PATH", "")
import agent_harness as ah  # noqa: E402


def tool_call(call_id, name="run_bash", arguments='{"command":"ps aux"}'):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": arguments}}


def response(message):
    return {"choices": [{"message": message}]}


class TestMultipleToolCalls(unittest.TestCase):
    """An OpenAI-compatible endpoint requires a `tool` reply per tool_call_id.
    Models routinely emit two or three despite being told to issue one - the
    system prompt is a request, not a constraint - and a single unanswered id
    makes every later request 400 for the rest of the match."""

    @staticmethod
    def unanswered(messages):
        missing = []
        for index, message in enumerate(messages):
            if message.get("role") != "assistant" or not message.get("tool_calls"):
                continue
            ids = {call["id"] for call in message["tool_calls"]}
            replied = {m.get("tool_call_id") for m in messages[index + 1:]
                       if m.get("role") == "tool"}
            missing += sorted(ids - replied)
        return missing

    def test_every_id_gets_a_reply(self):
        message = {"role": "assistant", "content": None,
                   "tool_calls": [tool_call("c1"), tool_call("c2"), tool_call("c3")]}
        action = ah.extract_command(response(message))
        assistant = ah.normalize_assistant_message(action["message"])
        answered = action["tool_call_id"]
        built = ([assistant, {"role": "tool", "tool_call_id": answered, "content": "{}"}]
                 + ah.unanswered_tool_replies(assistant, answered, 1))
        self.assertEqual(self.unanswered(built), [])

    def test_the_executed_call_is_not_answered_twice(self):
        assistant = {"role": "assistant", "tool_calls": [tool_call("c1"), tool_call("c2")]}
        replies = ah.unanswered_tool_replies(assistant, "c1", 1)
        self.assertEqual([r["tool_call_id"] for r in replies], ["c2"])

    def test_a_hallucinated_tool_name_still_needs_answering(self):
        # It falls through to kind "none", but its id is still in the history.
        assistant = {"role": "assistant", "tool_calls": [tool_call("z1", name="browse")]}
        self.assertEqual([r["tool_call_id"]
                          for r in ah.unanswered_tool_replies(assistant, None, 1)], ["z1"])

    def test_a_single_call_needs_no_extra_replies(self):
        assistant = {"role": "assistant", "tool_calls": [tool_call("c1")]}
        self.assertEqual(ah.unanswered_tool_replies(assistant, "c1", 1), [])

    def test_calls_without_ids_are_skipped(self):
        assistant = {"role": "assistant", "tool_calls": [{"function": {"name": "x"}}]}
        self.assertEqual(ah.unanswered_tool_replies(assistant, None, 1), [])


class TestContentShapes(unittest.TestCase):
    """Content as a list of typed parts is common from reasoning models. A list
    is truthy, so it used to reach the fence regex and raise TypeError out of
    main() - killing the container and handing the opponent a RATED kill."""

    def test_a_list_of_parts_does_not_raise(self):
        action = ah.extract_command(response(
            {"content": [{"type": "text", "text": "I will run:\n```bash\nwhoami\n```"}]}))
        self.assertEqual(action["kind"], "fence")
        self.assertEqual(action["command"], "whoami")

    def test_plain_strings_in_a_list_are_joined(self):
        self.assertEqual(ah.coerce_content(["a", "b"]), "a\nb")

    def test_none_and_str_are_unchanged(self):
        self.assertEqual(ah.coerce_content(None), "")
        self.assertEqual(ah.coerce_content("hi"), "hi")

    def test_unexpected_types_degrade_rather_than_raise(self):
        self.assertIsInstance(ah.coerce_content({"weird": 1}), str)
        self.assertIsInstance(ah.coerce_content(42), str)

    def test_parts_without_text_are_dropped(self):
        self.assertEqual(ah.coerce_content([{"type": "image", "url": "x"}]), "")


class TestTruncatedArguments(unittest.TestCase):
    """If the max_tokens cap truncates mid-arguments, json.loads fails. Falling
    back to the raw string used to hand the broken JSON fragment to bash -c."""

    def test_a_truncated_object_is_an_error_not_a_command(self):
        action = ah.extract_command(response({"tool_calls": [
            tool_call("c", arguments='{"command":"rm -rf /battle/logs && echo don')]}))
        self.assertEqual(action["kind"], "error")
        self.assertTrue(action["truncated"])
        self.assertIsNone(action.get("command"))

    def test_a_truncated_array_is_also_refused(self):
        action = ah.extract_command(response({"tool_calls": [
            tool_call("c", arguments='[{"command":"ls')]}))
        self.assertEqual(action["kind"], "error")

    def test_a_bare_command_string_is_still_accepted(self):
        # Not meant to be JSON, so not a truncation - some models do this.
        action = ah.extract_command(response({"tool_calls": [
            tool_call("c", arguments="ps aux")]}))
        self.assertEqual((action["kind"], action["command"]), ("tool_call", "ps aux"))

    def test_a_valid_object_is_unaffected(self):
        action = ah.extract_command(response({"tool_calls": [tool_call("c")]}))
        self.assertEqual(action["command"], "ps aux")


class TestUpstreamBodyErrors(unittest.TestCase):
    """OpenRouter answers 200 with a body-level error when a provider fails
    AFTER accepting the request. Trusting the status charged the time bank and
    turned a provider outage into a rated protocol forfeit."""

    def setUp(self):
        os.environ.setdefault("TOKENS_JSON", json.dumps({"t": "m"}))
        import model_proxy
        self.mp = model_proxy

    def _forward(self, payload, status=200):
        class FakeResponse:
            def __init__(self, body, code):
                self._body, self.status = json.dumps(body).encode(), code

            def read(self):
                return self._body

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        with mock.patch.object(self.mp.urllib.request, "urlopen",
                               return_value=FakeResponse(payload, status)):
            return self.mp.forward_openrouter({"messages": []}, "m/x")

    def test_200_with_a_body_error_becomes_503_upstream(self):
        status, body = self._forward(
            {"error": {"code": 429, "message": "provider is rate limiting"}})
        self.assertEqual(status, 503)
        self.assertEqual(body["error_kind"], "upstream")
        self.assertEqual(body["upstream_status"], 429)

    def test_200_with_empty_choices_is_also_upstream(self):
        self.assertEqual(self._forward({"choices": []})[0], 503)

    def test_a_real_completion_passes_through(self):
        status, body = self._forward(
            {"choices": [{"message": {"content": "hi"}}], "usage": {"total_tokens": 5}})
        self.assertEqual(status, 200)
        self.assertEqual(body["usage"]["total_tokens"], 5)

    def test_the_provider_message_is_surfaced_for_diagnosis(self):
        _, body = self._forward({"error": {"code": 502, "message": "upstream exploded"}})
        self.assertIn("exploded", body["error"])
        self.assertEqual(body["stage"], "decode")


class TestRedaction(unittest.TestCase):
    """Logging upstream error bodies is only safe if no secret can survive it.
    The old proxy leaked raw agent tokens into proxy.jsonl."""

    def setUp(self):
        import model_proxy
        self.mp = model_proxy

    def test_every_secret_the_proxy_holds_is_stripped(self):
        secrets = [s for s in [self.mp.API_KEY, self.mp.CONTROL_TOKEN, *self.mp.TOKENS]
                   if s and len(s) >= 8]
        for secret in secrets:
            self.assertNotIn(secret, self.mp._redact(f"upstream said {secret} was bad"))

    def test_non_strings_are_handled(self):
        self.assertIsInstance(self.mp._redact({"a": 1}), str)
        self.assertIsInstance(self.mp._redact(None), str)

    def test_ordinary_text_is_untouched(self):
        self.assertEqual(self.mp._redact("no secrets here"), "no secrets here")


if __name__ == "__main__":
    unittest.main()
