"""Regression tests for the agent harness."""
import importlib
import json
import os
import sys
import time
import unittest
from pathlib import Path

SRC = str(Path(__file__).resolve().parent.parent / "src")
sys.path.insert(0, SRC)
os.environ.setdefault("LOG_PATH", "")
os.environ.setdefault("BATTLE_DIR", "/tmp")
os.environ.setdefault("COMMAND_TIMEOUT", "5")
import agent_harness as ah  # noqa: E402


def tool_response(arguments, name="run_bash"):
    return {"choices": [{"message": {"tool_calls": [
        {"id": "c1", "type": "function",
         "function": {"name": name, "arguments": arguments}}]}}]}


class TestExtractCommand(unittest.TestCase):
    """Audit H4: a non-dict `arguments` payload raised AttributeError straight
    out of main(), killing the container and handing the opponent a win."""

    def test_object_arguments(self):
        a = ah.extract_command(tool_response('{"command": "ps aux"}'))
        self.assertEqual((a["kind"], a["command"]), ("tool_call", "ps aux"))

    def test_bare_string_arguments_keep_no_quotes(self):
        a = ah.extract_command(tool_response('"ls -la"'))
        self.assertEqual((a["kind"], a["command"]), ("tool_call", "ls -la"))

    def test_list_arguments_do_not_crash(self):
        self.assertEqual(ah.extract_command(tool_response("[1,2]"))["kind"], "none")

    def test_number_arguments_do_not_crash(self):
        self.assertEqual(ah.extract_command(tool_response("42"))["kind"], "none")

    def test_null_arguments_do_not_crash(self):
        self.assertEqual(ah.extract_command(tool_response("null"))["kind"], "none")

    def test_unparseable_arguments_fall_back_to_raw(self):
        a = ah.extract_command(tool_response("ps aux"))
        self.assertEqual((a["kind"], a["command"]), ("tool_call", "ps aux"))

    def test_non_run_bash_tool_is_ignored(self):
        self.assertEqual(
            ah.extract_command(tool_response('{"command":"x"}', name="other"))["kind"],
            "none")

    def test_missing_choices_is_an_error_not_a_crash(self):
        self.assertEqual(ah.extract_command({})["kind"], "error")

    def test_fenced_content_fallback(self):
        resp = {"choices": [{"message": {"content": "sure:\n```bash\nwhoami\n```"}}]}
        a = ah.extract_command(resp)
        self.assertEqual((a["kind"], a["command"]), ("fence", "whoami"))


class TestTrimMessages(unittest.TestCase):
    """Audit L3: messages[-0:] is the whole list, so a small MAX_MESSAGES made
    the trimmer grow the context it exists to shrink."""

    @staticmethod
    def _conversation(turns=5):
        msgs = [{"role": "system", "content": "s"}]
        for i in range(turns):
            msgs.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": f"c{i}", "type": "function",
                 "function": {"name": "run_bash", "arguments": "{}"}}]})
            msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "o"})
        return msgs

    def _reload(self, max_messages):
        os.environ["MAX_MESSAGES"] = str(max_messages)
        return importlib.reload(ah)

    def tearDown(self):
        os.environ["MAX_MESSAGES"] = "120"
        importlib.reload(ah)

    def test_never_grows_the_list(self):
        for mm in (1, 2, 3, 4, 8):
            mod = self._reload(mm)
            msgs = self._conversation()
            self.assertLessEqual(len(mod.trim_messages(msgs)), len(msgs),
                                 f"MAX_MESSAGES={mm} grew the conversation")

    def test_never_orphans_a_tool_message(self):
        # An orphaned `tool` message makes every later request 400 upstream.
        for mm in (1, 2, 3, 4, 8, 12):
            mod = self._reload(mm)
            out = mod.trim_messages(self._conversation())
            for i, m in enumerate(out):
                if m["role"] == "tool":
                    self.assertTrue(
                        i > 0 and out[i - 1].get("role") == "assistant"
                        and out[i - 1].get("tool_calls"),
                        f"MAX_MESSAGES={mm} orphaned a tool message")

    def test_keeps_the_system_prompt_exactly_once(self):
        mod = self._reload(6)
        out = mod.trim_messages(self._conversation())
        self.assertEqual(sum(1 for m in out if m["role"] == "system"), 1)

    def test_short_conversation_is_untouched(self):
        mod = self._reload(120)
        msgs = self._conversation(2)
        self.assertEqual(mod.trim_messages(msgs), msgs)


class TestRunCommand(unittest.TestCase):
    """Audit H10: with pipes, a backgrounded process inherits the write end and
    blocks the parent for the full timeout, so a command that succeeded
    instantly was reported to the model as a timeout."""

    def test_backgrounded_process_does_not_block(self):
        start = time.time()
        r = ah.run_command("(sleep 30 &) ; echo hi")
        self.assertLess(time.time() - start, 3.0)
        self.assertFalse(r["timed_out"])
        self.assertIn("hi", r["stdout"])

    def test_genuine_hang_still_times_out(self):
        start = time.time()
        r = ah.run_command("sleep 30")
        self.assertTrue(r["timed_out"])
        self.assertLess(time.time() - start, 10.0)

    def test_exit_code_and_streams_are_preserved(self):
        r = ah.run_command("echo out; echo err >&2; exit 7")
        self.assertEqual(r["exit_code"], 7)
        self.assertIn("out", r["stdout"])
        self.assertIn("err", r["stderr"])

    def test_output_is_truncated_not_unbounded(self):
        r = ah.run_command("head -c 100000 /dev/zero | tr '\\0' 'x'")
        self.assertLessEqual(len(r["stdout"]), ah.MAX_OUTPUT + 200)


class TestParseFenced(unittest.TestCase):
    def test_plain_fence(self):
        self.assertEqual(ah.parse_fenced_command("```\nid\n```"), "id")

    def test_dollar_prefix(self):
        self.assertEqual(ah.parse_fenced_command("run this:\n$ uname -a"), "uname -a")

    def test_no_command_returns_none(self):
        self.assertIsNone(ah.parse_fenced_command("I will think about it."))


class TestTailText(unittest.TestCase):
    def test_short_text_unchanged(self):
        self.assertEqual(ah.tail_text("abc", 10), "abc")

    def test_long_text_keeps_the_tail(self):
        out = ah.tail_text("x" * 100, 10)
        self.assertTrue(out.endswith("x" * 10))
        self.assertIn("truncated", out)

    def test_none_is_empty(self):
        self.assertEqual(ah.tail_text(None, 10), "")


if __name__ == "__main__":
    unittest.main()
