"""The warfare preset is shipped to orchestrator, proxy and harness as one
frozen value, exactly like modes.py - the three components must not disagree
about whether the fight is one-shot or many-layered."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import warfare  # noqa: E402


class TestResolution(unittest.TestCase):
    def test_default_is_warfare(self):
        w = warfare.resolve()
        self.assertTrue(w.enabled)
        self.assertTrue(w.stealth)
        self.assertTrue(w.process_bulwark)
        self.assertEqual(w.prompt_style, "warfare")

    def test_classic_disables_everything(self):
        w = warfare.resolve(False)
        self.assertFalse(w.enabled)
        self.assertFalse(w.stealth)
        self.assertFalse(w.process_bulwark)
        self.assertEqual(w.prompt_style, "classic")
        self.assertEqual(w.heartbeat_rebind_attempts, 1)

    def test_env_round_trip_is_exact(self):
        for w in (warfare.WARFARE_ON, warfare.CLASSIC):
            self.assertEqual(warfare.from_env(warfare.to_env(w)), w)

    def test_from_env_defaults_to_warfare(self):
        self.assertEqual(warfare.from_env({}), warfare.WARFARE_ON)


class TestPromptNote(unittest.TestCase):
    def test_warfare_note_mentions_shared_ip_not_localhost(self):
        note = warfare.prompt_note(warfare.WARFARE_ON)
        self.assertIn("hostname -I", note)
        self.assertIn("NOT on localhost", note)

    def test_classic_prompt_note_is_empty(self):
        self.assertEqual(warfare.prompt_note(warfare.CLASSIC), "")

    def test_to_dict_is_plain(self):
        self.assertEqual(warfare.to_dict(warfare.WARFARE_ON),
                         warfare.WARFARE_ON._asdict())
