"""The container sandbox: what the two agents share, and what bounds them.

These assert on the podman invocation the orchestrator builds, because that is
where the properties live - there is no Python object to interrogate.
"""
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
import orchestrator as orch  # noqa: E402

SOURCE = (SRC / "orchestrator.py").read_text()


class TestNamespaceSharing(unittest.TestCase):
    """The agents must see and be able to signal each other - and nothing else."""

    def test_the_pid_namespace_is_shared(self):
        # Without it neither agent can find or kill the other and the whole
        # premise of the benchmark is void.
        self.assertIn('"--share", "net,pid,uts"', SOURCE)

    def test_the_ipc_namespace_is_NOT_shared(self):
        """Sharing IPC gives pod members a common /dev/shm - a read-write
        channel between two agents that are supposed to be adversaries. Verified
        exploitable before this changed: one wrote a string, the other read it."""
        self.assertNotIn("net,pid,ipc,uts", SOURCE)
        self.assertNotIn("net,ipc,uts", SOURCE)

    def test_the_degraded_fallback_also_drops_ipc(self):
        # create_pod retries without pid when shared-pid pod creation fails; that
        # path must not quietly restore the channel.
        self.assertIn('"--share", "net,uts"', SOURCE)

    def test_a_match_records_that_ipc_was_not_shared(self):
        self.assertIn('"ipc_shared": False', SOURCE)


class TestFilesystemBound(unittest.TestCase):
    """Agents must still be able to write and run code. What changes is that the
    space is bounded, so filling it costs the agent its own quota rather than
    the host's disk."""

    def test_the_working_directory_stays_writable(self):
        # mode=1777 because a tmpfs OVERLAYS the image directory with a fresh
        # root-owned one and podman's --tmpfs takes no uid/gid option. Without
        # this the agent cannot write code at all.
        self.assertIn("/battle:size={args.battle_size},mode=1777,exec", SOURCE)

    def test_scratch_and_home_are_writable_too(self):
        self.assertIn('"/tmp:size=32m,mode=1777,exec"', SOURCE)
        self.assertIn('"/home/battler:size=32m,mode=1777,exec"', SOURCE)

    def test_the_working_directory_allows_execution(self):
        # An agent that writes a script must be able to chmod +x and run it.
        self.assertIn("mode=1777,exec", SOURCE)

    def test_the_rest_of_the_filesystem_is_read_only_by_default(self):
        self.assertIn('"--read-only"', SOURCE)
        self.assertIn("read_only_fs", SOURCE)

    def test_the_bound_can_be_lifted_deliberately(self):
        # Escape hatches, because a bounded arena is a judgement call.
        self.assertIn("--unbounded-fs", SOURCE)
        self.assertIn("--no-read-only-fs", SOURCE)

    def test_the_bound_is_recorded_in_the_result(self):
        self.assertIn('"battle_size"', SOURCE)
        self.assertIn('"read_only_fs"', SOURCE)


class TestExistingHardeningKept(unittest.TestCase):
    """Regression guard: this round must not have loosened anything."""

    def test_capabilities_are_still_dropped(self):
        self.assertIn('"--cap-drop", "ALL"', SOURCE)

    def test_privilege_escalation_is_still_blocked(self):
        self.assertIn('"no-new-privileges"', SOURCE)

    def test_agents_still_have_no_writable_host_mount(self):
        # The battle log is container stdout, collected on the host, so an agent
        # cannot rewrite the record that judges it.
        self.assertIn('"LOG_PATH": ""', SOURCE)

    def test_resource_limits_are_still_applied(self):
        for flag in ("--memory", "--cpus", "--pids-limit"):
            self.assertIn(f'"{flag}"', SOURCE)


if __name__ == "__main__":
    unittest.main()
