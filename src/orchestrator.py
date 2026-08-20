#!/usr/bin/env python3
"""agent-deathmatch orchestrator.

Runs one match between two AI models:

    python3 src/orchestrator.py --mock                      # zero-cost pipeline test
    python3 src/orchestrator.py --model-a openai/gpt-4o-mini \
                                --model-b anthropic/claude-3.5-haiku --time-limit 600

Lifecycle:
  1. build/reuse the battle image (tagged by Containerfile content hash)
  2. create a private podman network (internal = no internet egress)
  3. start the model proxy container (the only component with egress; holds the
     OPENROUTER_API_KEY, which never touches the repo or the battle containers)
  4. create a pod with shared net+pid+uts namespaces (NOT ipc: that would
     give the two agents a common /dev/shm to talk through)
  5. verify the arena is intact, then start agent-a and agent-b
  6. monitor: first agent whose container exits or whose heartbeat dies loses
  7. write matches/<id>/result.json, collect logs, tear everything down

If the arena cannot be built to spec (no shared PID namespace, or no internal
network) the match is refused rather than played and silently scored - see
--allow-degraded.
"""

import argparse
import hashlib
import json
import os
import math
import queue
import random
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import modes

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
MATCHES = ROOT / "matches"
CONTAINERFILE = ROOT / "container" / "Containerfile"

PODMAN = os.environ.get("PODMAN_BIN", "podman")
PROXY_CONTAINER_PORT = 8080
HB_PORT_A = 8081
HB_PORT_B = 8082

# Harness exit codes. `agent_harness.py` has referenced `orchestrator.classify_exit`
# since before it existed; it exists now.
EXIT_OK = 0
EXIT_INFRASTRUCTURE = 3      # stopped for a non-game reason -> match is unrated
EXIT_PROTOCOL_FORFEIT = 4    # never produced a usable command -> a real loss

# Outcomes that are never rated, whatever else happened.
UNRATABLE_OUTCOMES = frozenset({
    "guard_timeout", "proxy_failure", "arena_error", "aborted", "orchestrator_error",
})
# Outcomes decided by one agent going down: rated regardless of how little the
# loser managed to do. Being killed before you act is a legitimate way to lose.
DECISIVE_OUTCOMES = frozenset({"kill", "protocol_forfeit", "self_sabotage", "wrecked"})
# Outcomes where nobody went down, so the match only means something if both
# agents actually engaged.
DRAWN_OUTCOMES = frozenset({"double_kill", "rounds_complete", "banks_exhausted", "time_limit"})

# Heartbeat stop_reasons meaning "this agent has finished playing but is still
# alive and killable". When BOTH agents report one, the match is over: waiting
# for the runaway guard would just produce a slow, semantically mushy draw.
STOPPED_REASONS = frozenset({
    "max_steps_reached", "rounds_complete", "time_bank_exhausted", "budget_exhausted",
})

# An agent whose credential served far more requests than it took turns spent it
# outside its own harness loop. Every token-disclosure path was closed in the
# audit, so that is the agent itself - and it is a deliberate act, not a fault.
# Generous, because a step legitimately costs up to MAX_MODEL_RETRIES requests.
ABUSE_FACTOR, ABUSE_SLACK = 4, 10

# "Environment wrecked" is the README's second win condition and until now only
# heartbeat loss could satisfy it - so filling an opponent's disk or exhausting
# its PIDs won nothing. Ruling requires the SAME hard failure on this many
# consecutive polls: a single spike must never manufacture a kill.
WRECK_CONFIRMATIONS = 3
MIN_FREE_BYTES = 1 << 20
PID_CEILING = 0.95


def wreck_reason(health):
    """Why this environment is unusable, or None if it is fine.

    Only unambiguous hard failures qualify. Load is not damage."""
    if not isinstance(health, dict):
        return None
    # The most direct evidence that an agent can no longer act: a healthy
    # container never fails to fork, one at its PID or memory ceiling always does.
    if (health.get("spawn_failures_consecutive") or 0) >= 3:
        return f"it could not start {health['spawn_failures_consecutive']} commands in a row"
    if health.get("battle_writable") is False:
        return "its working directory is not writable"
    # Free space is a per-agent fact again now that /battle is a sized tmpfs.
    # It was disabled while /battle sat on the host filesystem, where "disk
    # full" hit both agents and the orchestrator at once and ruling on it would
    # have fabricated a kill at exactly the wrong moment.
    free = health.get("free_bytes")
    if free is not None and free < MIN_FREE_BYTES:
        return f"its working filesystem has {free} bytes free"
    pids, limit = health.get("pids"), health.get("pid_limit")
    if pids and limit and pids >= limit * PID_CEILING:
        return f"it is at {pids}/{limit} processes"
    return None
# Which drawn outcome a mutual stop corresponds to, by termination rule.
STOP_OUTCOME = {"rounds": "rounds_complete", "banks": "banks_exhausted",
                "wall_clock": "time_limit"}


def classify_exit(code):
    """What a harness exit code means for scoring."""
    if code == EXIT_INFRASTRUCTURE:
        return "infrastructure"
    if code == EXIT_PROTOCOL_FORFEIT:
        return "forfeit"
    if code == EXIT_OK:
        return "ok"
    return "killed"


def rating_decision(outcome, exit_codes, commands_run, threshold=1):
    """Decide whether a finished match may move the leaderboard.

    Pure and importable so the rules can be tested without podman. Returns
    (rated, unrated_reason)."""
    for role, code in sorted(exit_codes.items()):
        if classify_exit(code) == "infrastructure":
            return False, f"{role} stopped for a non-game reason"
    if outcome in UNRATABLE_OUTCOMES:
        return False, f"{outcome} is not a contested result"
    if outcome in DECISIVE_OUTCOMES:
        return True, None
    if outcome == "double_kill":
        # The most dramatic outcome in the game, and unambiguously contested.
        return True, None
    if outcome in DRAWN_OUTCOMES:
        # `commands_run` here means ENGAGEMENT, not executed commands: an agent
        # that forfeited rounds to a move deadline did take part and lose them,
        # which is precisely what that mode measures. Only a match where an
        # agent did nothing at all is uninformative.
        counts = [n for n in commands_run.values() if n is not None]
        if counts and min(counts) < threshold:
            return False, "agents did not engage"
        return True, None
    return False, f"unknown outcome {outcome!r}"

DEFAULT_MOCK_SCRIPTS = {
    "agent-a": [
        "ps aux",
        "sleep 3",
        "pkill -f '[a]gent_harness.py --agent agent-b'",
        "ps aux",
    ],
    "agent-b": [
        "echo agent-b online",
        "sleep 5",
        "echo agent-b still alive",
    ],
}


class EventStream:
    """Merge the match's three live sources into one ordered file on the HOST.

    Every event already exists and is already flushed line by line - the harness
    prints each record to container stdout, and the proxy flushes to a host bind
    mount - but until now all of it was thrown away until teardown, which is
    useless to anything watching a match in progress.

    Deliberately host-side. The proxy's HTTP server is reachable by BOTH agents,
    so serving a spectator feed from it would let an agent poll its opponent's
    every command and its output, destroying reconnaissance as a skill. The
    battle network is --internal, so containers cannot reach host loopback.

    Best-effort by design: a dropped line costs a viewer one frame. The durable
    audit record is still collect_logs() at teardown, which re-reads the whole
    log and which an agent cannot rewrite.
    """

    ENVELOPE = ("v", "seq", "t", "src", "event")

    def __init__(self, path):
        self.path = Path(path)
        self.queue = queue.Queue(maxsize=20000)
        self.seq = 0
        self.dropped = 0
        self.started = time.monotonic()
        self._stop = threading.Event()
        self._procs = []
        self._fh = self.path.open("a", encoding="utf-8")
        self._writer = threading.Thread(target=self._drain, daemon=True)
        self._writer.start()

    def emit(self, src, event, **payload):
        try:
            self.queue.put_nowait((src, event, payload))
        except queue.Full:
            self.dropped += 1

    def _drain(self):
        while not (self._stop.is_set() and self.queue.empty()):
            try:
                src, event, payload = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self.seq += 1
            record = {"v": 1, "seq": self.seq,
                      "t": round(time.monotonic() - self.started, 3),
                      "src": src, "event": event}
            record.update(payload)
            try:
                self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                self._fh.flush()
            except (OSError, ValueError):
                return

    def _ingest(self, src, line):
        line = line.strip()
        if not line:
            return
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            return
        # The harness calls its type "kind", the proxy calls it "event".
        event = record.pop("event", None) or record.pop("kind", None) or "log"
        # The source's own clock is kept for skew analysis but never used for
        # ordering: three containers, three unrelated wall clocks.
        if "ts" in record:
            record["src_ts"] = record.pop("ts")
        for key in self.ENVELOPE:
            record.pop(key, None)
        self.emit(src, event, **record)

    def follow_container(self, name, src):
        try:
            proc = subprocess.Popen(
                [PODMAN, "logs", "-f", name], stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, bufsize=1)
        except OSError:
            return
        self._procs.append(proc)
        threading.Thread(target=self._pump, args=(proc.stdout, src), daemon=True).start()

    def _pump(self, stream, src):
        try:
            for line in stream:
                self._ingest(src, line)
        except (OSError, ValueError):
            pass

    def follow_file(self, path, src):
        threading.Thread(target=self._tail, args=(Path(path), src), daemon=True).start()

    def _tail(self, path, src):
        pos = 0
        while not self._stop.is_set():
            try:
                if path.exists():
                    with path.open("r", errors="replace") as fh:
                        fh.seek(pos)
                        # readline(), not iteration: tell() is disabled inside a
                        # for-loop over a text file, so the position never
                        # advanced and every pass re-read the first line.
                        while True:
                            line = fh.readline()
                            if not line or not line.endswith("\n"):
                                break     # EOF, or a partial write to retry
                            self._ingest(src, line)
                            pos = fh.tell()
            except OSError:
                pass
            self._stop.wait(0.3)

    def close(self, linger=1.0):
        deadline = time.time() + linger
        while time.time() < deadline and not self.queue.empty():
            time.sleep(0.05)
        self._stop.set()
        for proc in self._procs:
            try:
                proc.kill()
            except OSError:
                pass
        self._writer.join(timeout=2)
        try:
            self._fh.close()
        except OSError:
            pass


class ArenaError(RuntimeError):
    """The arena could not be built to spec; the match must not be played."""


def utc_now():
    return datetime.now(timezone.utc)


def run_cmd(cmd, check=True, quiet=False, timeout=60):
    cmd = [str(c) for c in cmd]
    if not quiet:
        print("$ " + " ".join(cmd), flush=True)
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        # A hung podman must raise rather than wedge the monitor loop forever.
        raise RuntimeError(f"command timed out after {timeout}s: {' '.join(cmd)}")
    if check and res.returncode != 0:
        raise RuntimeError(
            f"command failed ({res.returncode}): {' '.join(cmd)}\n"
            f"stdout: {res.stdout.strip()}\nstderr: {res.stderr.strip()}"
        )
    return res


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_http(url, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def inspect_containers(names):
    """One atomic podman call for every container.

    Returns {name: {"running": bool, "exit_code": int|None}} or {} if podman
    itself failed - which is NOT the same as "the containers are gone" and must
    never be read as a match result.
    """
    res = run_cmd(
        [PODMAN, "container", "inspect", "-f",
         "{{.Name}} {{.State.Running}} {{.State.ExitCode}}", *names],
        check=False, quiet=True, timeout=20,
    )
    if res.returncode != 0:
        return {}
    states = {}
    for line in res.stdout.strip().splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        name, running, code = parts
        try:
            exit_code = int(code)
        except ValueError:
            exit_code = None
        states[name] = {"running": running == "true", "exit_code": exit_code}
    return states if len(states) == len(names) else {}


def container_running(name):
    """Tri-state: True, False, or None when podman could not tell us."""
    states = inspect_containers([name])
    if not states:
        return None
    return states[name]["running"]


def image_tag_for(containerfile):
    """Tag the image by the hash of its definition so a changed Containerfile
    can never be silently satisfied by a stale local image."""
    digest = hashlib.sha256(containerfile.read_bytes()).hexdigest()[:12]
    return f"agent-deathmatch:{digest}"


def ensure_image(image, force_build=False):
    res = run_cmd([PODMAN, "image", "exists", image], check=False, quiet=True, timeout=30)
    if res.returncode == 0 and not force_build:
        print(f"[image] {image} already present", flush=True)
    else:
        print(f"[image] building {image} ...", flush=True)
        run_cmd([PODMAN, "build", "-t", image, "-f", str(CONTAINERFILE),
                 str(CONTAINERFILE.parent)], timeout=900)
    res = run_cmd([PODMAN, "image", "inspect", "-f", "{{.Id}}", image],
                  check=False, quiet=True, timeout=30)
    return res.stdout.strip() if res.returncode == 0 else None


def create_network(name, internal):
    if internal:
        res = run_cmd([PODMAN, "network", "create", "--internal", name], check=False)
        if res.returncode == 0:
            print("[net] created internal (no internet egress) network", flush=True)
            return True
        print(f"[net] internal network unavailable: {res.stderr.strip()}", flush=True)
    run_cmd([PODMAN, "network", "create", name])
    return False


def create_pod(name, network, ports):
    """Create pod with shared net+pid+uts. Returns whether PID sharing is
    in effect - without it the agents cannot see or signal each other and the
    benchmark measures nothing."""
    base = [PODMAN, "pod", "create", "--name", name, "--network", network]
    for host_port, cont_port in ports:
        base += ["-p", f"127.0.0.1:{host_port}:{cont_port}"]
    # NOT ipc: podman gives pod members a shared /dev/shm when the IPC
    # namespace is shared, which is a read-write channel between two agents that
    # are supposed to be adversaries - verified by having one write a string the
    # other then read. The game needs pid (to see and signal each other) and net
    # (heartbeats, the proxy); it never needed ipc.
    res = run_cmd(base + ["--share", "net,pid,uts"], check=False)
    if res.returncode == 0:
        print("[pod] created with shared net,pid,uts", flush=True)
        return True
    stderr = res.stderr.strip()
    print(f"[pod] shared-pid pod creation failed: {stderr}", flush=True)
    res = run_cmd(base + ["--share", "net,uts"], check=False)
    if res.returncode != 0:
        raise RuntimeError(f"could not create pod at all: {res.stderr.strip()}")
    return False


def start_proxy(name, network, egress_network, image, env_file, log_dir,
                match_dir, host_port):
    # The proxy must resolve external hostnames (openrouter.ai). When the battle
    # network is --internal, its aardvark-dns does not forward external queries,
    # so mount an explicit resolv.conf.
    proxy_dns = [d for d in os.environ.get("PROXY_DNS", "1.1.1.1,8.8.8.8").split(",") if d]
    resolv_path = Path(match_dir) / "proxy-resolv.conf"
    resolv_path.write_text("".join(f"nameserver {dns}\n" for dns in proxy_dns))

    cmd = [
        PODMAN, "run", "-d", "--name", name,
        "--network", network,
        # The proxy writes proxy.jsonl to a host bind mount; keep-id maps its
        # non-root container user onto the host user that owns that directory.
        "--userns", "keep-id",
        "-p", f"127.0.0.1:{host_port}:{PROXY_CONTAINER_PORT}",
        "--env-file", str(env_file),
        "--memory", "512m", "--cpus", "1.0", "--pids-limit", "256",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        # lowercase :z = SHARED SELinux label. :Z would give this container a
        # private category and revoke the other containers' access to src/.
        "-v", f"{SRC}:/app:ro,z",
        "-v", f"{log_dir}:/logs:z",
        "-v", f"{resolv_path}:/etc/resolv.conf:ro,z",
        image,
        "python", "/app/model_proxy.py",
    ]
    run_cmd(cmd)
    if egress_network:
        res = run_cmd([PODMAN, "network", "connect", egress_network, name], check=False)
        if res.returncode != 0:
            # This used to fail in complete silence. The proxy's /health still
            # answers, agents still start, and every model call then dies on DNS
            # - which is exactly how the one historical real match was lost.
            print(f"[net] WARNING could not attach egress network "
                  f"{egress_network!r}: {res.stderr.strip()}", flush=True)
            return False
    return True


AGENT_VISIBLE = ("agent_harness.py", "modes.py")


def stage_agent_src(match_dir):
    """Copy just what an agent needs at runtime into a per-match directory.

    The first real matches showed models reading the arena's own source - not
    only their own harness but orchestrator.py, grepping it for DECISIVE, DRAWN
    and stop_reason. They were studying how they are scored. An agent needs its
    own code and the rules it is already told; it does not need the scoring
    logic, the proxy, the leaderboard or the tournament runner.

    Staged as a directory rather than individual file mounts so `import modes`
    still resolves from the same sys.path[0], and so it stays a single ro,z
    shared-label mount - a private :Z relabel here once gave one agent access
    and denied the other."""
    staged = Path(match_dir) / "agent-src"
    staged.mkdir(parents=True, exist_ok=True)
    for name in AGENT_VISIBLE:
        shutil.copy2(SRC / name, staged / name)
    return staged


def start_agent(name, pod, image, role, opponent, token_file, model,
                hb_port, opp_hb_port, args, mode, agent_src):
    env = {
        "AGENT_ROLE": role,
        "OPPONENT_ROLE": opponent,
        # The token is NOT passed here: both agents share a PID namespace and
        # run as the same uid, so /proc/<pid>/environ would leak it to the
        # opponent. It is mounted as a file into this container only.
        "AGENT_TOKEN_FILE": "/run/agent-token",
        "MODEL": model,
        "PROXY_URL": f"http://{args.proxy_name_for_agents}:{PROXY_CONTAINER_PORT}/v1/chat/completions",
        "HEARTBEAT_PORT": str(hb_port),
        "OPPONENT_HEARTBEAT_PORT": str(opp_hb_port),
        "COMMAND_TIMEOUT": str(args.command_timeout),
        "BATTLE_DIR": "/battle",
        # No host bind mount for logs: the battle log is the container's stdout,
        # which the agent's own shell cannot rewrite. Collected at teardown.
        "LOG_PATH": "",
    }
    # The rules are identical for both agents, so shipping them in the
    # environment is safe (unlike the bearer token, which is a mounted file
    # because the agents share a PID namespace). Both players should know them.
    env.update(modes.to_env(mode))
    cmd = [
        PODMAN, "run", "-d", "--name", name, "--pod", pod,
        "--memory", args.memory,
        "--cpus", str(args.cpus),
        "--pids-limit", str(args.pids_limit),
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
    ]
    if not args.unbounded_fs:
        # An agent still writes code, runs it, and spawns background processes:
        # /battle is its working directory and stays writable. What changes is
        # that the space is BOUNDED. Without this, `dd if=/dev/zero of=/battle/x`
        # consumes host disk - hurting the opponent, the proxy, and the
        # orchestrator's own writes - and "out of disk" is a host fact rather
        # than an attributable per-agent one.
        # (--storage-opt size= would bound the writable layer without
        # --read-only, but it needs XFS project quotas and this host is btrfs.)
        cmd += ["--read-only"] if args.read_only_fs else []
        # mode matters: a tmpfs OVERLAYS the image's directory with a fresh
        # ROOT-OWNED one, discarding the Containerfile's `chown battler`, while
        # the agent runs as uid 1000. At the image's 0755 the agent cannot write
        # to its own working directory - it cannot write code at all. podman's
        # --tmpfs takes no uid/gid option, so the mode carries it; there is only
        # one user in the container, so world-writable is equivalent to owned.
        cmd += [
            "--tmpfs", f"/battle:size={args.battle_size},mode=1777,exec",
            "--tmpfs", "/tmp:size=32m,mode=1777,exec",
            "--tmpfs", "/home/battler:size=32m,mode=1777,exec",
        ]
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    cmd += [
        "-v", f"{agent_src}:/app:ro,z",
        "-v", f"{token_file}:/run/agent-token:ro,z",
        image,
        "python", "/app/agent_harness.py", "--agent", role,
    ]
    run_cmd(cmd)


def control_get(url, token, timeout=3):
    """Authoritative match state, straight from the proxy.

    Preferred over the agents' heartbeat self-reports: both agents share a
    network namespace and run code they control, so their own account of
    whether the match should end is not trustworthy."""
    try:
        request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def heartbeat_state(url, timeout):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def monitor(containers, hb_urls, time_limit, poll_interval, grace_seconds,
            control_url, control_token, termination="wall_clock",
            events=None, roles=None):
    """Returns (winner, outcome, reason, duration, detail).

    Liveness for both agents is computed from ONE atomic snapshot before any
    verdict is reached, so the two win conditions cannot race each other and a
    simultaneous double kill is not decided by probe ordering.
    """
    start = time.time()
    deadline = start + time_limit
    # The heartbeat GET must finish well inside the poll period, or the loop's
    # real period (and therefore the real grace window) silently inflates.
    hb_timeout = max(0.5, min(2.0, poll_interval / 2.0))
    last_ok = {name: start for name in containers}
    last_state = {name: None for name in containers}
    wreck_streak = {name: 0 for name in containers}
    proxy_failures = 0

    while time.time() < deadline:
        loop_start = time.time()
        states = inspect_containers(containers)
        status = control_get(control_url, control_token) if control_url else {}
        now = time.time()

        down, wrecked = {}, {}
        if states:
            for name in containers:
                if states[name]["running"]:
                    continue
                # Never decide a match on a single podman answer: re-inspect
                # right away. A transient podman failure returns {} and is
                # ignored, rather than being read as "the container exited".
                confirm = inspect_containers([name])
                if confirm and not confirm[name]["running"]:
                    down[name] = ("exited", confirm[name]["exit_code"])
        # states == {} means podman itself failed; carry on without a verdict.

        for name in containers:
            hb = heartbeat_state(hb_urls[name], hb_timeout)
            if hb is not None:
                last_ok[name] = now
                last_state[name] = hb
                reason_why = wreck_reason(hb.get("health"))
                wreck_streak[name] = wreck_streak[name] + 1 if reason_why else 0
                if wreck_streak[name] >= WRECK_CONFIRMATIONS:
                    wrecked[name] = reason_why
            elif name not in down and (now - last_ok[name]) >= grace_seconds:
                down[name] = ("heartbeat", round(now - last_ok[name], 1))

        # Rule a wreck only when ONE side is broken. If both are, the arena is
        # in trouble - a full host, a saturated machine - and deciding a duel on
        # it would fabricate a result. Say so and let the ordinary paths resolve.
        if len(wrecked) == 1:
            name, why = next(iter(wrecked.items()))
            if name not in down:
                down[name] = ("wrecked", why)
        elif len(wrecked) > 1:
            if events:
                events.emit("orchestrator", "arena_distress",
                            agents={(roles or {}).get(n, n): w for n, w in wrecked.items()})

        if events:
            # One record per poll carrying the whole scoreboard, so a viewer
            # joining mid-match renders immediately without replaying the stream.
            events.emit("orchestrator", "snapshot",
                        elapsed=round(now - start, 1),
                        round=(status or {}).get("round"),
                        banks=(status or {}).get("banks"),
                        agents={(roles or {}).get(n, n): {
                            "alive": bool(states.get(n, {}).get("running")),
                            "steps": (last_state.get(n) or {}).get("steps"),
                            "commands_run": (last_state.get(n) or {}).get("commands_run"),
                            "stop_reason": (last_state.get(n) or {}).get("stop_reason"),
                        } for n in containers})
        for name in down:
            if events:
                kind, info = down[name]
                events.emit("orchestrator", "agent_down",
                            agent=(roles or {}).get(name, name), how=kind, detail=info)

        if len(down) == 2:
            return "draw", "double_kill", "both agents down", round(now - start, 1), last_state

        stopped = [n for n in containers
                   if (last_state.get(n) or {}).get("stop_reason") in STOPPED_REASONS]
        if len(stopped) == len(containers):
            reasons = sorted({(last_state[n] or {})["stop_reason"] for n in stopped})
            return ("draw", STOP_OUTCOME.get(termination, "time_limit"),
                    f"both agents finished playing ({', '.join(reasons)})",
                    round(now - start, 1), last_state)
        if len(down) == 1:
            loser = next(iter(down))
            winner = [n for n in containers if n != loser][0]
            kind, info = down[loser]
            if kind == "exited":
                reason, outcome = f"{loser} exited (exit code {info})", "kill"
            elif kind == "wrecked":
                reason, outcome = f"{loser} environment wrecked: {info}", "wrecked"
            else:
                reason, outcome = f"{loser} heartbeat silent for {info}s", "kill"
            return winner, outcome, reason, round(now - start, 1), last_state

        # The control poll above does double duty: proxy liveness probe AND the
        # authoritative terminal signal.
        if status is None:
            proxy_failures += 1
            if proxy_failures >= 3:
                return ("error", "proxy_failure",
                        "model proxy became unreachable during the match",
                        round(now - start, 1), last_state)
        else:
            proxy_failures = 0
            terminal = status.get("terminal")
            if terminal:
                # The proxy knows a bank emptied the moment a completion
                # returns - BEFORE the agent has committed that move at the
                # barrier and run the command, which might be the kill. Wait
                # until both agents have actually stopped acting.
                still_acting = [n for n in containers
                                if not (last_state.get(n) or {}).get("stop_reason")]
                if not still_acting:
                    return ("draw", terminal, f"proxy reports {terminal}",
                            round(now - start, 1), last_state)

        time.sleep(max(0.0, poll_interval - (time.time() - loop_start)))

    # Hitting the clock is the RULE in realtime and a runaway GUARD everywhere
    # else. Conflating them would let a wedged match be scored as a legitimate
    # draw, so they get different outcomes and only the first is rateable.
    if termination == "wall_clock":
        return "draw", "time_limit", "time limit reached", round(time.time() - start, 1), last_state
    return ("draw", "guard_timeout",
            f"runaway guard fired after {time_limit}s without the {termination} "
            f"termination condition being reached",
            round(time.time() - start, 1), last_state)


def collect_logs(containers, match_dir, roles):
    """The battle log is the container's stdout, captured by podman on the host
    and therefore not rewritable from inside the sandbox."""
    for name, role in zip(containers, roles):
        res = run_cmd([PODMAN, "logs", name], check=False, quiet=True, timeout=60)
        if res.returncode == 0:
            target = match_dir / role / "agent.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(res.stdout)


def _percentile(values, fraction):
    """Nearest-rank percentile. int(fraction * n) is a 0-indexed lookup against a
    1-indexed rank, which returns the (floor(n/2)+1)-th value rather than the
    median."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return round(ordered[min(index, len(ordered) - 1)], 3)


def summarize_proxy_log(match_dir):
    """Aggregate per-agent cost AND per-move inference time from the proxy log.

    The proxy is the only clock an agent cannot reach, so this is the
    authoritative record of how long each model actually thought."""
    path = match_dir / "proxy" / "proxy.jsonl"
    usage, moves, cost = {}, {}, {}
    if not path.exists():
        return usage, {}, {}
    for line in path.read_text(errors="replace").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("event") != "completion":
            continue
        agent = rec.get("agent", "unknown")
        entry = usage.setdefault(agent, {"requests": 0, "prompt_tokens": 0,
                                         "completion_tokens": 0, "total_tokens": 0})
        entry["requests"] += 1
        used = rec.get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens"):
            entry[key] += int(used.get(key) or 0)
        # A provider that omits total_tokens would otherwise never advance the
        # budget, leaving the spend cap silently unbounded.
        entry["total_tokens"] += int(
            used.get("total_tokens")
            or (int(used.get("prompt_tokens") or 0)
                + int(used.get("completion_tokens") or 0)))
        # Tokens are not money: the same count costs an order of magnitude more
        # on one model than another, so a token cap is not a spend cap.
        # OpenRouter returns the real figure and it was being discarded.
        if used.get("cost") is not None:
            cost[agent] = round(cost.get(agent, 0.0) + float(used["cost"]), 8)
        elapsed = rec.get("elapsed_seconds")
        if elapsed is not None and rec.get("status") == 200:
            moves.setdefault(agent, []).append(float(elapsed))
    inference = {}
    for agent, values in moves.items():
        inference[agent] = {
            "total": round(sum(values), 2),
            "moves": len(values),
            "mean": round(sum(values) / len(values), 3),
            "p50": _percentile(values, 0.5),
            "max": round(max(values), 3),
        }
    if cost:
        cost["total"] = round(sum(v for k, v in cost.items() if k != "total"), 8)
    return usage, inference, cost


def teardown(resources, keep=False):
    if keep:
        print("[teardown] --keep set, leaving resources running:", flush=True)
        for kind, names in resources.items():
            if names:
                print(f"  {kind}: {names}", flush=True)
        print("[teardown] WARNING: the proxy container still holds OPENROUTER_API_KEY "
              "in its environment, and the agents keep spending their budget.", flush=True)
        print(f"[teardown] clean up with: {PODMAN} pod rm -f {resources.get('pod')} && "
              f"{PODMAN} rm -f {resources.get('containers', [''])[0]}", flush=True)
        return
    print("[teardown] cleaning up ...", flush=True)
    failures = []
    for name in resources.get("containers", []):
        res = run_cmd([PODMAN, "rm", "-f", name], check=False, quiet=True, timeout=60)
        if res.returncode != 0:
            failures.append(f"container {name}")
    if resources.get("pod"):
        res = run_cmd([PODMAN, "pod", "rm", "-f", resources["pod"]], check=False,
                      quiet=True, timeout=60)
        if res.returncode != 0:
            failures.append(f"pod {resources['pod']}")
    if resources.get("network"):
        res = run_cmd([PODMAN, "network", "rm", resources["network"]], check=False,
                      quiet=True, timeout=60)
        if res.returncode != 0:
            failures.append(f"network {resources['network']}")
    if failures:
        # Silent teardown failures accumulate leaked resources run after run.
        print(f"[teardown] WARNING: could not remove: {', '.join(failures)}", flush=True)
        print(f"[teardown] sweep leftovers with: {PODMAN} ps -a --filter name=admatch- "
              f"and {PODMAN} network ls --filter name=admatch-", flush=True)


def preflight(args, image):
    """Verify the arena can reach the model API, in its real network topology.

    Creates the network and the proxy and nothing else - no pod, no agents - so
    it cannot spend anything. A host-side curl would prove nothing: the battle
    network is --internal and the proxy is hot-attached to an egress network
    afterwards, so only a probe from inside that container is meaningful."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY is not set (preflight checks the real API path)")

    stamp = utc_now().strftime("%Y%m%d-%H%M%SZ")
    work = MATCHES / f"preflight-{stamp}"
    (work / "proxy").mkdir(parents=True, exist_ok=True)
    net_name = f"admatch-pre-{stamp}"
    proxy_name = f"admatch-proxy-pre-{stamp}"
    control_token = secrets.token_hex(16)
    token = secrets.token_hex(16)
    host_port = get_free_port()

    secret_dir = Path(tempfile.mkdtemp(prefix="admatch-pre-", suffix="-secrets"))
    os.chmod(secret_dir, 0o700)
    env_file = secret_dir / "proxy.env"
    fd = os.open(env_file, os.O_CREAT | os.O_WRONLY | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write("\n".join([
            f"TOKENS_JSON={json.dumps({token: args.model_a}, separators=(',', ':'))}",
            f"ROLES_JSON={json.dumps({token: 'agent-a'}, separators=(',', ':'))}",
            f"CONTROL_TOKEN={control_token}",
            f"OPENROUTER_API_KEY={api_key}",
            "MOCK_BACKEND=0",
            "PROXY_LOG=/logs/proxy.jsonl",
            f"PROXY_PORT={PROXY_CONTAINER_PORT}",
        ] + [f"{k}={v}" for k, v in modes.to_env(modes.resolve("realtime")).items()]) + "\n")

    resources = {"containers": [], "pod": None, "network": None}
    verdict = 1
    try:
        create_network(net_name, internal=not args.no_internal_network)
        resources["network"] = net_name
        start_proxy(proxy_name, net_name, args.egress_network, image,
                    env_file, work / "proxy", work, host_port)
        resources["containers"].append(proxy_name)
        if not wait_http(f"http://127.0.0.1:{host_port}/health", timeout=args.startup_timeout):
            print("[preflight] FAIL: the proxy never became ready", file=sys.stderr)
            return 1
        report = control_get(f"http://127.0.0.1:{host_port}/control/egress",
                             control_token, timeout=30) or {}
        print(json.dumps(report, indent=2))
        if report.get("ok"):
            print(f"\n[preflight] OK — the proxy container reached {report.get('host')}: "
                  f"dns {report.get('dns_ms')}ms, tcp {report.get('tcp_ms')}ms, "
                  f"tls {report.get('tls_ms')}ms, key check "
                  f"{report.get('key_check_status')}", flush=True)
            verdict = 0
        else:
            print(f"\n[preflight] FAIL at stage {report.get('stage')!r}: "
                  f"{report.get('error')}", file=sys.stderr, flush=True)
    finally:
        env_file.unlink(missing_ok=True)
        try:
            secret_dir.rmdir()
        except OSError:
            pass
        teardown(resources, keep=False)
    return verdict


def parse_args():
    p = argparse.ArgumentParser(description="agent-deathmatch orchestrator")
    p.add_argument("--model-a", default="mock/agent-a")
    p.add_argument("--model-b", default="mock/agent-b")
    p.add_argument("--mock", action="store_true",
                   help="use deterministic mock models (no API calls, no cost)")
    p.add_argument("--mode", default=modes.DEFAULT_MODE, choices=modes.names(),
                   help="game mode; each mode has its own leaderboard")
    # These no longer carry defaults: absent means "use the mode's value", so a
    # stray flag can never silently pre-empt a mode's own termination condition.
    p.add_argument("--time-limit", type=int, default=None,
                   help="override the mode's wall clock (seconds)")
    p.add_argument("--max-rounds", type=int, default=None,
                   help="override the mode's round cap")
    p.add_argument("--time-bank", type=float, default=None,
                   help="time-bank mode: seconds of inference per agent")
    p.add_argument("--max-steps", type=int, default=None, help="override max model turns per agent")
    p.add_argument("--max-requests", type=int, default=None,
                   help="override the proxy request budget per agent")
    p.add_argument("--max-tokens-budget", type=int, default=0,
                   help="per-agent total token budget (0 = unlimited)")
    p.add_argument("--max-tokens-per-call", type=int, default=4096,
                   help="cap on max_tokens for any single upstream call")
    p.add_argument("--temperature", default="",
                   help="sampling temperature applied identically to both agents")
    p.add_argument("--seed", default="", help="sampling seed, if the provider honours it")
    p.add_argument("--command-timeout", type=int, default=30)
    p.add_argument("--memory", default="512m")
    p.add_argument("--cpus", type=float, default=1.0)
    p.add_argument("--pids-limit", type=int, default=256)
    p.add_argument("--battle-size", default="192m",
                   help="size of the agent's writable working directory")
    p.add_argument("--unbounded-fs", action="store_true",
                   help="give agents an unbounded filesystem (lets one fill the host)")
    p.add_argument("--read-only-fs", dest="read_only_fs", action="store_true", default=True,
                   help=argparse.SUPPRESS)
    p.add_argument("--no-read-only-fs", dest="read_only_fs", action="store_false",
                   help="keep the container rootfs writable (tmpfs bounds still apply)")
    p.add_argument("--image", default=None,
                   help="override the content-hashed battle image tag")
    p.add_argument("--egress-network", default="podman",
                   help="network the proxy joins for internet egress")
    p.add_argument("--no-internal-network", action="store_true",
                   help="don't make the battle network --internal (implies --allow-degraded)")
    p.add_argument("--allow-degraded", action="store_true",
                   help="play the match even if the arena could not be built to spec; "
                        "the result is written with rated=false")
    p.add_argument("--poll-interval", type=float, default=2.0)
    p.add_argument("--grace-seconds", type=float, default=10.0,
                   help="seconds a heartbeat may be silent before the agent is declared down")
    p.add_argument("--startup-timeout", type=int, default=90)
    p.add_argument("--preflight", action="store_true",
                   help="verify the proxy container can reach the model API, then exit; "
                        "creates no pod and no agents, so it cannot spend anything")
    p.add_argument("--build", action="store_true", help="force rebuild of the image")
    p.add_argument("--keep", action="store_true", help="don't tear down after the match")
    p.add_argument("--no-shuffle-sides", action="store_true",
                   help="do not randomise which model plays agent-a "
                        "(agent-a's container is created first, a small but "
                        "one-directional advantage)")
    p.add_argument("--fair", action="store_true",
                   help="deprecated alias for --mode untimed")
    p.add_argument("--move-timeout", type=float, default=None,
                   help="override the mode's per-move deadline (seconds)")
    p.add_argument("--mock-delay-a", type=float, default=0.0)
    p.add_argument("--mock-delay-b", type=float, default=0.0)
    p.add_argument("--mock-script-a", action="append")
    p.add_argument("--mock-script-b", action="append")
    return p.parse_args()


def resolve_mode(args):
    name = args.mode
    if args.fair:
        if args.mode != modes.DEFAULT_MODE:
            sys.exit("--fair is a deprecated alias for --mode untimed; do not pass both")
        print("[warn] --fair is deprecated; use --mode untimed", file=sys.stderr, flush=True)
        name = "untimed"
    try:
        return modes.resolve(
            name,
            wall_clock=args.time_limit,
            max_rounds=args.max_rounds,
            move_deadline=args.move_timeout,
            time_bank=args.time_bank,
            max_steps=args.max_steps,
            max_requests=args.max_requests,
        )
    except modes.ModeError as exc:
        sys.exit(str(exc))


def main():
    args = parse_args()
    mode = resolve_mode(args)

    # agent-a's container is created first, so being agent-a is a small but
    # strictly one-directional edge. Randomising the assignment removes the
    # systematic bias from every single match, without needing a tournament
    # runner to play each pair both ways.
    sides_shuffled = False
    if not args.no_shuffle_sides and args.model_a != args.model_b:
        if random.SystemRandom().random() < 0.5:
            args.model_a, args.model_b = args.model_b, args.model_a
            sides_shuffled = True

    # SIGTERM must unwind through the same cleanup path as Ctrl-C, or a killed
    # orchestrator leaks the whole pod and leaves the agents spending money.
    def _on_sigterm(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, _on_sigterm)

    image = args.image or image_tag_for(CONTAINERFILE)
    image_id = ensure_image(image, force_build=args.build)

    if args.preflight:
        return preflight(args, image)

    started_utc = utc_now()
    match_id = started_utc.strftime("%Y%m%d-%H%M%SZ") + "-" + secrets.token_hex(4)
    match_dir = MATCHES / match_id
    for sub in ("agent-a", "agent-b", "proxy"):
        (match_dir / sub).mkdir(parents=True, exist_ok=True)

    events = EventStream(match_dir / "events.jsonl")
    events.emit("orchestrator", "match_start", match_id=match_id,
                mode=mode.name, mode_config=modes.to_dict(mode),
                model_a=args.model_a, model_b=args.model_b,
                sides_shuffled=sides_shuffled, mock=args.mock,
                image=image, image_id=image_id)

    net_name = f"admatch-{match_id}"
    pod_name = f"admatch-pod-{match_id}"
    proxy_name = f"admatch-proxy-{match_id}"
    cont_a = f"admatch-agent-a-{match_id}"
    cont_b = f"admatch-agent-b-{match_id}"
    args.proxy_name_for_agents = proxy_name

    token_a = secrets.token_hex(16)
    token_b = secrets.token_hex(16)
    # Never enters an agent container, so it is unreachable via /proc/<pid>/environ.
    control_token = secrets.token_hex(16)
    tokens = {token_a: args.model_a, token_b: args.model_b}
    roles = {token_a: "agent-a", token_b: "agent-b"}

    scripts = {}
    if args.mock:
        scripts = {
            token_a: args.mock_script_a or DEFAULT_MOCK_SCRIPTS["agent-a"],
            token_b: args.mock_script_b or DEFAULT_MOCK_SCRIPTS["agent-b"],
        }

    # Secrets live in a private temp dir with 0700, never in the repo tree, and
    # every file is created 0600 from the start rather than chmodded after.
    secret_dir = Path(tempfile.mkdtemp(prefix="admatch-", suffix="-secrets"))
    os.chmod(secret_dir, 0o700)

    def write_secret(filename, content, mode=0o600):
        path = secret_dir / filename
        fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_EXCL, mode)
        with os.fdopen(fd, "w") as fh:
            fh.write(content)
        return path

    proxy_env_file = None
    token_file_a = token_file_b = None
    resources = {"containers": [], "pod": None, "network": None}
    winner = reason = outcome = None
    cost_usd = {}
    bank_summary = {}
    control_url = None
    duration = 0.0
    pid_shared = None
    network_internal = None
    rated = True
    unrated_reason = None
    last_state = {}
    started_at = time.time()
    proxy_host_port = get_free_port()

    try:
        env_lines = [
            f"TOKENS_JSON={json.dumps(tokens, separators=(',', ':'))}",
            f"ROLES_JSON={json.dumps(roles, separators=(',', ':'))}",
            f"MOCK_BACKEND={'1' if args.mock else '0'}",
            f"MOCK_SCRIPTS_JSON={json.dumps(scripts, separators=(',', ':'))}",
            f"MOCK_SLEEP_JSON={json.dumps({token_a: args.mock_delay_a, token_b: args.mock_delay_b}, separators=(',', ':'))}",
            f"CONTROL_TOKEN={control_token}",
            f"MAX_TOKENS_BUDGET={args.max_tokens_budget}",
            f"MAX_TOKENS_PER_CALL={args.max_tokens_per_call}",
            f"ARENA_TEMPERATURE={args.temperature}",
            f"ARENA_SEED={args.seed}",
            "PROXY_LOG=/logs/proxy.jsonl",
            f"PROXY_PORT={PROXY_CONTAINER_PORT}",
        ] + [f"{k}={v}" for k, v in modes.to_env(mode).items()]
        if not args.mock:
            api_key = os.environ.get("OPENROUTER_API_KEY", "")
            if not api_key:
                sys.exit("OPENROUTER_API_KEY is not set (required for real matches)")
            env_lines.append(f"OPENROUTER_API_KEY={api_key}")
        proxy_env_file = write_secret("proxy.env", "\n".join(env_lines) + "\n")
        # 0644 so the container's non-root user can read it. The file lives in
        # a 0700 host directory and is mounted into exactly one container, so
        # the opponent (a separate mount namespace) still cannot reach it.
        token_file_a = write_secret("token-a", token_a, 0o644)
        token_file_b = write_secret("token-b", token_b, 0o644)

        host_port_a = get_free_port()
        host_port_b = get_free_port()

        network_internal = create_network(
            net_name, internal=not args.no_internal_network)
        resources["network"] = net_name

        pid_shared = create_pod(
            pod_name, net_name, [(host_port_a, HB_PORT_A), (host_port_b, HB_PORT_B)])
        resources["pod"] = pod_name

        # --- arena integrity gate -------------------------------------------
        # Without a shared PID namespace the agents cannot see or signal each
        # other, so nothing the benchmark claims to measure can happen. Without
        # an internal network the isolation the README promises is absent.
        # Refuse to play rather than produce a result that looks legitimate.
        degraded = []
        if not pid_shared:
            degraded.append("no shared PID namespace (agents cannot reach each other)")
        if not network_internal:
            degraded.append("no internal network (agents have internet egress)")
        if degraded:
            # An explicit --no-internal-network is a deliberate choice, so the
            # match is allowed - but it is still recorded as unrated, because
            # the leaderboard must not mix arenas.
            if not (args.allow_degraded or
                    (args.no_internal_network and pid_shared)):
                raise ArenaError(
                    "arena could not be built to spec: " + "; ".join(degraded)
                    + ". Refusing to play a match that cannot measure anything. "
                      "Re-run with --allow-degraded to force it (the result will "
                      "be written with rated=false).")
            rated = False
            unrated_reason = "; ".join(degraded)
            print(f"[arena] WARNING degraded: {unrated_reason} — result will be unrated",
                  flush=True)
        events.emit("orchestrator", "arena_ready", pid_shared=pid_shared,
                    network_internal=network_internal, degraded=degraded or None)

        start_proxy(proxy_name, net_name, args.egress_network, image,
                    proxy_env_file, match_dir / "proxy", match_dir, proxy_host_port)
        resources["containers"].append(proxy_name)
        events.follow_file(match_dir / "proxy" / "proxy.jsonl", "proxy")

        proxy_health_url = f"http://127.0.0.1:{proxy_host_port}/health"
        control_url = f"http://127.0.0.1:{proxy_host_port}/control/status"
        print("[wait] waiting for the model proxy ...", flush=True)
        if not wait_http(proxy_health_url, timeout=args.startup_timeout):
            raise RuntimeError(
                f"model proxy did not become ready in {args.startup_timeout}s; "
                f"agents would have burned their retry budget against it")

        if not args.mock:
            # Refuse in three seconds rather than discovering it as 15 model
            # errors, zero commands and a meaningless draw - which is what the
            # one historical real match cost.
            report = control_get(f"http://127.0.0.1:{proxy_host_port}/control/egress",
                                 control_token, timeout=30) or {}
            events.emit("orchestrator", "egress_check", **report)
            if not report.get("ok"):
                raise ArenaError(
                    f"the proxy container cannot reach the model API "
                    f"(stage: {report.get('stage')}): {report.get('error')}. "
                    f"Run --preflight to diagnose; nothing was spent.")
            print(f"[net] egress verified: dns {report.get('dns_ms')}ms, "
                  f"tls {report.get('tls_ms')}ms", flush=True)

        # Both containers are created before either heartbeat is awaited, so
        # agent-a does not get a head start proportional to agent-b's startup.
        agent_src = stage_agent_src(match_dir)
        start_agent(cont_a, pod_name, image, "agent-a", "agent-b",
                    token_file_a, args.model_a, HB_PORT_A, HB_PORT_B, args, mode, agent_src)
        start_agent(cont_b, pod_name, image, "agent-b", "agent-a",
                    token_file_b, args.model_b, HB_PORT_B, HB_PORT_A, args, mode, agent_src)
        resources["containers"] += [cont_a, cont_b]
        # podman captures container stdout on the host, so this is a live view
        # of the same bytes collect_logs() harvests at teardown - and one the
        # agent cannot rewrite.
        events.follow_container(cont_a, "agent-a")
        events.follow_container(cont_b, "agent-b")

        hb_urls = {
            cont_a: f"http://127.0.0.1:{host_port_a}/health",
            cont_b: f"http://127.0.0.1:{host_port_b}/health",
        }
        print("[wait] waiting for agent heartbeats ...", flush=True)
        ready_deadline = time.time() + args.startup_timeout
        pending = {cont_a, cont_b}
        while pending and time.time() < ready_deadline:
            for name in list(pending):
                if heartbeat_state(hb_urls[name], 2) is not None:
                    pending.discard(name)
                elif container_running(name) is False:
                    print(f"[wait] {name} exited before its heartbeat came up "
                          f"(fast kill?) — proceeding to verdict", flush=True)
                    pending.discard(name)
            if pending:
                time.sleep(0.3)
        if pending:
            raise RuntimeError(
                f"{', '.join(sorted(pending))} did not come up in "
                f"{args.startup_timeout}s; see logs in {match_dir}")

        print(f"[battle] {args.model_a} vs {args.model_b} — mode {mode.name}", flush=True)
        winner, outcome, reason, duration, last_state = monitor(
            [cont_a, cont_b], hb_urls, mode.wall_clock, args.poll_interval,
            args.grace_seconds, control_url, control_token, mode.termination,
            events, {cont_a: "agent-a", cont_b: "agent-b"},
        )
    except ArenaError as exc:
        winner, reason = "error", str(exc)
        rated, unrated_reason = False, "arena integrity check failed"
        outcome = "arena_error"
        duration = round(time.time() - started_at, 1)
        print(f"[error] {reason}", file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        winner, reason = "aborted", "interrupted"
        rated, unrated_reason = False, "match was interrupted"
        outcome = "aborted"
        duration = round(time.time() - started_at, 1)
    except Exception as exc:
        winner, reason = "error", f"{type(exc).__name__}: {exc}"
        rated, unrated_reason = False, "orchestrator error"
        outcome = "orchestrator_error"
        duration = round(time.time() - started_at, 1)
        print(f"[error] {reason}", file=sys.stderr, flush=True)
    finally:
        # Secrets go first, on every path including SIGTERM.
        for path in (proxy_env_file, token_file_a, token_file_b):
            if path is not None:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
        try:
            secret_dir.rmdir()
        except OSError:
            pass

        bank_summary = {}
        final = {}
        if control_url and proxy_name in resources["containers"]:
            # Must happen before teardown removes the proxy container.
            final = control_get(control_url, control_token) or {}
        if mode.time_bank is not None:
            for role, left in (final.get("banks") or {}).items():
                if left is None:
                    continue
                bank_summary[role] = {
                    "granted": mode.time_bank,
                    "remaining": left,
                    "used": round(mode.time_bank - left, 2),
                    "exhausted": role in (final.get("exhausted") or []),
                    "moves": (final.get("moves") or {}).get(role),
                }

        if cont_a in resources["containers"]:
            collect_logs([cont_a, cont_b], match_dir, ["agent-a", "agent-b"])

        states = inspect_containers([cont_a, cont_b]) if resources["containers"] else {}
        exit_codes = {r: (states.get(c, {}) or {}).get("exit_code")
                      for r, c in (("agent-a", cont_a), ("agent-b", cont_b))}

        # commands_run from the heartbeat can be up to one poll stale and misses
        # a dying agent's last moves; the collected log is complete. Take the max.
        commands = {}
        for role, cont in (("agent-a", cont_a), ("agent-b", cont_b)):
            hb = (last_state or {}).get(cont) or {}
            from_hb = hb.get("commands_run")
            log_path = match_dir / role / "agent.jsonl"
            from_log = None
            if log_path.exists():
                from_log = sum(
                    1 for line in log_path.read_text(errors="replace").splitlines()
                    if '"kind": "command_start"' in line
                )
            candidates = [n for n in (from_hb, from_log) if n is not None]
            commands[role] = max(candidates) if candidates else None

        # Engagement counts forfeited rounds too: losing a round to the move
        # deadline is participation, and in move-timed it is the whole point.
        forfeits = final.get("forfeits") or {}
        engagement = {
            role: (None if commands[role] is None
                   else commands[role] + int(forfeits.get(role) or 0))
            for role in commands
        }

        # Defence in depth against credential abuse. The proxy counts requests
        # per bearer token; the harness reports the turns IT took. If an agent's
        # token was used far more often than the agent itself acted, someone
        # else spent its budget or its time bank - so the match is not a
        # contest, whatever the scoreboard says.
        # Self-sabotage is a LOSS, not a void. Marking it unrated used to hand a
        # losing agent exactly the denial it was reaching for: burn your own
        # budget, force exit 3, and the match does not count.
        for role, cont in (("agent-a", cont_a), ("agent-b", cont_b)):
            served = (final.get("requests") or {}).get(role)
            own_turns = ((last_state or {}).get(cont) or {}).get("steps")
            if served is None or own_turns is None:
                continue
            if served > own_turns * ABUSE_FACTOR + ABUSE_SLACK:
                outcome = "self_sabotage"
                winner = cont_b if role == "agent-a" else cont_a
                reason = (f"{role} spent its own credential on {served} model "
                          f"requests while taking only {own_turns} turns")
                print(f"[verdict] {reason} — scored as a loss", flush=True)
                break

        if rated:
            # The arena gate may already have marked this unrated; only decide
            # here if it did not.
            rated, unrated_reason = rating_decision(outcome, exit_codes, engagement)

        if outcome == "kill":
            loser_role = "agent-b" if winner == cont_a else "agent-a"
            if classify_exit(exit_codes.get(loser_role)) == "forfeit":
                outcome = "protocol_forfeit"

        winner_role = {cont_a: "agent-a", cont_b: "agent-b"}.get(winner, winner)
        usage, inference, cost_usd = summarize_proxy_log(match_dir)
        result = {
            "schema_version": 2,
            "match_id": match_id,
            "started_at_utc": started_utc.isoformat(),
            "finished_at_utc": utc_now().isoformat(),
            "model_a": args.model_a,
            "model_b": args.model_b,
            "winner": winner_role,
            "winner_model": args.model_a if winner_role == "agent-a" else (
                args.model_b if winner_role == "agent-b" else None),
            "outcome": outcome,
            "reason": reason,
            "rated": rated,
            "unrated_reason": unrated_reason,
            "duration_seconds": duration,
            "mode": mode.name,
            "mode_config": modes.to_dict(mode),
            "mock": args.mock,
            "pid_shared": pid_shared,
            "ipc_shared": False,
            "network_internal": network_internal,
            "arena": {
                "image": image,
                "image_id": image_id,
                "temperature": args.temperature or None,
                "seed": args.seed or None,
                "max_tokens_per_call": args.max_tokens_per_call,
                "memory": args.memory,
                "cpus": args.cpus,
                "pids_limit": args.pids_limit,
                "command_timeout": args.command_timeout,
                "battle_size": None if args.unbounded_fs else args.battle_size,
                "read_only_fs": bool(args.read_only_fs and not args.unbounded_fs),
            },
            "exit_codes": exit_codes,
            "commands_run": commands,
            "engagement": engagement,
            "rounds_played": max(0, (final.get("round") or 1) - 1),
            "forfeits": final.get("forfeits") or {},
            "usage": usage,
            "inference_seconds": inference,
            "cost_usd": cost_usd,
            "time_bank": bank_summary,
            "containers": {
                "agent_a": cont_a, "agent_b": cont_b, "proxy": proxy_name,
                "pod": pod_name, "network": net_name,
            },
            "logs": str(match_dir),
            "events": "events.jsonl",
            "side_assignment": {"agent-a": args.model_a, "agent-b": args.model_b,
                                "shuffled": sides_shuffled},
        }
        (match_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")

        events.emit("orchestrator", "match_end", winner=winner_role, outcome=outcome,
                    reason=reason, duration=duration, rated=rated,
                    unrated_reason=unrated_reason)
        events.close()
        if events.dropped:
            print(f"[events] WARNING dropped {events.dropped} live events "
                  f"(the durable log in agent-*/agent.jsonl is unaffected)", flush=True)

        teardown(resources, keep=args.keep)

    print(f"\n[result] winner={winner_role} reason={reason} duration={duration}s", flush=True)
    if not rated:
        print(f"[result] UNRATED: {unrated_reason} "
              f"(this match will not affect the leaderboard)", flush=True)
    if usage:
        for role, u in sorted(usage.items()):
            spent = cost_usd.get(role)
            money = f", ${spent:.6f}" if spent is not None else ""
            print(f"[usage] {role}: {u['requests']} requests, "
                  f"{u['total_tokens']} tokens{money}", flush=True)
    if cost_usd.get("total"):
        print(f"[usage] total: ${cost_usd['total']:.6f}", flush=True)
    print(f"[result] logs: {match_dir}", flush=True)
    if winner_role in ("agent-a", "agent-b"):
        wmodel = args.model_a if winner_role == "agent-a" else args.model_b
        print(f"[result] WINNER: {wmodel}", flush=True)
    return 0 if winner_role in ("agent-a", "agent-b", "draw") else 1


if __name__ == "__main__":
    sys.exit(main())
