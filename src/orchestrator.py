#!/usr/bin/env python3
"""agent-deathmatch orchestrator.

Runs one match between two AI models:

    python3 src/orchestrator.py --mock                      # zero-cost pipeline test
    python3 src/orchestrator.py --model-a openai/gpt-4o-mini \
                                --model-b anthropic/claude-3.5-haiku --time-limit 600

Lifecycle:
  1. build/reuse the battle image
  2. create a private podman network (internal = no internet egress when supported)
  3. start the model proxy container (the only component with egress, via the
     default network; holds the OPENROUTER_API_KEY so agents never see it)
  4. create a pod with shared net+pid+ipc+uts namespaces
  5. start agent-a and agent-b containers in the pod
  6. monitor: first agent whose container exits or whose heartbeat dies loses
  7. write matches/<id>/result.json, collect logs, tear everything down
"""

import argparse
import json
import os
import secrets
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
MATCHES = ROOT / "matches"
CONTAINERFILE = ROOT / "container" / "Containerfile"

PODMAN = os.environ.get("PODMAN_BIN", "podman")
DEFAULT_IMAGE = "agent-deathmatch:latest"
PROXY_CONTAINER_PORT = 8080
HB_PORT_A = 8081
HB_PORT_B = 8082

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


def run_cmd(cmd, check=True, quiet=False):
    cmd = [str(c) for c in cmd]
    if not quiet:
        print("$ " + " ".join(cmd), flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
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
        time.sleep(1)
    return False


def wait_agent_ready(name, url, timeout=90):
    """Wait for an agent's heartbeat. Returns 'up', 'dead' (container exited
    before its heartbeat ever responded — possible in a fast kill), or
    'timeout'."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not container_running(name):
            return "dead"
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return "up"
        except Exception:
            pass
        time.sleep(0.3)
    return "timeout"


def container_running(name):
    res = run_cmd(
        [PODMAN, "container", "inspect", "-f", "{{.State.Running}}", name],
        check=False,
        quiet=True,
    )
    if res.returncode != 0:
        return False
    return res.stdout.strip() == "true"


def container_exit_code(name):
    res = run_cmd(
        [PODMAN, "container", "inspect", "-f", "{{.State.ExitCode}}", name],
        check=False,
        quiet=True,
    )
    if res.returncode != 0:
        return None
    try:
        return int(res.stdout.strip())
    except ValueError:
        return None


def ensure_image(image, force_build=False):
    res = run_cmd([PODMAN, "image", "exists", image], check=False, quiet=True)
    if res.returncode == 0 and not force_build:
        print(f"[image] {image} already present", flush=True)
        return
    print(f"[image] building {image} ...", flush=True)
    run_cmd([PODMAN, "build", "-t", image, "-f", str(CONTAINERFILE), str(CONTAINERFILE.parent)])


def create_network(name, internal):
    if internal:
        res = run_cmd([PODMAN, "network", "create", "--internal", name], check=False)
        if res.returncode == 0:
            print("[net] created internal (no internet egress) network", flush=True)
            return True
        print(f"[net] internal network failed, falling back to regular: {res.stderr.strip()}", flush=True)
    run_cmd([PODMAN, "network", "create", name])
    return False


def create_pod(name, network, ports):
    """Create pod with shared net+pid+ipc+uts. Falls back to no-pid if unsupported."""
    base = [
        PODMAN, "pod", "create", "--name", name,
        "--network", network,
    ]
    for host_port, cont_port in ports:
        base += ["-p", f"127.0.0.1:{host_port}:{cont_port}"]
    res = run_cmd(base + ["--share", "net,pid,ipc,uts"], check=False)
    if res.returncode == 0:
        print("[pod] created with shared net,pid,ipc,uts", flush=True)
        return True
    print(f"[pod] pid-sharing failed ({res.stderr.strip()}); retrying without pid", flush=True)
    run_cmd(base + ["--share", "net,ipc,uts"])
    return False


def start_proxy(name, network, egress_network, image, env_file, log_dir, match_dir):
    # The proxy must resolve external hostnames (openrouter.ai). When the battle
    # network is --internal, its aardvark-dns (first in resolv.conf) does not
    # forward external queries, and container-level --dns flags are ignored by
    # netavark on dns-enabled networks. Mount an explicit resolv.conf instead.
    proxy_dns = [d for d in os.environ.get("PROXY_DNS", "1.1.1.1,8.8.8.8").split(",") if d]
    resolv_path = Path(match_dir) / "proxy-resolv.conf"
    resolv_path.write_text("".join(f"nameserver {dns}\n" for dns in proxy_dns))

    cmd = [
        PODMAN, "run", "-d", "--name", name,
        "--network", network,
        "--env-file", str(env_file),
        "-v", f"{SRC}:/app:ro,Z",
        "-v", f"{log_dir}:/logs:Z",
        "-v", f"{resolv_path}:/etc/resolv.conf:ro,Z",
        image,
        "python", "/app/model_proxy.py",
    ]
    run_cmd(cmd)
    if egress_network:
        run_cmd([PODMAN, "network", "connect", egress_network, name], check=False)


def start_agent(name, pod, image, role, opponent, token, model,
                hb_port, opp_hb_port, log_dir, args):
    env = {
        "AGENT_ROLE": role,
        "OPPONENT_ROLE": opponent,
        "AGENT_TOKEN": token,
        "MODEL": model,
        "PROXY_URL": f"http://{args.proxy_name_for_agents}:{PROXY_CONTAINER_PORT}/v1/chat/completions",
        "HEARTBEAT_PORT": str(hb_port),
        "OPPONENT_HEARTBEAT_PORT": str(opp_hb_port),
        "MAX_STEPS": str(args.max_steps),
        "COMMAND_TIMEOUT": str(args.command_timeout),
        "LOCKSTEP": "1" if args.fair else "0",
        "ROUND_TIMEOUT": str(args.move_timeout),
        "BATTLE_DIR": "/battle",
        "LOG_PATH": "/logs/agent.jsonl",
    }
    cmd = [
        PODMAN, "run", "-d", "--name", name, "--pod", pod,
        "--memory", args.memory,
        "--cpus", str(args.cpus),
        "--pids-limit", str(args.pids_limit),
    ]
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    cmd += [
        "-v", f"{SRC}:/app:ro,Z",
        "-v", f"{log_dir}:/logs:Z",
        image,
        "python", "/app/agent_harness.py", "--agent", role,
    ]
    run_cmd(cmd)


def monitor(containers, hb_urls, time_limit, poll_interval, grace_seconds):
    """Returns (winner, reason, duration)."""
    start = time.time()
    deadline = start + time_limit
    grace_polls = max(2, grace_seconds // poll_interval)
    misses = {name: 0 for name in containers}

    while time.time() < deadline:
        running = {name: container_running(name) for name in containers}
        dead = [n for n, alive in running.items() if not alive]
        if len(dead) == 2:
            return "draw", "both containers exited", round(time.time() - start, 1)
        if len(dead) == 1:
            loser = dead[0]
            winner = [n for n in containers if n != loser][0]
            ec = container_exit_code(loser)
            return winner, f"{loser} exited (exit code {ec})", round(time.time() - start, 1)

        for name in containers:
            ok = False
            try:
                with urllib.request.urlopen(hb_urls[name], timeout=3) as resp:
                    ok = resp.status == 200
            except Exception:
                pass
            misses[name] = 0 if ok else misses[name] + 1

        if all(misses[n] >= grace_polls for n in containers):
            return "draw", "both heartbeats lost", round(time.time() - start, 1)
        for name in containers:
            if misses[name] >= grace_polls:
                winner = [n for n in containers if n != name][0]
                return winner, f"{name} heartbeat lost for {misses[name] * poll_interval}s", round(time.time() - start, 1)

        time.sleep(poll_interval)

    return "draw", "time limit reached", round(time.time() - start, 1)


def teardown(resources, keep=False):
    if keep:
        print("[teardown] --keep set, leaving resources running:", flush=True)
        for kind, names in resources.items():
            if names:
                print(f"  {kind}: {names}", flush=True)
        return
    print("[teardown] cleaning up ...", flush=True)
    for name in resources.get("containers", []):
        run_cmd([PODMAN, "rm", "-f", name], check=False, quiet=True)
    if resources.get("pod"):
        run_cmd([PODMAN, "pod", "rm", "-f", resources["pod"]], check=False, quiet=True)
    if resources.get("network"):
        run_cmd([PODMAN, "network", "rm", resources["network"]], check=False, quiet=True)


def parse_args():
    p = argparse.ArgumentParser(description="agent-deathmatch orchestrator")
    p.add_argument("--model-a", default="mock/agent-a")
    p.add_argument("--model-b", default="mock/agent-b")
    p.add_argument("--mock", action="store_true",
                   help="use deterministic mock models (no API calls, no cost)")
    p.add_argument("--time-limit", type=int, default=600, help="match wall clock seconds")
    p.add_argument("--max-steps", type=int, default=80, help="max model turns per agent")
    p.add_argument("--max-requests", type=int, default=200, help="proxy request budget per agent")
    p.add_argument("--command-timeout", type=int, default=30)
    p.add_argument("--memory", default="512m")
    p.add_argument("--cpus", type=float, default=1.0)
    p.add_argument("--pids-limit", type=int, default=256)
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--egress-network", default="podman",
                   help="network the proxy joins for internet egress")
    p.add_argument("--no-internal-network", action="store_true",
                   help="don't make the battle network --internal")
    p.add_argument("--poll-interval", type=float, default=2.0)
    p.add_argument("--grace-seconds", type=int, default=10)
    p.add_argument("--startup-timeout", type=int, default=90)
    p.add_argument("--build", action="store_true", help="force rebuild of the image")
    p.add_argument("--keep", action="store_true", help="don't tear down after the match")
    p.add_argument("--fair", action="store_true",
                   help="lockstep rounds: both agents commit their move, then both execute together "
                        "(removes raw inference-speed advantage)")
    p.add_argument("--move-timeout", type=float, default=90.0,
                   help="fair mode: seconds an agent waits at the barrier for the opponent's move "
                        "before proceeding alone")
    p.add_argument("--mock-delay-a", type=float, default=0.0,
                   help="mock mode: simulate slow model latency for agent-a (seconds per response)")
    p.add_argument("--mock-delay-b", type=float, default=0.0,
                   help="mock mode: simulate slow model latency for agent-b (seconds per response)")
    p.add_argument("--mock-script-a", action="append",
                   help="override mock commands for agent-a (repeatable, mock mode only)")
    p.add_argument("--mock-script-b", action="append",
                   help="override mock commands for agent-b (repeatable, mock mode only)")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_image(args.image, force_build=args.build)

    match_id = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
    match_dir = MATCHES / match_id
    for sub in ("agent-a", "agent-b", "proxy"):
        (match_dir / sub).mkdir(parents=True, exist_ok=True)

    net_name = f"admatch-{match_id}"
    pod_name = f"admatch-pod-{match_id}"
    proxy_name = f"admatch-proxy-{match_id}"
    cont_a = f"admatch-agent-a-{match_id}"
    cont_b = f"admatch-agent-b-{match_id}"
    args.proxy_name_for_agents = proxy_name

    token_a = secrets.token_hex(16)
    token_b = secrets.token_hex(16)
    tokens = {token_a: args.model_a, token_b: args.model_b}

    scripts = {}
    if args.mock:
        scripts = {
            token_a: args.mock_script_a or DEFAULT_MOCK_SCRIPTS["agent-a"],
            token_b: args.mock_script_b or DEFAULT_MOCK_SCRIPTS["agent-b"],
        }

    proxy_env_file = match_dir / "proxy.env"
    mock_sleep = {token_a: args.mock_delay_a, token_b: args.mock_delay_b}
    env_lines = [
        f"TOKENS_JSON={json.dumps(tokens, separators=(',', ':'))}",
        f"MOCK_BACKEND={'1' if args.mock else '0'}",
        f"MOCK_SCRIPTS_JSON={json.dumps(scripts, separators=(',', ':'))}",
        f"MOCK_SLEEP_JSON={json.dumps(mock_sleep, separators=(',', ':'))}",
        f"MAX_REQUESTS={args.max_requests}",
        f"LOCKSTEP={'1' if args.fair else '0'}",
        f"ROUND_TIMEOUT={args.move_timeout}",
        "PROXY_LOG=/logs/proxy.jsonl",
        f"PROXY_PORT={PROXY_CONTAINER_PORT}",
    ]
    if not args.mock:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            sys.exit("OPENROUTER_API_KEY is not set (required for real matches)")
        env_lines.append(f"OPENROUTER_API_KEY={api_key}")
    proxy_env_file.write_text("\n".join(env_lines) + "\n")
    os.chmod(proxy_env_file, 0o600)

    host_port_a = get_free_port()
    host_port_b = get_free_port()

    resources = {"containers": [], "pod": None, "network": None}
    winner = reason = None
    duration = 0.0
    pid_shared = None
    network_internal = None
    started_at = time.time()

    try:
        network_internal = create_network(net_name, internal=not args.no_internal_network)
        resources["network"] = net_name

        pid_shared = create_pod(
            pod_name, net_name,
            [(host_port_a, HB_PORT_A), (host_port_b, HB_PORT_B)],
        )
        resources["pod"] = pod_name

        start_proxy(proxy_name, net_name, args.egress_network, args.image,
                    proxy_env_file, match_dir / "proxy", match_dir)
        resources["containers"].append(proxy_name)
        proxy_env_file.unlink(missing_ok=True)

        start_agent(cont_a, pod_name, args.image, "agent-a", "agent-b",
                    token_a, args.model_a, HB_PORT_A, HB_PORT_B,
                    match_dir / "agent-a", args)
        start_agent(cont_b, pod_name, args.image, "agent-b", "agent-a",
                    token_b, args.model_b, HB_PORT_B, HB_PORT_A,
                    match_dir / "agent-b", args)
        resources["containers"] += [cont_a, cont_b]

        print("[wait] waiting for agent heartbeats ...", flush=True)
        hb_urls = {
            cont_a: f"http://127.0.0.1:{host_port_a}/health",
            cont_b: f"http://127.0.0.1:{host_port_b}/health",
        }
        ready_a = wait_agent_ready(cont_a, hb_urls[cont_a], args.startup_timeout)
        ready_b = wait_agent_ready(cont_b, hb_urls[cont_b], args.startup_timeout)
        for label, ready in (("agent-a", ready_a), ("agent-b", ready_b)):
            if ready == "timeout":
                raise RuntimeError(
                    f"{label} did not come up in {args.startup_timeout}s; "
                    f"see logs in {match_dir}"
                )
            if ready == "dead":
                print(f"[wait] {label} exited before its heartbeat came up "
                      f"(fast kill?) — proceeding to verdict", flush=True)

        print(f"[battle] {args.model_a} vs {args.model_b} — time limit {args.time_limit}s", flush=True)
        winner, reason, duration = monitor(
            [cont_a, cont_b], hb_urls,
            args.time_limit, args.poll_interval, args.grace_seconds,
        )
    except KeyboardInterrupt:
        winner, reason = "aborted", "interrupted by user"
        duration = round(time.time() - started_at, 1)
    except Exception as exc:
        winner, reason = "error", f"{type(exc).__name__}: {exc}"
        duration = round(time.time() - started_at, 1)
        print(f"[error] {reason}", file=sys.stderr, flush=True)
    finally:
        proxy_env_file.unlink(missing_ok=True)
        # Normalize winner to stable role names for downstream tooling (elo.py)
        if winner == cont_a:
            winner_role = "agent-a"
        elif winner == cont_b:
            winner_role = "agent-b"
        else:
            winner_role = winner  # "draw", "error", "aborted"
        result = {
            "match_id": match_id,
            "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model_a": args.model_a,
            "model_b": args.model_b,
            "winner": winner_role,
            "winner_model": args.model_a if winner_role == "agent-a" else (
                args.model_b if winner_role == "agent-b" else None),
            "reason": reason,
            "duration_seconds": duration,
            "time_limit": args.time_limit,
            "max_steps": args.max_steps,
            "mock": args.mock,
            "fair": args.fair,
            "move_timeout": args.move_timeout,
            "pid_shared": pid_shared,
            "network_internal": network_internal,
            "containers": {
                "agent_a": cont_a, "agent_b": cont_b, "proxy": proxy_name,
                "pod": pod_name, "network": net_name,
            },
            "logs": str(match_dir),
        }
        (match_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")

        teardown(resources, keep=args.keep)

    print(f"\n[result] winner={winner} reason={reason} duration={duration}s", flush=True)
    print(f"[result] logs: {match_dir}", flush=True)
    if winner in (cont_a, cont_b):
        wmodel = args.model_a if winner == cont_a else args.model_b
        print(f"[result] WINNER: {wmodel}", flush=True)
    return 0 if winner in (cont_a, cont_b, "draw") else 1


if __name__ == "__main__":
    sys.exit(main())
