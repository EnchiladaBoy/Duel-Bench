#!/usr/bin/env python3
"""Model proxy for agent-deathmatch.

Runs in its own container on the battle network. Agent harnesses POST
OpenAI-format chat completions with a per-agent bearer token; the proxy:
- maps token -> configured model (agents can never pick a different model);
- injects the host OPENROUTER_API_KEY (battle containers never see it);
- rebuilds the upstream payload from an allowlist so an agent cannot smuggle
  sampling or routing parameters and give itself an advantage;
- enforces a per-agent request AND token budget;
- hosts the lockstep barrier for --fair mode;
- in MOCK_BACKEND=1 mode returns deterministic scripted tool calls.

Nothing served by this process ever echoes a bearer token: agents are
identified to each other, and in the logs, by role name only.
"""

import http.client
import json
import os
import socket
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import modes

PORT = int(os.environ.get("PROXY_PORT", "8080"))
TOKENS = json.loads(os.environ.get("TOKENS_JSON", "{}"))
ROLES = json.loads(os.environ.get("ROLES_JSON", "{}"))
OPENROUTER_URL = os.environ.get(
    "OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions"
)
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
# Held only by the orchestrator and the proxy. It is never placed in an agent
# container's environment or mounted into one, so it is not reachable through
# the shared PID namespace via /proc/<pid>/environ.
CONTROL_TOKEN = os.environ.get("CONTROL_TOKEN", "")
MOCK = os.environ.get("MOCK_BACKEND", "0") == "1"
MOCK_SCRIPTS = json.loads(os.environ.get("MOCK_SCRIPTS_JSON", "{}"))
MOCK_SLEEP = json.loads(os.environ.get("MOCK_SLEEP_JSON", "{}"))
MODE = modes.from_env(os.environ)
MAX_REQUESTS = MODE.max_requests
MAX_TOKENS_BUDGET = int(os.environ.get("MAX_TOKENS_BUDGET", "0"))  # 0 = unlimited
MAX_TOKENS_PER_CALL = int(os.environ.get("MAX_TOKENS_PER_CALL", "4096"))
ARENA_TEMPERATURE = os.environ.get("ARENA_TEMPERATURE", "")
ARENA_SEED = os.environ.get("ARENA_SEED", "")
LOG_PATH = os.environ.get("PROXY_LOG", "/logs/proxy.jsonl")
LOCKSTEP = MODE.lockstep
ROUND_TIMEOUT = MODE.move_deadline or 90.0
MAX_MISSED_ROUNDS = MODE.max_missed_rounds or 2
MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", str(4 * 1024 * 1024)))
# The harness must wait LONGER than this (see agent_harness.CLIENT_TIMEOUT), or
# it abandons a response the proxy is still fetching, retries, and spends a
# second upstream call - which with a time bank double-charges one move.
UPSTREAM_TIMEOUT = float(os.environ.get("UPSTREAM_TIMEOUT", "180"))
BANK_GRACE = 5.0
THINKING_TICK = float(os.environ.get("THINKING_TICK", "2"))
STARTING_GUN = os.environ.get("STARTING_GUN", "1") == "1"
STARTING_GUN_TIMEOUT = float(os.environ.get("STARTING_GUN_TIMEOUT", "60"))
KEEP_RELEASES = 200

LOCK = threading.Lock()
REQUEST_COUNT = {token: 0 for token in TOKENS}
TOKEN_USAGE = {token: 0 for token in TOKENS}
# Seconds of inference left to each agent. Charged for time a completion is
# actually in flight; waiting at the barrier is free, so a fast model is never
# punished for having a slow opponent.
BANK_REMAINING = {token: MODE.time_bank for token in TOKENS}
MOVE_SECONDS = {token: [] for token in TOKENS}
# How long the move an agent is about to commit actually took it to produce.
# Measuring the agent's OWN latency is what makes a deadline about your speed
# rather than about who happened to think second.
LAST_MOVE_SECONDS = {token: None for token in TOKENS}
FORFEITS = {token: 0 for token in TOKENS}
# Mutated and read under LOCK only, so the exhaustion test is atomic with the
# charge. Barrier retirement happens separately, outside LOCK.
EXHAUSTED = set()
# An agent has at most one move in flight. Without this, K concurrent requests
# on one token each pass the bank check before any of them charges, delivering
# K x the bank's worth of inference for the price of one. Maps token -> the
# monotonic time the move started, so the ticker can report live progress.
IN_FLIGHT = {}
MOCK_INDEX = {token: 0 for token in TOKENS}
_LOG_FH = None

# Only these keys are forwarded upstream. Everything else an agent might send
# (n, temperature, reasoning, provider, transforms, logit_bias, ...) is dropped
# so both agents are always sampled under identical arena settings.
ALLOWED_BODY_KEYS = ("messages", "tools", "tool_choice")


def resolvers():
    """Nameservers actually in effect inside this container."""
    found = []
    try:
        with open("/etc/resolv.conf", "r") as fh:
            for line in fh:
                if line.startswith("nameserver"):
                    parts = line.split()
                    if len(parts) > 1:
                        found.append(parts[1])
    except OSError:
        pass
    return found


def egress_report(timeout=8.0):
    """Prove - from inside the proxy container, in its real network topology -
    that openrouter.ai is reachable, and name the stage that fails if not.

    The battle network is --internal and the proxy is hot-attached to an egress
    network afterwards; a host-side curl proves nothing about either. The one
    historical real match died at exactly the first stage here and its logs
    recorded only a repeated 502.
    """
    parsed = urllib.parse.urlparse(OPENROUTER_URL)
    host, port = parsed.hostname, parsed.port or 443
    report = {"host": host, "resolvers": resolvers()}

    started = time.monotonic()
    try:
        addresses = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)
    except Exception as exc:
        report.update(ok=False, stage="dns", errno=getattr(exc, "errno", None),
                      error=_redact(f"{type(exc).__name__}: {exc}"))
        return report
    report["dns_ms"] = round((time.monotonic() - started) * 1000, 1)

    started = time.monotonic()
    sock = None
    try:
        sock = socket.create_connection(addresses[0][4], timeout=timeout)
    except Exception as exc:
        report.update(ok=False, stage="tcp", errno=getattr(exc, "errno", None),
                      error=_redact(f"{type(exc).__name__}: {exc}"))
        return report
    report["tcp_ms"] = round((time.monotonic() - started) * 1000, 1)

    started = time.monotonic()
    try:
        wrapped = ssl.create_default_context().wrap_socket(sock, server_hostname=host)
    except Exception as exc:
        sock.close()
        report.update(ok=False, stage="tls",
                      error=_redact(f"{type(exc).__name__}: {exc}"))
        return report
    report["tls_ms"] = round((time.monotonic() - started) * 1000, 1)

    # GET /api/v1/key: validates the key and costs zero tokens.
    started = time.monotonic()
    try:
        conn = http.client.HTTPSConnection(host, port, timeout=timeout)
        conn.sock = wrapped
        conn.request("GET", "/api/v1/key", headers={
            "Authorization": f"Bearer {API_KEY}", "Accept": "application/json"})
        answer = conn.getresponse()
        report["key_check_status"] = answer.status
        answer.read()
        conn.close()
    except Exception as exc:
        report.update(ok=False, stage="http",
                      error=_redact(f"{type(exc).__name__}: {exc}"))
        return report
    report["key_ms"] = round((time.monotonic() - started) * 1000, 1)
    report["ok"] = report["key_check_status"] == 200
    if not report["ok"]:
        # 401 bad key, 402 out of credit - both are far cheaper to learn here.
        report["stage"] = "auth"
    return report


def _redact(text):
    """Strip every secret this process holds out of a string.

    The precondition for logging upstream error bodies at all: the old proxy
    leaked raw agent tokens into proxy.jsonl, and diagnostics must not
    reintroduce that."""
    if not isinstance(text, str):
        text = str(text)
    for secret in [API_KEY, CONTROL_TOKEN, *TOKENS]:
        if secret and len(secret) >= 8:
            text = text.replace(secret, "<redacted>")
    return text


def role_of(token):
    return ROLES.get(token, "unknown")


def log(record):
    global _LOG_FH
    record = {"ts": time.time(), **record}
    line = json.dumps(record, ensure_ascii=False)
    try:
        with LOCK:
            if _LOG_FH is None:
                log_dir = os.path.dirname(LOG_PATH)
                if log_dir:
                    os.makedirs(log_dir, exist_ok=True)
                _LOG_FH = open(LOG_PATH, "a", encoding="utf-8")
            _LOG_FH.write(line + "\n")
            _LOG_FH.flush()
    except Exception:
        pass
    print(line, flush=True)


# --------------------------------------------------------------- lockstep

BARRIER_COND = threading.Condition()


def reset_move_state():
    """Test helper: clear per-move state without touching the barrier."""
    for token in TOKENS:
        LAST_MOVE_SECONDS[token] = None
        FORFEITS[token] = 0


def initial_barrier_state():
    """Single source of truth for barrier state, so tests resetting it cannot
    drift out of sync as new keys are added."""
    return {
        "round": 1,
        "joined": set(),        # tokens that committed a move this round
        "deadline": None,       # absolute time this round auto-releases
        "releases": {},         # round -> {"both": bool, "joined": [roles]}
        "missed": {token: 0 for token in TOKENS},
        "active": set(TOKENS),  # tokens still expected to join
    }


barrier = initial_barrier_state()


def _prune_releases_unlocked():
    if len(barrier["releases"]) > KEEP_RELEASES:
        for stale in sorted(barrier["releases"])[:-KEEP_RELEASES]:
            del barrier["releases"][stale]


def _release_unlocked(complete):
    """Close the current round. Agents that did not join have their miss count
    incremented, and an agent that misses MAX_MISSED_ROUNDS consecutive rounds
    drops out of the quorum so a survivor does not pay ROUND_TIMEOUT forever."""
    joined = set(barrier["joined"])
    record = {
        # True only if every agent still in the game committed a move this
        # round - not merely that a shrunken quorum was satisfied.
        "both": complete and len(joined) >= len(TOKENS),
        "joined": sorted(role_of(t) for t in joined),
    }
    barrier["releases"][barrier["round"]] = record

    for token in list(barrier["active"]):
        if token in joined:
            barrier["missed"][token] = 0
        else:
            barrier["missed"][token] = barrier["missed"].get(token, 0) + 1
            if barrier["missed"][token] >= MAX_MISSED_ROUNDS:
                barrier["active"].discard(token)

    barrier["round"] += 1
    barrier["joined"] = set()
    barrier["deadline"] = None
    _prune_releases_unlocked()
    BARRIER_COND.notify_all()
    return record


def barrier_join(token):
    """Commit this agent's move for the current round."""
    with BARRIER_COND:
        round_no = barrier["round"]
        # A move that took longer than the deadline forfeits this round: the
        # agent still JOINS, so the round releases normally and its opponent is
        # not stalled, but the move does not execute.
        forfeit = False
        if MODE.deadline_effect == "forfeit" and MODE.move_deadline:
            last = LAST_MOVE_SECONDS.get(token)
            if last is not None and last > MODE.move_deadline:
                forfeit = True
                with LOCK:
                    FORFEITS[token] += 1
        barrier["joined"].add(token)
        # A merely-absent agent rejoins the quorum; an agent whose time bank is
        # spent does not, or it would stall the survivor every round forever.
        if token not in EXHAUSTED:
            barrier["active"].add(token)
        barrier["missed"][token] = 0
        if barrier["deadline"] is None:
            barrier["deadline"] = time.time() + ROUND_TIMEOUT
        # An empty active set means every agent has retired (all banks spent).
        # Short-circuiting to False there would leave a round that BOTH agents
        # joined permanently unreleased, stalling them for a full deadline.
        if forfeit:
            log({"event": "move_forfeit", "agent": role_of(token), "round": round_no,
                 "took": round(LAST_MOVE_SECONDS[token], 2),
                 "deadline": MODE.move_deadline})
        if barrier["joined"] >= barrier["active"]:
            record = _release_unlocked(complete=bool(barrier["active"]))
            return {"round": round_no, "released": True, "forfeit": forfeit, **record}
        return {
            "round": round_no,
            "released": False,
            "forfeit": forfeit,
            "joined": sorted(role_of(t) for t in barrier["joined"]),
        }


# ------------------------------------------------------------ starting gun

START_COND = threading.Condition()
START_ARRIVED = set()
START_STATE = {"released": not STARTING_GUN}


def starting_gun(token):
    """Hold each agent's FIRST model call until both have arrived, then release
    them together.

    Agent containers are created sequentially, so one is ready fractionally
    earlier. In realtime that head start is a real edge, and to a spectator a
    fight decided by container-start jitter looks rigged. Called before
    admission and before the clock starts, so waiting here costs nothing and is
    never charged to a time bank."""
    with START_COND:
        if START_STATE["released"]:
            return
        START_ARRIVED.add(token)
        if len(START_ARRIVED) >= len(TOKENS):
            START_STATE["released"] = True
            START_COND.notify_all()
            log({"event": "go", "agents": sorted(role_of(t) for t in START_ARRIVED)})
            return
        deadline = time.monotonic() + STARTING_GUN_TIMEOUT
        while not START_STATE["released"]:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                # An opponent that never turns up must not hold the match.
                START_STATE["released"] = True
                START_COND.notify_all()
                log({"event": "go", "reason": "timeout",
                     "agents": sorted(role_of(t) for t in START_ARRIVED)})
                break
            START_COND.wait(min(remaining, 1.0))


def thinking_ticker():
    """Emit progress while a completion is in flight.

    Without this a lockstep round is a long silence followed by two commands;
    with it a viewer sees two clocks racing. Costs one record per agent per
    tick and never touches the request path."""
    while True:
        time.sleep(THINKING_TICK)
        with LOCK:
            snapshot = [(t, started) for t, started in IN_FLIGHT.items()]
        now = time.monotonic()
        for token, started in snapshot:
            record = {"event": "thinking", "agent": role_of(token),
                      "elapsed": round(now - started, 1)}
            if MODE.time_bank is not None and BANK_REMAINING.get(token) is not None:
                # What the clock UI needs between move_start and move_end.
                record["bank_remaining"] = round(
                    max(0.0, BANK_REMAINING[token] - (now - started)), 1)
            log(record)


def retire_from_barrier(token):
    """Retire an agent from the barrier the instant its bank empties.

    Without the immediate release the survivor blocks in barrier_wait until the
    round deadline expires, then pays it again until max_missed_rounds ejects
    the dead token - minutes of dead time at the most dramatic moment of the
    match."""
    with BARRIER_COND:
        barrier["active"].discard(token)
        if barrier["joined"] >= barrier["active"]:
            _release_unlocked(complete=bool(barrier["active"]))
        BARRIER_COND.notify_all()


def barrier_wait(round_no, token):
    """Block until the given round is released. Only a round the caller has
    actually joined can be waited on, and the wait path never creates release
    records - otherwise an agent could pre-release every future round with a
    handful of requests and turn --fair back into a free-for-all."""
    with BARRIER_COND:
        if round_no > barrier["round"]:
            return {"error": "round is in the future", "round": round_no}, 400
        if round_no == barrier["round"] and token not in barrier["joined"]:
            return {"error": "join this round before waiting on it", "round": round_no}, 400

        while round_no not in barrier["releases"]:
            if barrier["round"] != round_no:
                # Round advanced but its record was pruned; synthesize a reply
                # without storing it.
                return {"round": round_no, "both": False, "joined": []}, 200
            deadline = barrier["deadline"]
            if deadline is None:
                return {"round": round_no, "both": False, "joined": []}, 200
            remaining = deadline - time.time()
            if remaining <= 0:
                _release_unlocked(complete=False)
                break
            BARRIER_COND.wait(min(remaining, 5.0))
        return {"round": round_no, **barrier["releases"][round_no]}, 200


# --------------------------------------------------------------- backends

def mock_delay_for(token, index):
    """MOCK_SLEEP may be a scalar or a per-move list (cycled), so a time bank
    can be exercised against varying inference latency at zero API cost."""
    spec = MOCK_SLEEP.get(token, 0)
    if isinstance(spec, list):
        return float(spec[index % len(spec)]) if spec else 0.0
    return float(spec or 0)


def mock_completion(token):
    with LOCK:
        delay_index = MOCK_INDEX.get(token, 0)
    delay = mock_delay_for(token, delay_index)
    if delay > 0:
        time.sleep(delay)
    with LOCK:
        idx = MOCK_INDEX.get(token, 0)
        script = MOCK_SCRIPTS.get(token, [])
        if idx < len(script):
            command = script[idx]
            MOCK_INDEX[token] = idx + 1
            return {
                "id": f"mock-{role_of(token)}-{idx}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "mock",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": f"mock_call_{role_of(token)}_{idx}",
                            "type": "function",
                            "function": {
                                "name": "run_bash",
                                "arguments": json.dumps({"command": command}),
                            },
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        return {
            "id": f"mock-{role_of(token)}-done",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mock",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "No further actions."},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }


def build_upstream_payload(body, model):
    """Rebuild the request from an allowlist so the arena, not the agent,
    decides how the model is sampled."""
    payload = {"model": model, "n": 1}
    for key in ALLOWED_BODY_KEYS:
        if key in body:
            payload[key] = body[key]
    requested = body.get("max_tokens")
    try:
        requested = int(requested)
    except (TypeError, ValueError):
        requested = MAX_TOKENS_PER_CALL
    payload["max_tokens"] = max(1, min(requested, MAX_TOKENS_PER_CALL))
    if ARENA_TEMPERATURE:
        payload["temperature"] = float(ARENA_TEMPERATURE)
    if ARENA_SEED:
        payload["seed"] = int(ARENA_SEED)
    return payload


def forward_openrouter(body, model, timeout=None):
    payload = build_upstream_payload(body, model)
    request = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
            "HTTP-Referer": "https://localhost/agent-deathmatch",
            "X-Title": "agent-deathmatch",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout or UPSTREAM_TIMEOUT) as response:
            body = json.loads(response.read().decode("utf-8"))
            choices = body.get("choices") if isinstance(body, dict) else None
            if not (isinstance(choices, list) and choices):
                # OpenRouter answers 200 with a body-level error when a provider
                # fails AFTER accepting the request. Trusting the status charges
                # the time bank and turns a provider outage into a rated
                # protocol forfeit - the model blamed for the network.
                err = (body.get("error") or {}) if isinstance(body, dict) else {}
                return 503, {
                    "error": _redact(err.get("message") or "upstream returned no choices")[:500],
                    "error_kind": "upstream",
                    "upstream_status": err.get("code"),
                    "provider": body.get("provider") if isinstance(body, dict) else None,
                    "generation_id": body.get("id") if isinstance(body, dict) else None,
                    "stage": "decode",
                }
            return response.status, body
    except urllib.error.HTTPError as exc:
        raw = _redact(exc.read().decode(errors="replace"))
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"error": raw[:2000]}
        if not isinstance(parsed, dict):
            parsed = {"error": str(parsed)[:2000]}
        # Upstream throttling must NOT reach the harness as a 429: the harness
        # treats 429 as its own budget being spent and stops for good.
        status = 503 if exc.code in (429, 500, 502, 503, 504) else exc.code
        parsed["error_kind"] = "upstream"
        parsed["upstream_status"] = exc.code
        parsed["stage"] = "http"
        return status, parsed
    except Exception as exc:
        # Distinguish "DNS is broken" from "TLS failed" from "the socket timed
        # out". The one historical real failure was a getaddrinfo error and the
        # logs recorded nothing but a repeated 502.
        name = type(exc).__name__
        stage = {"gaierror": "dns", "SSLError": "tls", "SSLCertVerificationError": "tls",
                 "timeout": "timeout", "ConnectionRefusedError": "tcp"}.get(name)
        if stage is None:
            stage = "dns" if "Name or service not known" in str(exc) else "connect"
        return 503, {
            "error": _redact(f"{name}: {exc}")[:500],
            "error_kind": "upstream",
            "stage": stage,
            "errno": getattr(exc, "errno", None),
        }


def send_json(handler, status, obj):
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


class ProxyHandler(BaseHTTPRequestHandler):
    server_version = "AgentDeathmatchProxy/2.0"
    timeout = 30  # per-connection socket timeout; a trickling client cannot pin a thread

    def _bearer(self):
        auth = self.headers.get("Authorization", "")
        return auth[7:].strip() if auth.lower().startswith("bearer ") else ""

    def _auth_token(self):
        token = self._bearer()
        return token if token in TOKENS else None

    def _is_control(self):
        return bool(CONTROL_TOKEN) and self._bearer() == CONTROL_TOKEN

    def _read_body(self):
        """Returns (body_dict, error_response_or_None)."""
        raw_len = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_len)
        except (TypeError, ValueError):
            return None, (400, {"error": "invalid Content-Length"})
        if length < 0 or length > MAX_BODY_BYTES:
            return None, (413, {"error": f"body exceeds {MAX_BODY_BYTES} bytes"})
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None, (400, {"error": "invalid JSON body"})
        if not isinstance(body, dict):
            return None, (400, {"error": "body must be a JSON object"})
        if not isinstance(body.get("messages"), list):
            return None, (400, {"error": "body.messages must be a list"})
        return body, None

    def do_GET(self):
        try:
            self._do_GET()
        except Exception as exc:  # never leave a client hanging
            try:
                send_json(self, 500, {"error": f"{type(exc).__name__}: {exc}"})
            except Exception:
                pass

    def _do_GET(self):
        if self.path == "/health":
            # Deliberately minimal: this endpoint is reachable by both agents,
            # so it must not disclose tokens, per-agent counts, or any other
            # signal about the opponent.
            send_json(self, 200, {"status": "ok", "mock": MOCK,
                                  "agents": len(TOKENS), "lockstep": LOCKSTEP})
            return

        if self.path == "/control/egress":
            # Control-gated: an agent must never be able to ask whether the
            # arena has egress, nor use this to probe arbitrary hosts.
            if not self._is_control():
                send_json(self, 401, {"error": "control token required"})
                return
            report = egress_report()
            log({"event": "egress_check", **report})
            send_json(self, 200 if report.get("ok") else 503, report)
            return

        if self.path == "/control/status":
            # Match-control state for the orchestrator. Never reachable by an
            # agent: it would disclose the opponent's progress, and later its
            # remaining time bank.
            if not self._is_control():
                send_json(self, 401, {"error": "control token required"})
                return
            with LOCK:
                requests_used = dict(REQUEST_COUNT)
                tokens_used = dict(TOKEN_USAGE)
            with BARRIER_COND:
                round_no = barrier["round"]
                active = sorted(role_of(t) for t in barrier["active"])
            with LOCK:
                # Snapshot under the lock that guards them: iterating EXHAUSTED
                # while a completion thread adds to it raises "Set changed size
                # during iteration" and 500s the orchestrator's poll.
                exhausted = sorted(role_of(t) for t in EXHAUSTED)
                banks = {role_of(t): (round(v, 2) if v is not None else None)
                         for t, v in BANK_REMAINING.items()}
                moves = {role_of(t): len(v) for t, v in MOVE_SECONDS.items()}
                all_spent = len(EXHAUSTED) >= len(TOKENS)
            terminal = None
            if MODE.termination == "banks" and all_spent:
                terminal = "banks_exhausted"
            elif MODE.max_rounds and round_no > MODE.max_rounds:
                terminal = "rounds_complete"
            send_json(self, 200, {
                "mode": MODE.name,
                "round": round_no,
                "active": active,
                "exhausted": exhausted,
                "banks": banks,
                "moves": moves,
                "requests": {role_of(t): n for t, n in requests_used.items()},
                "forfeits": {role_of(t): n for t, n in FORFEITS.items()},
                "tokens": {role_of(t): n for t, n in tokens_used.items()},
                "terminal": terminal,
            })
            return

        if self.path.startswith("/barrier/wait"):
            if not LOCKSTEP:
                send_json(self, 400, {"error": "lockstep disabled"})
                return
            token = self._auth_token()
            if not token:
                send_json(self, 401, {"error": "unknown agent token"})
                return
            query = self.path.split("?", 1)[1] if "?" in self.path else ""
            params = urllib.parse.parse_qs(query)
            try:
                round_no = int(params.get("round", ["0"])[0])
            except ValueError:
                send_json(self, 400, {"error": "invalid round parameter"})
                return
            result, status = barrier_wait(round_no, token)
            if status == 200:
                log({"event": "barrier_release", "agent": role_of(token), **result})
            send_json(self, status, result)
            return

        send_json(self, 404, {"error": "not found"})

    def do_POST(self):
        try:
            self._do_POST()
        except Exception as exc:
            try:
                send_json(self, 500, {"error": f"{type(exc).__name__}: {exc}"})
            except Exception:
                pass

    def _do_POST(self):
        if self.path == "/barrier/join":
            if not LOCKSTEP:
                send_json(self, 400, {"error": "lockstep disabled"})
                return
            token = self._auth_token()
            if not token:
                send_json(self, 401, {"error": "unknown agent token"})
                return
            send_json(self, 200, barrier_join(token))
            return

        if self.path != "/v1/chat/completions":
            send_json(self, 404, {"error": "not found"})
            return

        token = self._auth_token()
        if not token:
            log({"event": "auth_rejected"})
            send_json(self, 401, {"error": "unknown agent token"})
            return

        body, err = self._read_body()
        if err:
            # Validation happens before the budget is touched, so a malformed
            # request can never cost an agent a turn or trigger a paid call.
            send_json(self, err[0], err[1])
            return

        # Both agents are released together on their first move (below), so no
        # state is held while waiting and the wait is never charged.
        starting_gun(token)

        # The proxy ENFORCES the round cap. Leaving it to the orchestrator's
        # poll would hand whichever agent asked first a bonus move, and "equal
        # turns each" would quietly stop being true.
        # Enforced wherever a cap exists, not only where rounds are the
        # TERMINATION rule - otherwise --max-rounds is accepted in time-bank and
        # silently enforces nothing, which is a safety cap that does not exist.
        if MODE.max_rounds:
            with BARRIER_COND:
                past_cap = barrier["round"] > MODE.max_rounds
            if past_cap:
                send_json(self, 429, {
                    "error": f"all {MODE.max_rounds} rounds played",
                    "error_kind": "rounds_complete",
                })
                return

        # Admission is decided in ONE critical section: bank exhaustion, the
        # request budget, and the one-move-in-flight rule. Splitting these let
        # concurrent requests on a single token each pass a nearly-empty bank.
        with LOCK:
            if MODE.time_bank is not None and token in EXHAUSTED:
                refusal, remaining = "time_bank_exhausted", None
            elif token in IN_FLIGHT:
                refusal, remaining = "concurrent_request", None
            elif REQUEST_COUNT[token] >= MAX_REQUESTS:
                # With max_requests > max_steps * MAX_MODEL_RETRIES enforced in
                # the mode table, a correct harness CANNOT reach this - so it is
                # evidence of an out-of-band caller rather than a heuristic.
                refusal, remaining = "request_budget", None
            elif MAX_TOKENS_BUDGET and TOKEN_USAGE[token] >= MAX_TOKENS_BUDGET:
                # A spend allowance applied equally to both agents, reachable by
                # a perfectly-behaved verbose model. A game resource, like a
                # time bank - not a fault.
                refusal, remaining = "token_budget", None
            else:
                REQUEST_COUNT[token] += 1
                IN_FLIGHT[token] = time.monotonic()
                refusal, remaining = None, BANK_REMAINING.get(token)

        if refusal == "time_bank_exhausted":
            # A GAME outcome, so it must not reuse "proxy_budget", which the
            # harness treats as terminal infrastructure failure - that would
            # mark every time-bank match unrated, the inverse of the intent.
            send_json(self, 429, {"error": "time bank exhausted",
                                  "error_kind": "time_bank_exhausted"})
            return
        if refusal == "concurrent_request":
            send_json(self, 429, {"error": "one move in flight at a time",
                                  "error_kind": "concurrent_request"})
            return
        if refusal in ("request_budget", "token_budget"):
            log({"event": "budget_exhausted", "kind": refusal, "agent": role_of(token),
                 "requests": REQUEST_COUNT[token], "tokens": TOKEN_USAGE[token]})
            send_json(self, 429, {"error": f"{refusal.replace('_', ' ')} exhausted",
                                  "error_kind": refusal})
            return

        # Clamp the call so an agent with two seconds left cannot burn a full
        # upstream timeout. Without this the bank is not a bank.
        call_timeout = None
        if MODE.time_bank is not None and remaining is not None:
            call_timeout = max(1.0, min(UPSTREAM_TIMEOUT, remaining + BANK_GRACE))

        log({"event": "move_start", "agent": role_of(token),
             "round": barrier["round"] if LOCKSTEP else None,
             "bank_remaining": (round(remaining, 2) if remaining is not None else None)})

        # Timed here - after validation and admission, around the dispatch only -
        # and with a monotonic clock, so a wall-clock step can neither refund nor
        # steal a bank. Measured proxy-side because the harness runs in a
        # container the agent has a shell in.
        started = time.monotonic()
        try:
            if MOCK:
                status, response = 200, mock_completion(token)
            else:
                if not API_KEY:
                    with LOCK:
                        IN_FLIGHT.pop(token, None)
                    send_json(self, 500, {"error": "proxy has no OPENROUTER_API_KEY"})
                    return
                status, response = forward_openrouter(body, TOKENS[token], timeout=call_timeout)
        except Exception:
            with LOCK:
                IN_FLIGHT.pop(token, None)
            raise
        elapsed = time.monotonic() - started

        # A call cut short by the bank clamp comes back as an upstream failure,
        # but the time was really spent. Charging only successes would leave the
        # bank stuck just above zero forever: the agent would never exhaust,
        # `terminal` would never fire, and the match would run to the guard.
        bank_overrun = (call_timeout is not None and status != 200
                        and elapsed >= call_timeout - 1.0)

        usage = response.get("usage") if isinstance(response, dict) else None
        newly_exhausted = False
        with LOCK:
            IN_FLIGHT.pop(token, None)
            if isinstance(usage, dict):
                try:
                    TOKEN_USAGE[token] += int(usage.get("total_tokens") or 0)
                except (TypeError, ValueError):
                    # A provider returning a non-numeric token count must not
                    # blow up the handler and hand the agent a free move.
                    pass
            if status == 200 or bank_overrun:
                # Charge for a usable answer, or for time the clamp cut short.
                # An ordinary upstream failure is not the model's fault.
                MOVE_SECONDS[token].append(round(elapsed, 3))
                LAST_MOVE_SECONDS[token] = elapsed
                if MODE.time_bank is not None and BANK_REMAINING[token] is not None:
                    BANK_REMAINING[token] = max(0.0, BANK_REMAINING[token] - elapsed)
                    if BANK_REMAINING[token] <= 0 and token not in EXHAUSTED:
                        EXHAUSTED.add(token)
                        newly_exhausted = True
        if newly_exhausted:
            # Outside LOCK: retire_from_barrier takes BARRIER_COND.
            retire_from_barrier(token)
            log({"event": "bank_exhausted", "agent": role_of(token),
                 "moves": len(MOVE_SECONDS[token]),
                 "cause": "bank_clamp" if bank_overrun else "spent"})

        if bank_overrun:
            log({"event": "completion", "agent": role_of(token), "model": TOKENS[token],
                 "mock": MOCK, "status": 429, "elapsed_seconds": round(elapsed, 3),
                 "usage": None, "bank_remaining": 0.0,
                 "requests_used": REQUEST_COUNT[token], "messages_in": len(body.get("messages") or [])})
            send_json(self, 429, {"error": "time bank exhausted mid-call",
                                  "error_kind": "time_bank_exhausted"})
            return

        if isinstance(response, dict):
            response["arena"] = self._arena_block(token, elapsed)

        log({
            "event": "completion",
            "agent": role_of(token),
            "model": TOKENS[token],
            "mock": MOCK,
            "status": status,
            "elapsed_seconds": round(elapsed, 3),
            "usage": usage,
            "cumulative_tokens": TOKEN_USAGE[token],
            "requests_used": REQUEST_COUNT[token],
            "bank_remaining": (round(BANK_REMAINING[token], 2)
                               if BANK_REMAINING.get(token) is not None else None),
            "messages_in": len(body.get("messages") or []),
        })
        send_json(self, status, response)

    def _arena_block(self, token, elapsed):
        """State the model needs to play this mode, returned in the response
        body (headers are dropped by the harness's call_model). Keyed by role,
        never by token."""
        arena = {"mode": MODE.name, "move_seconds": round(elapsed, 2)}
        if LOCKSTEP:
            with BARRIER_COND:
                arena["round"] = barrier["round"]
        if MODE.time_bank is None:
            return arena
        arena["bank_remaining"] = round(BANK_REMAINING[token] or 0.0, 1)
        arena["exhausted"] = token in EXHAUSTED
        if MODE.reveal_opponent_bank:
            # Disclosed deliberately, rounded to whole seconds. In lockstep the
            # time an agent spends blocked at the barrier already IS its
            # opponent's think time, so hiding it would only reward whichever
            # model thought to measure that - and chess clocks are public.
            for other, left in BANK_REMAINING.items():
                if other != token:
                    arena["opponent"] = role_of(other)
                    # round, not floor: int() reported an opponent with 0.4s
                    # left as "~0s", i.e. already beaten, at the exact
                    # moment that claim is most consequential.
                    arena["opponent_bank_remaining"] = max(0, round(left or 0))
        return arena

    def log_message(self, fmt, *args):
        pass


def main():
    if not TOKENS:
        raise SystemExit("TOKENS_JSON is empty - no agents configured")
    server = ThreadingHTTPServer(("0.0.0.0", PORT), ProxyHandler)
    server.daemon_threads = True
    threading.Thread(target=thinking_ticker, daemon=True).start()
    log({
        "event": "proxy_start",
        "port": PORT,
        "mock": MOCK,
        "mode": MODE.name,
        "models": sorted(TOKENS.values()),
        "max_requests": MAX_REQUESTS,
        "max_tokens_budget": MAX_TOKENS_BUDGET or None,
        "lockstep": LOCKSTEP,
    })
    server.serve_forever()


if __name__ == "__main__":
    main()
