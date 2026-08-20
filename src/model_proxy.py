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

import json
import os
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
KEEP_RELEASES = 200

LOCK = threading.Lock()
REQUEST_COUNT = {token: 0 for token in TOKENS}
TOKEN_USAGE = {token: 0 for token in TOKENS}
MOCK_INDEX = {token: 0 for token in TOKENS}
_LOG_FH = None

# Only these keys are forwarded upstream. Everything else an agent might send
# (n, temperature, reasoning, provider, transforms, logit_bias, ...) is dropped
# so both agents are always sampled under identical arena settings.
ALLOWED_BODY_KEYS = ("messages", "tools", "tool_choice")


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
        "both": complete,
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
        barrier["joined"].add(token)
        barrier["active"].add(token)      # a returning agent rejoins the quorum
        barrier["missed"][token] = 0
        if barrier["deadline"] is None:
            barrier["deadline"] = time.time() + ROUND_TIMEOUT
        if barrier["active"] and barrier["joined"] >= barrier["active"]:
            record = _release_unlocked(complete=True)
            return {"round": round_no, "released": True, **record}
        return {
            "round": round_no,
            "released": False,
            "joined": sorted(role_of(t) for t in barrier["joined"]),
        }


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

def mock_completion(token):
    delay = float(MOCK_SLEEP.get(token, 0) or 0)
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


def forward_openrouter(body, model):
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
        with urllib.request.urlopen(request, timeout=180) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
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
        return status, parsed
    except Exception as exc:
        return 503, {
            "error": f"{type(exc).__name__}: {exc}",
            "error_kind": "upstream",
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
            send_json(self, 200, {
                "mode": MODE.name,
                "round": round_no,
                "active": active,
                "requests": {role_of(t): n for t, n in requests_used.items()},
                "tokens": {role_of(t): n for t, n in tokens_used.items()},
                # Populated once round/bank termination lands; the orchestrator
                # already treats a null terminal as "keep playing".
                "terminal": None,
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

        with LOCK:
            over_requests = REQUEST_COUNT[token] >= MAX_REQUESTS
            over_tokens = (
                MAX_TOKENS_BUDGET and TOKEN_USAGE[token] >= MAX_TOKENS_BUDGET
            )
            if not (over_requests or over_tokens):
                REQUEST_COUNT[token] += 1
        if over_requests or over_tokens:
            log({"event": "budget_exhausted", "agent": role_of(token),
                 "requests": REQUEST_COUNT[token], "tokens": TOKEN_USAGE[token]})
            send_json(self, 429, {
                "error": "request budget exhausted",
                "error_kind": "proxy_budget",
            })
            return

        if MOCK:
            status, response = 200, mock_completion(token)
        else:
            if not API_KEY:
                send_json(self, 500, {"error": "proxy has no OPENROUTER_API_KEY"})
                return
            status, response = forward_openrouter(body, TOKENS[token])

        usage = response.get("usage") if isinstance(response, dict) else None
        if isinstance(usage, dict):
            with LOCK:
                TOKEN_USAGE[token] += int(usage.get("total_tokens") or 0)

        log({
            "event": "completion",
            "agent": role_of(token),
            "model": TOKENS[token],
            "mock": MOCK,
            "status": status,
            "usage": usage,
            "cumulative_tokens": TOKEN_USAGE[token],
            "requests_used": REQUEST_COUNT[token],
            "messages_in": len(body.get("messages") or []),
        })
        send_json(self, status, response)

    def log_message(self, fmt, *args):
        pass


def main():
    if not TOKENS:
        raise SystemExit("TOKENS_JSON is empty - no agents configured")
    server = ThreadingHTTPServer(("0.0.0.0", PORT), ProxyHandler)
    server.daemon_threads = True
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
