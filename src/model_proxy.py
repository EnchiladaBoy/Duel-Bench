#!/usr/bin/env python3
"""Model proxy for agent-deathmatch.

Runs in its own container on the battle network. Agent harnesses POST
OpenAI-format chat completions with a per-agent bearer token; the proxy:
- maps token -> configured model (agents can never impersonate each other's
  model or pick arbitrary ones);
- injects the host OPENROUTER_API_KEY (battle containers never see it);
- enforces a per-agent request budget;
- in MOCK_BACKEND=1 mode returns deterministic scripted tool calls instead
  of calling any API (zero-cost pipeline verification).
"""

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("PROXY_PORT", "8080"))
TOKENS = json.loads(os.environ.get("TOKENS_JSON", "{}"))
OPENROUTER_URL = os.environ.get(
    "OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions"
)
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MOCK = os.environ.get("MOCK_BACKEND", "0") == "1"
MOCK_SCRIPTS = json.loads(os.environ.get("MOCK_SCRIPTS_JSON", "{}"))
MOCK_SLEEP = json.loads(os.environ.get("MOCK_SLEEP_JSON", "{}"))
MAX_REQUESTS = int(os.environ.get("MAX_REQUESTS", "200"))
LOG_PATH = os.environ.get("PROXY_LOG", "/logs/proxy.jsonl")
LOCKSTEP = os.environ.get("LOCKSTEP", "0") == "1"
ROUND_TIMEOUT = float(os.environ.get("ROUND_TIMEOUT", "90"))

LOCK = threading.Lock()
REQUEST_COUNT = {token: 0 for token in TOKENS}
MOCK_INDEX = {token: 0 for token in TOKENS}

# Lockstep barrier state
BARRIER_COND = threading.Condition()
barrier = {
    "round": 1,
    "joined": set(),
    "first_join": None,
    "releases": {},  # round -> {"both": bool, "joined": [tokens]}
}


def log(record):
    record = {"ts": time.time(), **record}
    line = json.dumps(record, ensure_ascii=False)
    try:
        with LOCK:
            log_dir = os.path.dirname(LOG_PATH)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            with open(LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception:
        pass
    print(line, flush=True)


def barrier_join(token):
    """Register this agent's move for the current round. Returns the round
    number joined. Releases the round immediately once every agent has joined."""
    with BARRIER_COND:
        st = barrier
        round_no = st["round"]
        st["joined"].add(token)
        if st["first_join"] is None:
            st["first_join"] = time.time()
        if len(st["joined"]) >= len(TOKENS):
            rel = _release_unlocked(both=True)
            return {"round": round_no, "released": True, **rel}
        return {"round": round_no, "released": False, "joined": sorted(st["joined"])}


def _release_unlocked(both):
    rel = {"both": both, "joined": sorted(barrier["joined"])}
    barrier["releases"][barrier["round"]] = rel
    round_no = barrier["round"]
    barrier["round"] += 1
    barrier["joined"] = set()
    barrier["first_join"] = None
    if len(barrier["releases"]) > 200:
        for stale in sorted(barrier["releases"])[:-200]:
            del barrier["releases"][stale]
    BARRIER_COND.notify_all()
    return rel


def barrier_wait(round_no):
    """Block until the given round is released: either every agent joined, or
    ROUND_TIMEOUT elapsed since the first join (partial release — joined agents
    proceed alone). Returns the release record for that round."""
    with BARRIER_COND:
        while round_no not in barrier["releases"]:
            if barrier["round"] != round_no:
                # Round advanced without a recorded release (shouldn't happen);
                # synthesize a partial release so waiters don't hang.
                barrier["releases"][round_no] = {"both": False, "joined": []}
                break
            first_join = barrier["first_join"] or time.time()
            remaining = (first_join + ROUND_TIMEOUT) - time.time()
            if remaining <= 0:
                _release_unlocked(both=False)
                break
            BARRIER_COND.wait(min(remaining, 5.0))
        return {"round": round_no, **barrier["releases"][round_no]}


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
            call_id = f"mock_call_{token}_{idx}"
            return {
                "id": f"mock-{token}-{idx}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "mock",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": "run_bash",
                                        "arguments": json.dumps(
                                            {"command": command}
                                        ),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
        return {
            "id": f"mock-{token}-done",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mock",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "No further actions.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }


def forward_openrouter(body, model):
    payload = dict(body)
    payload["model"] = model
    payload.pop("stream", None)
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
        return exc.code, parsed
    except Exception as exc:
        return 502, {"error": f"{type(exc).__name__}: {exc}"}


def send_json(handler, status, obj):
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


class ProxyHandler(BaseHTTPRequestHandler):
    server_version = "AgentDeathmatchProxy/1.0"

    def _auth_token(self):
        auth = self.headers.get("Authorization", "")
        token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        return token if token in TOKENS else None

    def do_GET(self):
        if self.path == "/health":
            send_json(
                self,
                200,
                {
                    "status": "ok",
                    "mock": MOCK,
                    "agents": len(TOKENS),
                    "requests_served": dict(REQUEST_COUNT),
                    "lockstep": LOCKSTEP,
                },
            )
        elif self.path.startswith("/barrier/wait"):
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
            result = barrier_wait(round_no)
            log({"event": "barrier_release", "token": token, **result})
            send_json(self, 200, result)
        else:
            send_json(self, 404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/barrier/join":
            if not LOCKSTEP:
                send_json(self, 400, {"error": "lockstep disabled"})
                return
            token = self._auth_token()
            if not token:
                send_json(self, 401, {"error": "unknown agent token"})
                return
            result = barrier_join(token)
            send_json(self, 200, result)
            return

        if self.path != "/v1/chat/completions":
            send_json(self, 404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            send_json(self, 400, {"error": "invalid JSON body"})
            return

        auth = self.headers.get("Authorization", "")
        token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        if token not in TOKENS:
            log({"event": "auth_rejected", "has_token": bool(token)})
            send_json(self, 401, {"error": "unknown agent token"})
            return

        with LOCK:
            if REQUEST_COUNT[token] >= MAX_REQUESTS:
                over_budget = True
            else:
                REQUEST_COUNT[token] += 1
                over_budget = False
        if over_budget:
            log({"event": "budget_exhausted", "token": token})
            send_json(self, 429, {"error": "request budget exhausted"})
            return

        if MOCK:
            status, response = 200, mock_completion(token)
        else:
            if not API_KEY:
                send_json(self, 500, {"error": "proxy has no OPENROUTER_API_KEY"})
                return
            status, response = forward_openrouter(body, TOKENS[token])

        log(
            {
                "event": "completion",
                "token": token,
                "model": TOKENS[token],
                "mock": MOCK,
                "status": status,
                "usage": response.get("usage"),
                "messages_in": len(body.get("messages") or []),
            }
        )
        send_json(self, status, response)

    def log_message(self, fmt, *args):
        pass


def main():
    if not TOKENS:
        raise SystemExit("TOKENS_JSON is empty - no agents configured")
    server = ThreadingHTTPServer(("0.0.0.0", PORT), ProxyHandler)
    log(
        {
            "event": "proxy_start",
            "port": PORT,
            "mock": MOCK,
            "models": list(TOKENS.values()),
            "max_requests": MAX_REQUESTS,
        }
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
