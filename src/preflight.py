#!/usr/bin/env python3
"""Checks to run BEFORE spending anything on a real match.

    python3 src/preflight.py --check key
    python3 src/preflight.py --check models --require '~deepseek/deepseek-v4-flash-latest,qwen/qwen3.8-27b'
    python3 src/preflight.py --check completion --model '<id>'

The first two cost nothing: one free metadata request and one public catalog
request. The third sends a single real completion (fractions of a cent) using the
production payload, so a malformed request costs that instead of a whole match.

Egress from inside the proxy container is a separate check, because a host-side
probe proves nothing about a container on an --internal network:

    python3 src/orchestrator.py --preflight --model-a X --model-b Y
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

API_BASE = os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
GREEN, RED, YELLOW, RESET, DIM = "\033[32m", "\033[31m", "\033[33m", "\033[0m", "\033[2m"


def fetch(path, token=None, timeout=30):
    request = urllib.request.Request(f"{API_BASE}{path}", headers={"Accept": "application/json"})
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode("utf-8"))
        except (json.JSONDecodeError, OSError):
            return exc.code, {}
    except Exception as exc:
        return 0, {"error": f"{type(exc).__name__}: {exc}"}


# ----------------------------------------------------------------- pure parts

def parse_catalog(payload):
    """Normalise /models into just the fields that decide whether a model is
    usable here."""
    models = []
    for entry in (payload or {}).get("data") or []:
        if not isinstance(entry, dict) or not entry.get("id"):
            continue
        pricing = entry.get("pricing") or {}
        top = entry.get("top_provider") or {}
        models.append({
            "id": entry["id"],
            "prompt": _price(pricing.get("prompt")),
            "completion": _price(pricing.get("completion")),
            "context_length": entry.get("context_length") or 0,
            "max_completion_tokens": top.get("max_completion_tokens"),
            "supported_parameters": entry.get("supported_parameters") or [],
        })
    return models


def _price(value):
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    return None if price < 0 else price          # -1 means variable pricing


def supports_tools(model):
    """Free, exhaustive tool-calling check: a model that cannot be told about
    run_bash cannot play at all."""
    return "tools" in (model.get("supported_parameters") or [])


def price_per_mtok(model):
    """Combined prompt+completion price per million tokens, or None if the
    model does not publish a fixed price."""
    if model.get("prompt") is None or model.get("completion") is None:
        return None
    return (model["prompt"] + model["completion"]) * 1_000_000


def verify_ids(catalog, wanted):
    """Byte-for-byte membership. No normalising, no fuzzy matching - a near miss
    is a 404 discovered by burning a match."""
    known = {m["id"] for m in catalog}
    return [i for i in wanted if i in known], [i for i in wanted if i not in known]


def assess_completion(body):
    """Judge one real completion against everything a match depends on."""
    problems = []
    choices = (body or {}).get("choices")
    if not isinstance(choices, list) or not choices:
        return ["no choices in the response (provider error carried in a 200 body)"]
    choice = choices[0]
    message = choice.get("message") or {}
    calls = message.get("tool_calls") or []

    if choice.get("finish_reason") == "length":
        problems.append("finish_reason 'length': the max_tokens cap truncated the reply")
    if not calls:
        problems.append("no tool_calls: the model did not use run_bash")
    if len(calls) > 1:
        problems.append(f"{len(calls)} tool calls for a one-command instruction")
    if isinstance(message.get("content"), list):
        problems.append("content is a list of parts, not a string")

    for call in calls[:1]:
        function = call.get("function") or {}
        if function.get("name") != "run_bash":
            problems.append(f"called {function.get('name')!r}, not run_bash")
            continue
        raw = function.get("arguments")
        args = raw if isinstance(raw, dict) else None
        if args is None:
            try:
                args = json.loads(raw or "")
            except (json.JSONDecodeError, TypeError):
                problems.append("arguments did not parse as JSON")
                continue
        if not isinstance(args, dict) or not isinstance(args.get("command"), str):
            problems.append("arguments carried no string 'command'")

    usage = (body or {}).get("usage") or {}
    if not usage.get("total_tokens"):
        problems.append("no usage.total_tokens (token budgets would never advance)")
    if usage.get("cost") is None:
        problems.append("no usage.cost (cost caps cannot be enforced)")
    return problems


# ----------------------------------------------------------------- the checks

def check_key(token):
    status, body = fetch("/key", token)
    if status != 200:
        print(f"{RED}FAIL{RESET} /key returned {status}: {json.dumps(body)[:300]}")
        if status == 401:
            print("  the key is invalid or revoked")
        if status == 402:
            print("  the account is out of credit")
        return 1
    data = body.get("data") or body
    limit, used = data.get("limit"), data.get("usage")
    print(f"{GREEN}OK{RESET}   key is valid   usage={used}  limit={limit}  "
          f"free_tier={data.get('is_free_tier')}")
    if limit is not None and used is not None and limit - used <= 0:
        print(f"{RED}FAIL{RESET} no credit remaining")
        return 1
    return 0


def check_models(required, want_tools, top, min_completion):
    status, body = fetch("/models")
    if status != 200:
        print(f"{RED}FAIL{RESET} /models returned {status}")
        return 1
    catalog = parse_catalog(body)
    print(f"{DIM}{len(catalog)} models in the catalog{RESET}")

    if required:
        present, missing = verify_ids(catalog, required)
        for model_id in present:
            entry = next(m for m in catalog if m["id"] == model_id)
            tools = supports_tools(entry)
            price = price_per_mtok(entry)
            mark = GREEN + "OK  " + RESET if tools else RED + "FAIL" + RESET
            print(f"{mark} {model_id}   tools={tools}  "
                  f"ctx={entry['context_length']}  "
                  f"max_completion={entry['max_completion_tokens']}  "
                  f"${price:.2f}/Mtok" if price is not None
                  else f"{mark} {model_id}   tools={tools}  price=variable")
            if not tools:
                return 1
            if (min_completion and entry["max_completion_tokens"]
                    and entry["max_completion_tokens"] < min_completion):
                print(f"{RED}FAIL{RESET} max_completion_tokens "
                      f"{entry['max_completion_tokens']} < --max-tokens-per-call "
                      f"{min_completion}")
                return 1
        for model_id in missing:
            print(f"{RED}FAIL{RESET} {model_id!r} is not in the catalog "
                  f"(a match would die on 404 after paying for the attempt)")
        return 1 if missing else 0

    usable = [m for m in catalog
              if (not want_tools or supports_tools(m)) and price_per_mtok(m) is not None]
    usable.sort(key=price_per_mtok)
    print(f"\n{'MODEL':<52} {'$/Mtok':>9} {'CTX':>9}  TOOLS")
    for model in usable[:top]:
        print(f"{model['id']:<52} {price_per_mtok(model):>9.3f} "
              f"{model['context_length']:>9}  {supports_tools(model)}")
    return 0


def check_completion(token, model_id, max_tokens):
    """One real call, built by the production code path so the probe cannot
    drift from what a match actually sends."""
    os.environ.setdefault("MAX_TOKENS_PER_CALL", str(max_tokens))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import agent_harness
    import model_proxy

    payload = model_proxy.build_upstream_payload({
        "messages": [
            {"role": "system", "content": "You are a shell agent."},
            {"role": "user", "content": "Run `echo ok` using the run_bash tool. One command."},
        ],
        "tools": agent_harness.TOOLS,
        "tool_choice": "auto",
    }, model_id)

    request = urllib.request.Request(
        f"{API_BASE}/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}",
                 "X-Title": "agent-deathmatch-preflight"}, method="POST")
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            status, body = response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(f"{RED}FAIL{RESET} {model_id}: HTTP {exc.code} "
              f"{exc.read().decode(errors='replace')[:300]}")
        return 1
    except Exception as exc:
        print(f"{RED}FAIL{RESET} {model_id}: {type(exc).__name__}: {exc}")
        return 1
    elapsed = time.monotonic() - started

    problems = assess_completion(body)
    usage = body.get("usage") or {}
    print(f"{'  ' if problems else ''}{RED + 'FAIL' + RESET if problems else GREEN + 'OK  ' + RESET} "
          f"{model_id}   {elapsed:.2f}s   tokens={usage.get('total_tokens')}  "
          f"cost=${usage.get('cost', 0) or 0:.6f}  "
          f"finish={(body.get('choices') or [{}])[0].get('finish_reason')}")
    for problem in problems:
        print(f"       {YELLOW}- {problem}{RESET}")
    if not problems:
        print(f"       {DIM}latency {elapsed:.2f}s — size --time-bank from this{RESET}")
    return 1 if problems else 0


def main():
    parser = argparse.ArgumentParser(description="pre-match checks for agent-deathmatch")
    parser.add_argument("--check", required=True,
                        choices=["key", "models", "completion"])
    parser.add_argument("--require", default="",
                        help="comma-separated model ids that must exist, byte for byte")
    parser.add_argument("--model", default="", help="model id for --check completion")
    parser.add_argument("--tools", action="store_true",
                        help="list only models that support tool calling")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--max-tokens-per-call", type=int, default=1024)
    args = parser.parse_args()

    token = os.environ.get("OPENROUTER_API_KEY", "")
    if args.check in ("key", "completion") and not token:
        sys.exit("OPENROUTER_API_KEY is not set")

    if args.check == "key":
        return check_key(token)
    if args.check == "models":
        required = [m.strip() for m in args.require.split(",") if m.strip()]
        return check_models(required, args.tools, args.top, args.max_tokens_per_call)
    if not args.model:
        sys.exit("--check completion needs --model")
    return check_completion(token, args.model, args.max_tokens_per_call)


if __name__ == "__main__":
    raise SystemExit(main())
