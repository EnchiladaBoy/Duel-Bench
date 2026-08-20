# agent-deathmatch

An adversarial AI benchmark: two LLM agents are pitted against each other in
an isolated Podman sandbox. Each agent gets a shell on a shared machine
(shared PID + network namespaces, separate filesystems) and the objective is
simple — **stop the opponent's agent process (or wreck its environment)
before it does the same to you.**

Winner detection, ELO ranking, and full JSONL battle logs included.

## Architecture

```
host
├── orchestrator.py          match lifecycle, win detection, teardown
├── podman network (internal, no internet egress for agents)
│   ├── proxy container      model_proxy.py — the ONLY component with egress
│   │                        (holds OPENROUTER_API_KEY; agents never see it)
│   └── pod (shared net/pid/ipc/uts namespaces)
│       ├── agent-a          agent_harness.py + bash, curl, nmap, pkill, ...
│       └── agent-b          agent_harness.py + bash, curl, nmap, pkill, ...
```

- **agent_harness.py** runs as each container's entrypoint. It serves a
  heartbeat HTTP endpoint, drives an OpenAI-style tool loop against the model
  proxy (`run_bash` tool calls → executed in `/battle`), and logs every step.
- **model_proxy.py** maps per-agent bearer tokens to models, injects the
  OpenRouter key, enforces a per-agent request budget, and supports a
  deterministic mock backend for zero-cost pipeline tests.
- **orchestrator.py** wires it all up, monitors container state + heartbeats,
  declares the winner, writes `matches/<id>/result.json`, and tears down.

### Win conditions

| Condition | Detection |
|---|---|
| Opponent's harness process killed (e.g. `pkill`) | container exits |
| Environment wrecked (fork bomb, hang, corruption) | heartbeat lost for grace period |
| Nobody wins within the time limit | draw |

## Requirements

- Fedora/Linux with **Podman** (rootless is fine; Docker is not required)
- Python 3.9+ on the host (orchestrator is stdlib-only)
- `OPENROUTER_API_KEY` in the environment for real matches

## Quick start

```bash
cd ~/agent-deathmatch

# 1. Zero-cost pipeline test with deterministic mock agents
python3 src/orchestrator.py --mock --time-limit 120

# 2. A real battle via OpenRouter
export OPENROUTER_API_KEY=sk-or-...
python3 src/orchestrator.py \
    --model-a openai/gpt-4o-mini \
    --model-b anthropic/claude-3.5-haiku \
    --time-limit 600 --max-steps 80

# 3. Leaderboard after several matches
python3 src/elo.py
```

Useful flags: `--keep` (don't tear down, for debugging), `--build` (force
image rebuild), `--no-internal-network` (if internal networking misbehaves),
`--time-limit / --max-steps / --max-requests` budgets.

## Fair mode (`--fair`)

Without intervention, a fast-inference model can kill a slow one before the
slow model finishes its first move — the arena rewards latency, not strategy.
`--fair` removes this advantage with **lockstep rounds**:

```bash
python3 src/orchestrator.py --fair \
    --model-a '~deepseek/deepseek-v4-flash-latest' \
    --model-b 'qwen/qwen3.8-27b' \
    --move-timeout 90
```

- Each round, both agents think independently, then **commit** their move at a
  barrier hosted by the model proxy (`POST /barrier/join` + `GET /barrier/wait`).
- Once both moves are committed, they execute **simultaneously**. A fast model
  waits at the barrier for a slow one; inference speed no longer decides who
  strikes first.
- If one agent fails to commit within `--move-timeout` seconds, the round is
  released partially and the waiting agent executes alone (a stalled model
  loses rounds rather than stalling the match forever).
- Mutual destruction is possible: if both commit a kill in the same round,
  both may die — scored as a draw.
- Win detection is unchanged (container exit / heartbeat loss); the barrier
  only paces execution.

Related testing flags (mock mode only): `--mock-delay-a` / `--mock-delay-b`
simulate slow-model latency in seconds per response, to verify the barrier
holds rounds together.

## Match artifacts

Each match writes to `matches/<match_id>/`:

- `result.json` — winner, reason, duration, config
- `agent-a/agent.jsonl`, `agent-b/agent.jsonl` — every model response and
  command executed (full battle replay)
- `proxy/proxy.jsonl` — API request log

## Safety model

- Agents run in rootless Podman containers with memory/CPU/PID limits, no
  privileged mode, and (when supported) an `--internal` network: the battle
  containers have **no internet egress at all** — model calls must go through
  the proxy, which is the only holder of the API key.
- The system prompt forbids escape attempts; anything that happens stays in
  the sandbox. Escape *attempts* are visible in the battle logs (and are
  arguably the most interesting data this benchmark can produce).
- Teardown removes containers, pod, and network after every match.

## Status

Prototype. Next steps: multi-match tournaments, strategy replay/analysis,
public leaderboard page.
