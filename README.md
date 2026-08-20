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
- Python 3.8+ on the host (everything is stdlib-only, no dependencies)
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

### Flags

| Flag | Default | Purpose |
|---|---|---|
| `--model-a` / `--model-b` | `mock/agent-*` | OpenRouter model ids |
| `--mock` | off | deterministic scripted agents, no API calls |
| `--fair` | off | lockstep rounds (see below) |
| `--time-limit` | 600 | match wall clock, seconds |
| `--max-steps` | 80 | model turns per agent |
| `--max-requests` | 200 | proxy request budget per agent |
| `--max-tokens-budget` | 0 | per-agent total token budget (0 = unlimited) |
| `--max-tokens-per-call` | 4096 | cap on `max_tokens` for any upstream call |
| `--temperature` / `--seed` | unset | applied identically to both agents, recorded in the result |
| `--command-timeout` | 30 | seconds per `run_bash` command |
| `--move-timeout` | 90 | fair mode: barrier wait per round |
| `--poll-interval` | 2.0 | monitor loop period |
| `--grace-seconds` | 10 | heartbeat silence before an agent is declared down |
| `--startup-timeout` | 90 | proxy and agent readiness |
| `--memory` / `--cpus` / `--pids-limit` | 512m / 1.0 / 256 | per-agent container limits |
| `--image` | content hash | override the battle image tag |
| `--build` | off | force an image rebuild |
| `--egress-network` | `podman` | network the proxy joins for internet access |
| `--no-internal-network` | off | skip internal networking (result is unrated) |
| `--allow-degraded` | off | play even if the arena is not to spec (result is unrated) |
| `--keep` | off | leave containers running for debugging |
| `--mock-script-a/-b`, `--mock-delay-a/-b` | — | mock-mode scripting and latency simulation |

`elo.py` takes `--matches-dir`, `--k`, `--min-games`, `--include-degraded`,
`--include-mock`, and `--quiet`.

## Arena integrity

A match only measures something if the arena is intact. Two conditions are
checked before any API budget is spent, and the match is **refused** rather
than played if either fails:

- **Shared PID namespace.** Without it the agents cannot see or signal each
  other, so no win condition can ever occur.
- **Internal network.** Without it the agents have unrestricted internet.

`--allow-degraded` forces the match anyway; the result is then written with
`"rated": false` and the leaderboard ignores it.

## What counts toward the leaderboard

`elo.py` rates a match only when all of these hold. Everything it skips is
listed on stdout with a reason, never dropped silently.

- the outcome is `agent-a`, `agent-b`, or `draw` (not `error` / `aborted`)
- it was not a `--mock` run
- the arena was intact (`pid_shared` and `network_internal`)
- the orchestrator did not mark it `rated: false` — which it does when an agent
  stopped for a non-game reason (request budget spent, proxy unreachable) or
  when an agent never executed a single command

Ratings are shown with a bootstrap 95% interval and marked provisional below
`--min-games`. With a handful of matches that interval spans most of the table,
which is the honest reading rather than a defect.

## Game modes

Speed and intelligence trade off against each other: fast models are usually less
capable. A wall-clock arena rewards provider latency; a purely turn-based one erases
speed entirely. Neither is "correct" — they answer different questions. So the arena has
four modes, **each with its own leaderboard**, the way chess keeps separate
classical/rapid/blitz ratings. Ratings are never pooled across modes.

| `--mode` | Rule | What it measures |
|---|---|---|
| `untimed` | Lockstep rounds, equal turns each, no clock at all. Ends after `--max-rounds`. | Pure strategy — speed is removed entirely. |
| `move-timed` | Lockstep rounds with a firm per-move deadline. Exceed it and that round is forfeited: your command does not run, your opponent's does. | Strategy with a latency floor to clear. |
| `time-bank` | Each agent gets a total thinking-time bank; its real inference time is deducted per move. **The default.** | The tradeoff itself. |
| `realtime` | No turn-taking, wall-clock bounded. Acting sooner is an advantage. | Speed as a legitimate weapon. |

The round cap and the move deadline are enforced **by the proxy**, not by the
orchestrator's polling — otherwise whichever agent asked first would get a bonus move
and "equal turns each" would quietly stop being true.

```bash
python3 src/orchestrator.py --mode time-bank --time-bank 300 \
    --model-a openai/gpt-4o-mini --model-b anthropic/claude-3.5-haiku
```

### time-bank

The most interesting of the four, and the one to watch. Both agents move in lockstep, but
each is charged the time its **own** reasoning actually took. **Waiting at the barrier is
free**, so a fast model is never punished for having a slow opponent. A 30s/move model and
a 3s/move model trade blows evenly until the slow one's bank empties — then it can no
longer act while the opponent plays on with everything it has left. Slow models get few
moves; fast models get many; budgeting is a strategy.

Both clocks are public. The remaining bank is shown to each model every turn, because in
lockstep the time spent waiting at the barrier already *is* the opponent's thinking time —
hiding it would only reward whichever model thought to measure that.

Per-move inference time is measured **in the proxy**, with a monotonic clock, around the
completion call only. The harness runs in a container the agent has a shell in, so nothing
that affects scoring is timed there.

### move-timed

A forfeited round is scored as engagement, not as absence: losing rounds to the deadline
is participation, and it is exactly what the mode measures. A model too slow to meet the
deadline therefore still produces a rating rather than an unrated no-contest.

### Mock-mode testing

`--mock-delay-a` / `--mock-delay-b` simulate slow-model latency, and the mock delay is
inside the timed region — so the whole time-bank mechanism is exercisable at **zero API
cost**:

```bash
python3 src/orchestrator.py --mock --mode time-bank --time-bank 20 \
    --mock-delay-a 5 --mock-delay-b 0.5
# agent-a gets 4 moves, agent-b gets 40, and the match ends on banks_exhausted
```

## How lockstep works

The three lockstep modes (`untimed`, `move-timed`, `time-bank`) share one mechanism:

- Each round, both agents think independently, then **commit** their move at a barrier
  hosted by the model proxy (`POST /barrier/join` + `GET /barrier/wait`).
- Once both moves are committed they execute **simultaneously**, so inference speed does
  not decide who strikes first.
- If an agent stops participating — a spent time bank, or repeated missed deadlines — it
  drops out of the quorum so the survivor is not stalled waiting for a player that will
  never move again.
- Mutual destruction is possible: if both commit a kill in the same round, both may die.
  That is scored as a draw, and always counts toward the leaderboard.
- Win detection is unchanged (container exit / heartbeat loss); the barrier only paces
  execution.

`--fair` is a **deprecated** alias for `--mode untimed` and prints a warning.

## Match artifacts

Each match writes to `matches/<match_id>/`:

- `result.json` — winner, reason, rated flag, duration, per-agent token usage,
  exit codes, commands executed, and the full arena configuration (image
  digest, sampling params, container limits) so a result can be audited later
- `agent-a/agent.jsonl`, `agent-b/agent.jsonl` — every model response and
  command executed (full battle replay). These are captured from the
  container's **stdout** on the host, so an agent cannot rewrite the record
  that judges it.
- `events.jsonl` — **the whole match as one ordered timeline**, merged live on the
  host from three sources (both agents' stdout via `podman logs -f`, and the proxy's
  own log). Written as the match happens, so it can be watched live or replayed
  afterwards. See below.
- `proxy/proxy.jsonl` — API request log with per-call token usage and per-move timing
- `proxy-resolv.conf` — DNS config mounted into the proxy container

The `OPENROUTER_API_KEY` is **never** written into this directory. It lives in
a `0600` file inside a `0700` private temp directory for the lifetime of the
match and is removed on every exit path, including `SIGTERM`.

## Watching a match

`matches/<id>/events.jsonl` is written **while the match runs**. Each record carries an
envelope — `seq` (total order), `t` (seconds since the match started), `src`, `event` —
so the three containers' unrelated wall clocks never decide ordering. A source's own
timestamp is kept as `src_ts` for skew analysis only, and a source cannot forge its own
`src` or `seq`.

```bash
tail -f matches/<id>/events.jsonl | python3 -c 'import json,sys
for l in sys.stdin:
    d = json.loads(l)
    print(f"{d[\"t\"]:>7.1f}  {d[\"src\"]:<12} {d[\"event\"]:<14} "
          f"{d.get(\"command\") or d.get(\"bank_remaining\") or \"\"}")'
```

Useful events: `match_start`, `arena_ready`, `go` (the starting gun), `move_start`,
`thinking` (progress while a model is still reasoning), `command_start`,
`command_result`, `barrier_release`, `bank_exhausted`, `snapshot` (the full scoreboard,
once per poll, so a viewer joining mid-match renders immediately), `agent_down`,
`match_end`.

For a clock UI, `bank_remaining` appears on `move_start` and on each `thinking` tick, so
the display can decrement locally between them and snap to the authoritative value —
which is how a chess clock is rendered.

**The feed is host-side only, deliberately.** The proxy's HTTP server is reachable by
both agents, so serving a spectator feed from it would let an agent poll its opponent's
every command and its output — destroying reconnaissance as a skill. Nothing an agent
can reach exposes the stream.

The live stream is **best-effort**: a dropped line costs a viewer one frame and is
counted and reported. The durable audit record is still `agent-*/agent.jsonl`, collected
in full at teardown, which an agent cannot rewrite.

## Fairness

Two mechanisms remove bias that has nothing to do with model skill:

- **The starting gun.** Agent containers are created sequentially, so one is ready
  fractionally earlier. Each agent's *first* model call is held until both have arrived,
  then both are released together. Waiting costs nothing and is never charged to a time
  bank.
- **Side shuffling** (on by default; `--no-shuffle-sides` to disable). Which model plays
  `agent-a` is randomised per match and recorded in `result.json` as `side_assignment`,
  so the first-mover edge cannot accumulate in one model's favour.

## Safety model

What the code actually enforces:

- Agents run in rootless Podman containers as a **non-root user**, with
  `--cap-drop=ALL`, `--security-opt=no-new-privileges`, and memory/CPU/PID
  limits. The battle image is pinned by digest.
- The battle network is `--internal`, so the containers cannot route to the
  internet. If that cannot be created, the match is refused (see Arena
  integrity) rather than quietly played on a routable network.
- The API key is held only by the proxy container and never appears in the
  agents' environment, the repo, or any log.
- Per-agent bearer tokens are never disclosed to the opponent: the token is
  mounted as a file into one container rather than passed in the environment
  (both agents share a PID namespace, so `/proc/<pid>/environ` would leak it),
  and no endpoint or barrier response echoes token material.
- Agents get **no writable host bind mount**. Battle logs are the container's
  stdout, collected on the host at teardown.
- Teardown removes containers, pod, and network on every exit path including
  `SIGTERM`, and warns loudly if any resource could not be removed.

What it does **not** enforce, stated plainly:

- The proxy is a relay to OpenRouter. An agent can send arbitrary prompt text
  out and read arbitrary text back within its request budget. "No egress" means
  no *routable* network, not no information channel.
- The system prompt forbids escape attempts. That is a prompt, not a control.
  Escape *attempts* are visible in the battle logs and are arguably the most
  interesting data this benchmark produces.
- Both agents run as the same uid in a shared PID namespace — that is the game.
  A sufficiently determined agent could use same-uid process access against the
  opponent's harness. Stealing the opponent's proxy token no longer wins,
  though: budget exhaustion marks the match unrated rather than scoring it.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

143 tests covering the scoring rules, the mode table, the lockstep barrier, the time
bank, request validation, credential disclosure, the event stream, and the command
runner. No dependencies, no containers needed.

## Status

Prototype. Next steps: multi-match tournaments **with side swapping** (agent-a
starts first, so pairs must be run both ways before any rating is meaningful),
strategy replay/analysis, and a public leaderboard page.

## License

MIT — see [LICENSE](LICENSE).
