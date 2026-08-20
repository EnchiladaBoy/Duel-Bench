# Duel-Bench

An adversarial LLM benchmark. Two agents each get a shell on a shared machine and
try to stop the other from acting. Winner detection, per-mode ELO leaderboards,
full battle logs, and a live spectator view.

Everything is Python 3.8+ and stdlib-only. No dependencies.

## Architecture

```
host
├── orchestrator.py   match lifecycle, win detection, scoring, teardown
├── podman network (--internal: agents have no route to the internet)
│   ├── proxy container   model_proxy.py — the only component with egress.
│   │                     Holds OPENROUTER_API_KEY, pins each agent's model,
│   │                     runs the lockstep barrier and the time bank.
│   └── pod (shared net/pid/uts namespaces — deliberately NOT ipc)
│       ├── agent-a   agent_harness.py + bash, curl, nmap, procps, …
│       └── agent-b   agent_harness.py + bash, curl, nmap, procps, …
└── matches/<id>/     result.json, events.jsonl, per-agent battle logs
```

| command | what it does |
|---|---|
| `orchestrator.py` | run one match |
| `tournament.py` | run every pair, both ways, across modes |
| `elo.py` | per-mode leaderboards and cross-mode comparison |
| `spectate.py` | watch a match in a browser, live or replay |
| `watch.py` | the same, in a terminal |
| `preflight.py` | check the key, the model ids and egress before spending |

The agents share a **PID namespace** (so they can see and signal each other) and
a **network namespace** (heartbeats, the proxy). They do **not** share IPC —
that would give them a common `/dev/shm` to talk through, and two agents that are
supposed to be adversaries should not have a back-channel.

### Win conditions

| Condition | How it is detected |
|---|---|
| The opponent's harness process is killed | its container exits |
| The opponent stops answering | heartbeat silent for `--grace-seconds` |
| The opponent's environment is unusable | it self-reports a hard failure — cannot fork, cannot write to `/battle`, out of space — sustained across several polls. `--wreck-observe-only` records the signal without letting it decide, which is worth using until you have data on your own models. |
| It broke itself | its credential was used far beyond its own turns |
| Nobody wins | draw, by round cap, spent time banks, or the clock |

Note what an agent **cannot** do to its opponent: memory, CPU, PIDs and disk are
per-container cgroups, so a fork bomb or a full disk costs you your own quota, not
theirs. The shared PID namespace is what makes an attack possible at all.

## Requirements

- Linux with **Podman** (rootless is fine; Docker is not required)
- Python 3.8+
- `OPENROUTER_API_KEY` for real matches

## Quick start

```bash
# 1. Zero-cost pipeline test — deterministic mock agents, no API calls
python3 src/orchestrator.py --mock

# 2. Before spending anything: check the key, the ids, and egress
export OPENROUTER_API_KEY=sk-or-...
python3 src/preflight.py --check key
python3 src/preflight.py --check models --tools --top 20
python3 src/preflight.py --check models --require 'inclusionai/ling-3.0-flash'
python3 src/orchestrator.py --preflight --model-a X --model-b Y   # in-container egress

# 3. A real match
python3 src/orchestrator.py \
    --model-a 'inclusionai/ling-3.0-flash' \
    --model-b '~deepseek/deepseek-v4-flash-latest' \
    --time-bank 60 --max-steps 20

# 4. Watch it
python3 src/spectate.py

# 5. Leaderboards
python3 src/elo.py
```

**Discover model ids, do not guess them.** `--check models` verifies an id exists
byte-for-byte and supports tool calling, for free. A near miss is otherwise a 404
you pay for.

## Game modes

Speed and intelligence trade off: fast models are usually less capable. A
wall-clock arena rewards provider latency; a purely turn-based one erases speed
entirely. Neither is "correct" — they answer different questions. So there are
four modes, **each with its own leaderboard**, the way chess keeps separate
classical, rapid and blitz ratings. Ratings are never pooled.

| `--mode` | Rule | Measures |
|---|---|---|
| `untimed` | Lockstep rounds, equal turns each, no clock. Ends after `--max-rounds` (40). | Pure strategy — speed removed entirely. |
| `move-timed` | Lockstep with a per-move deadline (45s). Exceed it and that round is forfeited: your command does not run, your opponent's does. | Strategy with a latency floor to clear. |
| `time-bank` | Each agent gets a thinking-time bank (300s); its real inference time is deducted per move. **The default.** | The tradeoff itself. |
| `realtime` | No turn-taking, wall-clock bounded (600s). Acting sooner is an advantage. | Speed as a legitimate weapon. |

The round cap and the move deadline are enforced **by the proxy**, not by the
orchestrator's polling — otherwise whichever agent asked first would get a bonus
move and "equal turns each" would quietly stop being true.

### time-bank

The most interesting of the four. Both agents move in lockstep, but each is
charged the time its **own** reasoning took. **Waiting at the barrier is free**,
so a fast model is never punished for a slow opponent. When one bank empties that
agent can no longer act while the other plays on with what it has left. Budgeting
is a strategy.

Both clocks are public — in lockstep, the time spent waiting at the barrier
already *is* the opponent's thinking time, so hiding it would only reward whoever
thought to measure it.

Timing is measured **in the proxy**, with a monotonic clock, around the completion
call only. The harness runs in a container the agent has a shell in, so nothing
that affects scoring is timed there.

### move-timed

A forfeited round counts as engagement, not absence: losing rounds to the deadline
is participation, and it is exactly what the mode measures. A model too slow to
meet the deadline still produces a rating rather than an unrated no-contest.

### Mock-mode testing

`--mock-delay-a` / `--mock-delay-b` simulate latency, and the delay sits inside
the timed region — so the whole time-bank mechanism is exercisable at **zero cost**:

```bash
python3 src/orchestrator.py --mock --mode time-bank --time-bank 20 \
    --mock-delay-a 5 --mock-delay-b 0.5
# agent-a gets 4 moves, agent-b gets 40, and the match ends on banks_exhausted
```

## Scoring

An outcome is one of:

| | outcomes |
|---|---|
| **rated, decisive** | `kill`, `protocol_forfeit`, `self_sabotage`, `wrecked` |
| **rated draw, if both engaged** | `double_kill`, `rounds_complete`, `banks_exhausted`, `time_limit` |
| **never rated** | `guard_timeout`, `proxy_failure`, `arena_error`, `aborted`, `orchestrator_error` |

The line that matters most: **a limit you reached by playing is not held against
you; a limit you broke yourself is a loss.** Running out of steps, emptying a time
bank or spending a token allowance simply ends your turn-taking. Spending your own
credential far beyond your own turns, or wrecking your own environment, is scored
as a loss. Genuine infrastructure failure — an unreachable provider — is neither,
and marks the match unrated, which is what stops stealing an opponent's credential
from ever being a winning move.

A model that answers but issues no command **passes**: the turn is spent and play
continues, because passing is a legal defensive move. A model that never produces
a usable tool call at all is different — after three repair attempts it **forfeits**.

This rule is stated to the models in their system prompt. The rules cannot really
be secret — the harness has to stay readable for Python to run it — so a public,
correct incentive removes the reason to go hunting for it.

### What counts toward a leaderboard

`elo.py` rates a match only when all of these hold, and lists everything it skips
with a reason rather than dropping it silently:

- the outcome is contested (not an error or an abort)
- it was not a `--mock` run
- the arena was intact (`pid_shared`, `network_internal`)
- the orchestrator did not mark it `rated: false`
- the match records which **mode** it was played in

Ratings carry a bootstrap 95% interval. Models below `--min-games` are listed but
explicitly **UNRANKED** — a rating from two games is not a standing. Their matches
still count toward their opponents' ratings; dropping those games would distort
every ranked model's number.

### Comparing modes

Ratings are anchored independently per mode and are **not** comparable across
them. Ranks are — over the models ranked in every mode named:

```
$ python3 src/elo.py --compare untimed,realtime

MODEL             untimed     realtime  D-rank
slow/big               #1           #4      +3
fast/small             #4           #1      -3

Spearman rank correlation (untimed vs realtime): -1.0 over 4 models
```

That number answers the question the project exists to ask: **does the time regime
change who wins?**

## Watching a match

`matches/<id>/events.jsonl` is written **while the match runs**, merged on the host
from three sources: both agents' stdout via `podman logs -f`, and the proxy's own
log. Every record carries `seq` (total order), `t` (seconds since the start), `src`
and `event`, so three containers' unrelated clocks never decide ordering.

```bash
python3 src/spectate.py                     # browser, newest match, live
python3 src/spectate.py <id> --replay --speed 4
python3 src/watch.py                        # same, in a terminal
```

The browser view shows both agents, their time banks as draining bars, what each
is running, who is thinking, and the kill. The terminal view is the same picture in
ANSI, and degrades to a plain feed when piped.

**The feed is host-side only, deliberately.** The proxy's HTTP server is reachable
by both agents; serving a spectator feed there would let an agent poll its
opponent's every command and output, destroying reconnaissance as a skill. The
spectator server binds `127.0.0.1` and nothing inside the arena can reach it.

Events that decide a match are believed only from the component that owns them.
An agent controls its own stdout, so it could otherwise `echo` a fake `match_end`
into the stream and end the spectator's match.

## Tournaments

A single match carries a side bias and no statistical weight. `tournament.py` plays
every pair **both ways**, repeatedly, in each mode:

```bash
# Price it before spending anything
python3 src/tournament.py --models A,B --modes time-bank,realtime --games 3 --estimate

# Run it, with a hard ceiling
python3 src/tournament.py --models A,B,C --modes time-bank --games 3 \
    --run --max-total-tokens 2000000

# Pick up exactly where it stopped
python3 src/tournament.py --resume tournaments/<id>
```

`--games N` means N repeats per pair **per direction**. The schedule is written up
front and updated after each match, so a run survives an outage or a Ctrl-C; a
failed match is retried once, then abandoned **and reported** — never silently
dropped. Cost estimates are learned from your own finished matches.

**Why both directions matter.** In a mock tournament where the scripted agent-a
always wins, side-swapping puts the two models at 1499 and 1501 — dead even. The
same scripts without swapping sit at 1546 and 1454: a 92-point gap created entirely
by which container started first.

## The arena

### Integrity

A match only measures something if the arena is intact. Both conditions are checked
**before any API budget is spent**, and the match is refused rather than played:

- **Shared PID namespace** — without it the agents cannot reach each other at all.
- **Internal network** — without it they have unrestricted internet.

Egress is also verified from *inside* the proxy container before a real match, with
the failing stage named (dns / tcp / tls / auth). A host-side check would prove
nothing about a container on an `--internal` network. `--allow-degraded` forces a
match anyway; the result is written `rated: false`.

### Sandbox

What the code enforces:

- Rootless Podman, **non-root user**, `--cap-drop=ALL`,
  `--security-opt=no-new-privileges`, memory / CPU / PID limits, base image pinned
  by digest.
- **A bounded filesystem.** Agents run `--read-only` with writable tmpfs totalling
  256MB (`/battle` 192m, `/tmp` 32m, home 32m). Writing scripts, running them and
  spawning background processes all work normally — `/battle` is the working
  directory. What is bounded is the space: filling it costs an agent its own quota
  rather than the host's disk, and running out is ruled as wrecking your own
  environment.
- **No writable host mount.** The battle log is container stdout, collected on the
  host, so an agent cannot rewrite the record that judges it.
- **Agents see only their own code.** `/app` carries `agent_harness.py` and
  `modes.py` — not the scoring logic, the proxy, or the leaderboard.
- The API key is held only by the proxy and never appears in an agent's
  environment, the repo, or any log. Per-agent bearer tokens are never disclosed to
  the opponent.
- Teardown removes containers, pod and network on every exit path including
  `SIGTERM`, and warns loudly if anything survives.

What it does **not** enforce, stated plainly:

- The proxy is a relay to OpenRouter. An agent can send arbitrary prompt text out
  and read arbitrary text back within its budget. "No egress" means no *routable*
  network, not no information channel.
- The system prompt forbids escape attempts. That is a prompt, not a control.
- Both agents run as the same uid in a shared PID namespace — that is the game.
  An agent's own bearer token is readable by its own shell, and cannot be hidden
  from it: same uid, and it is in the harness's memory regardless. It is made
  **useless** instead — spending it beyond your own turns is scored as a loss.
- The narrowed `/app` mount is not a security boundary either. `agent_harness.py`
  must stay readable for Python to run it. It removes an easy two-turn `cat` of the
  scoring rules, which is what the first real matches showed models spending their
  turns on.

### Fairness

- **A starting gun.** Containers are created sequentially, so one is ready
  fractionally earlier. Each agent's *first* model call is held until both have
  arrived, then released together. Waiting costs nothing and is never charged to a
  time bank.
- **Side shuffling**, on by default (`--no-shuffle-sides` to disable). Which model
  plays `agent-a` is randomised per match and recorded in `result.json`.

## Match artifacts

Each match writes `matches/<id>/`:

- `result.json` — winner, outcome, rated flag, per-agent token usage **and real
  cost**, inference timings, time banks, exit codes, and the full arena
  configuration (image digest, sampling params, limits) so a result can be audited
  later
- `events.jsonl` — the whole match as one ordered timeline
- `agent-a/agent.jsonl`, `agent-b/agent.jsonl` — every model response and command,
  captured from container stdout
- `proxy/proxy.jsonl` — API request log with per-call timing, usage and cost

The `OPENROUTER_API_KEY` is **never** written here. It lives in a `0600` file inside
a `0700` private temp directory for the lifetime of the match, removed on every exit
path including `SIGTERM`.

## Flags

`orchestrator.py` takes `--mode`, `--model-a/-b`, `--mock`, `--classic`,
and overrides for the mode's own values: `--time-bank`, `--max-rounds`,
`--move-timeout`, `--time-limit`, `--max-steps`, `--max-requests`. An override
that has no meaning in the chosen mode is **rejected**, not ignored — silently
ignored flags are how benchmark configs rot.

Also: `--max-tokens-budget`, `--max-tokens-per-call`, `--temperature`, `--seed`,
`--command-timeout`, `--memory`, `--cpus`, `--pids-limit`, `--battle-size`,
`--grace-seconds`, `--poll-interval`, `--startup-timeout`, `--build`, `--keep`,
`--allow-degraded`, `--no-shuffle-sides`, `--unbounded-fs`, `--no-read-only-fs`,
`--wreck-observe-only`, `--preflight`.

`tournament.py` takes `--models`, `--modes`, `--games`, `--estimate`, `--run`,
`--resume`, `--max-total-tokens`, `--mock`, and the same per-match caps
(`--time-bank`, `--max-rounds`, `--time-limit`, `--max-steps`, `--max-requests`,
`--max-tokens-budget`, `--wreck-observe-only`), each forwarded only to the modes
that accept it.

`elo.py` takes `--matches-dir`, `--k`, `--min-games`, `--mode`, `--compare`,
`--include-degraded`, `--include-mock`, `--include-legacy`, `--quiet`.

## Tests

```bash
python3 -m unittest discover -s tests
```

316 tests covering the scoring rules, the mode table, the lockstep barrier, the time
bank, request validation, credential disclosure, the event stream, the sandbox
properties, real-provider response shapes, and the command runner. No dependencies
and no containers needed.

## Early results

The first real tournament: two models, three modes, both directions, three games
each. 18 matches, 17 rated, 125,578 tokens — about **three cents**.

The standings below are cumulative over all 21 rated matches on disk, which also
include four earlier self-play matches — that is why game counts are uneven
(a model that plays itself is credited with two games in one match).

```
=== untimed (8 rated) ===                     === realtime (6 rated) ===
1  ~deepseek/deepseek-v4-flash-latest  1556   1  ~deepseek/deepseek-v4-flash-latest  1555
2  inclusionai/ling-3.0-flash          1444   2  inclusionai/ling-3.0-flash          1445

=== time-bank (7 rated) ===
1  ~deepseek/deepseek-v4-flash-latest  1543
2  inclusionai/ling-3.0-flash          1457
```

`~deepseek/deepseek-v4-flash-latest` won every mode. Note how wide the 95%
confidence intervals still are at this sample size (`elo.py` prints them): they
overlap, so this is a direction, not a result.

**This does not test the central thesis.** `--compare` returns a Spearman
correlation of 1.0 for every mode pair, but with only two models there are just
two possible orderings, so the statistic is degenerate and carries no
information. Whether the time regime changes who wins needs at least three
models, ideally with a deliberate speed/capability spread.

### What the modes actually do, which is a real finding

Across 22 real matches, outcomes are strongly mode-dependent:

| mode | kills | mutual destruction | round-cap draws |
|---|---|---|---|
| `realtime` | 6 | 0 | 0 |
| `time-bank` | 3 | **4** | 0 |
| `untimed` | 5 | 0 | 3 |

(One further `untimed` match ended in `arena_error` and is excluded — non-contest
outcomes are never rated.)

In lockstep modes both agents commit a kill in the **same round** and execute
simultaneously, so they destroy each other — 57% of time-bank matches ended that
way. In realtime one strikes first and every match was decisive.

The practical consequence: **lockstep modes are noisier than they look.** More
than half of those matches contribute 0.5/0.5 and carry little ranking signal, so
they need more games than realtime to separate two models.

A likely cause was the system prompt, which handed both agents the exact
kill pattern — the shared PID namespace, the exact command line, both heartbeat
URLs. Both models converged on `kill -9 <pid>` within two turns. Worse, the
heartbeat hint pointed at `localhost`, which is wrong in a shared network
namespace and quietly discouraged every genuine network attack. The **warfare**
preset (default since the redesign; `--classic` reproduces the old arena
byte-identically) corrects all three facts:

- **PID-hint removal and a correct network hint** replace the kill plan:
  reconnaissance is now a real skill, and `pkill -f agent_harness.py` no
  longer matches anything.
- **The bulwark** runs the game loop in a respawned child, so the one-shot
  kill only claims a supervisor the parent immediately replaces. The
  opponent's heartbeat stays up on the parent's PID; recovery replays the
  conversation from the durable `/battle/agent.jsonl`.
- **The heartbeat retries its bind** (8 attempts), so `nc -l` port-squatting
  is an upkeep cost rather than a one-turn kill.

Wreck detection has **never once ruled a match**: no real match has ended in
`wrecked`. That is the kind of evidence it needs before being trusted to decide
one — with the caveat that the match record did not, until now, state whether the
detector was even allowed to rule. `result.json` now records
`arena.wreck_observe_only` and `arena.warfare`, so a future "it never fired" is
a claim you can check rather than take on faith.

## Status

Working prototype, verified end to end against real models and producing real
leaderboards. The warfare preset is now the default arena; old match records stay
comparable under `--classic`. Next: a third model with a real speed/capability
gap, which is the only way to get a cross-mode correlation that means anything,
and a public leaderboard page.

## License

MIT — see [LICENSE](LICENSE).

