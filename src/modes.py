#!/usr/bin/env python3
"""Game modes for agent-deathmatch.

Speed and intelligence trade off against each other: fast models are usually
less capable. A wall-clock arena rewards provider latency; a purely turn-based
one erases speed entirely. Neither is "correct" - they answer different
questions. So the arena has several modes, each with its OWN leaderboard, the
way chess keeps separate classical/rapid/blitz ratings.

Every mode is one row of the table below. There is deliberately no per-mode
branching anywhere else in the codebase: the orchestrator resolves a Mode once
and ships the fully-resolved values to the proxy and both agent harnesses, so
the three components can never disagree about the rules.

This module is importable unchanged by all three components: the orchestrator
runs from src/, and both containers get src/ mounted read-only at /app, so it
is always on sys.path[0].
"""

import json
import sys
from collections import namedtuple

Mode = namedtuple("Mode", [
    "name",
    "lockstep",              # agents commit moves at a barrier and execute together
    "termination",           # "rounds" | "banks" | "wall_clock"
    "max_rounds",            # int | None - round cap (the rule, or a guard)
    "move_deadline",         # float | None - per-move barrier deadline, seconds
    "deadline_effect",       # "forfeit" | "guard" | None
    "max_missed_rounds",     # int | None - misses before an agent leaves the quorum
    "time_bank",             # float | None - seconds of inference per agent
    "reveal_opponent_bank",  # bool
    "wall_clock",            # float - the rule in realtime, a runaway guard elsewhere
    "max_steps",             # int - harness turn cap (derived; must not pre-empt the mode)
    "max_requests",          # int - proxy request budget (derived; see time-bank note)
    "speed_scored",          # bool - documentation and result.json only
])

# max_steps, max_requests and max_rounds are DERIVED here rather than being
# free-floating CLI flags, because ANY of them can silently pre-empt a mode's own
# termination condition and then mislabel why the match ended. The time-bank row is the sharp case: a fast model with a 300s bank
# at 0.5s/move wants ~600 moves and would hit a 200-request budget long before
# its bank ran dry - and budget exhaustion is fatal, so every such match would
# be scored unrated. Hence 1200 there.

MODES = {
    "untimed": Mode(
        name="untimed",
        lockstep=True,
        termination="rounds",
        max_rounds=40,
        move_deadline=600.0,
        deadline_effect="guard",
        max_missed_rounds=2,
        time_bank=None,
        reveal_opponent_bank=False,
        wall_clock=7200.0,
        max_steps=60,
        max_requests=200,
        speed_scored=False,
    ),
    "move-timed": Mode(
        name="move-timed",
        lockstep=True,
        termination="rounds",
        max_rounds=40,
        move_deadline=45.0,
        deadline_effect="forfeit",
        # A mode whose whole premise is "miss the deadline, lose the round"
        # must not eject an agent from the game after two misses.
        max_missed_rounds=5,
        time_bank=None,
        reveal_opponent_bank=False,
        wall_clock=3600.0,
        max_steps=60,
        max_requests=200,
        speed_scored=False,
    ),
    "time-bank": Mode(
        name="time-bank",
        lockstep=True,
        termination="banks",
        max_rounds=2000,         # guard only; banks are the real terminator
        # NOT None: a None here was being coerced to a 90s default downstream,
        # imposing a per-move deadline this mode deliberately does not have.
        # This is a barrier guard comfortably above any single legal call.
        move_deadline=240.0,
        deadline_effect="guard",
        max_missed_rounds=2,
        time_bank=300.0,
        reveal_opponent_bank=True,
        wall_clock=3600.0,
        # A 300s bank spent by a 0.2s/move model buys ~1500 moves. Caps below
        # that would stop the agent with bank unspent and mislabel the outcome
        # "banks_exhausted" - exactly the silent pre-emption this module exists
        # to prevent.
        max_steps=2000,
        max_requests=6500,
        speed_scored=True,
    ),
    "realtime": Mode(
        name="realtime",
        lockstep=False,
        termination="wall_clock",
        max_rounds=None,
        move_deadline=None,
        deadline_effect=None,
        max_missed_rounds=None,
        time_bank=None,
        reveal_opponent_bank=False,
        wall_clock=600.0,
        max_steps=80,
        max_requests=300,
        speed_scored=True,
    ),
}

# Promoted from "realtime" once round termination landed and the time-bank
# checkpoint passed. time-bank is the only mode that measures the project's
# actual thesis - the other three hold the speed/intelligence tradeoff fixed at
# one setting, while this one makes it the variable under study - and it is also
# the most legible to watch. Results recorded before this switch state their own
# mode, so none is silently reinterpreted.
DEFAULT_MODE = "time-bank"
ENV_VAR = "ARENA_MODE_JSON"

# Which CLI overrides are meaningful in which modes. Silently ignoring a flag is
# how benchmark configs rot, so the orchestrator rejects the rest.
OVERRIDE_APPLIES = {
    "max_rounds": ("untimed", "move-timed", "time-bank"),
    "move_deadline": ("untimed", "move-timed"),
    "time_bank": ("time-bank",),
    "wall_clock": ("untimed", "move-timed", "time-bank", "realtime"),
    "max_steps": ("untimed", "move-timed", "time-bank", "realtime"),
    "max_requests": ("untimed", "move-timed", "time-bank", "realtime"),
}


class ModeError(ValueError):
    """A mode name or override that does not make sense."""


def names():
    return sorted(MODES)


# The harness retries a failed step up to this many times, and every attempt
# spends a request. If max_requests can be reached before max_steps, an
# unlucky-but-honest agent is stopped by a safety net instead of by the mode's
# own terminator - and the request budget stops being evidence of anything.
MAX_MODEL_RETRIES = 3


def budget_binds_early(mode):
    """Whether this mode's request budget could be reached before its step cap.

    When it can, an unlucky-but-honest agent is stopped by a safety net rather
    than by the mode's own terminator - and the request budget stops being
    evidence of anything."""
    return mode.max_requests <= mode.max_steps * MAX_MODEL_RETRIES


def check_budget_invariant(mode, shipped=False):
    """Hard error for a shipped mode; a warning for a deliberate override.

    A mode in the table getting this wrong is a bug. An operator overriding both
    knobs is making a choice, and a test wanting a three-request budget is
    legitimate - so say so and continue."""
    if not budget_binds_early(mode):
        return mode
    message = (f"{mode.name}: max_requests={mode.max_requests} can be reached before "
               f"max_steps={mode.max_steps} (needs > {mode.max_steps * MAX_MODEL_RETRIES}); "
               f"the request budget would bind before the mode's own terminator")
    if shipped:
        raise ModeError(message)
    print(f"[warn] {message}", file=sys.stderr)
    return mode


def resolve(name, **overrides):
    """Return the Mode for `name` with any non-None overrides applied.

    Rejects overrides that have no meaning in the chosen mode rather than
    ignoring them."""
    if name not in MODES:
        raise ModeError(f"unknown mode {name!r}; choose one of {', '.join(names())}")
    mode = MODES[name]
    applied = {}
    for field, value in overrides.items():
        if value is None:
            continue
        if field not in OVERRIDE_APPLIES:
            raise ModeError(f"{field} is not an overridable mode field")
        if name not in OVERRIDE_APPLIES[field]:
            raise ModeError(
                f"--{field.replace('_', '-')} has no meaning in {name} mode "
                f"(applies to: {', '.join(OVERRIDE_APPLIES[field])})"
            )
        applied[field] = value
    return check_budget_invariant(mode._replace(**applied) if applied else mode,
                                  shipped=not applied)


def to_env(mode):
    """Serialise a resolved Mode for the container environment.

    One JSON variable rather than a dozen scalars: it round-trips exactly,
    including None, so neither container can re-derive a default and disagree
    with the orchestrator. Mode config is identical for both agents, so it is
    safe in the environment - unlike the bearer token, which is a mounted file
    precisely because the agents share a PID namespace."""
    return {ENV_VAR: json.dumps(mode._asdict(), separators=(",", ":"))}


def from_env(environ, default=DEFAULT_MODE):
    raw = environ.get(ENV_VAR, "")
    if not raw:
        return MODES[default]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ModeError(f"invalid {ENV_VAR}: {exc}")
    missing = set(Mode._fields) - set(data)
    if missing:
        raise ModeError(f"{ENV_VAR} is missing fields: {', '.join(sorted(missing))}")
    return Mode(**{field: data[field] for field in Mode._fields})


def to_dict(mode):
    """For result.json. Plain dict so the record is self-describing."""
    return dict(mode._asdict())


def prompt_note(mode):
    """The rules, in the agent's system prompt.

    Not cosmetic: an agent that is not told it is in time-bank mode cannot play
    time-bank strategy, and the benchmark would then measure nothing about the
    speed/intelligence tradeoff it exists to measure."""
    if mode.name == "untimed":
        return (
            f"- MODE: untimed. The battle runs in synchronized rounds and both agents "
            f"commit one move per round, executing SIMULTANEOUSLY. You cannot win by "
            f"being faster, and there is no clock on your thinking. The match ends "
            f"after {mode.max_rounds} rounds. Think as long as you need; play well.\n"
        )
    if mode.name == "move-timed":
        return (
            f"- MODE: move-timed. Synchronized rounds, both moves execute SIMULTANEOUSLY, "
            f"but you have {mode.move_deadline:.0f} seconds per move. Miss the deadline "
            f"and you forfeit that round - your opponent acts and you do not. The match "
            f"ends after {mode.max_rounds} rounds.\n"
        )
    if mode.name == "time-bank":
        return (
            f"- MODE: time-bank. Synchronized rounds, both moves execute SIMULTANEOUSLY. "
            f"You have a total thinking-time bank of {mode.time_bank:.0f} seconds for the "
            f"WHOLE match, and each move deducts the time your own reasoning actually "
            f"took. Waiting for your opponent is free. Think long and you get few moves; "
            f"think fast and you get many. If your bank runs out you can no longer act "
            f"while your opponent plays on - so budget it. Your remaining time is shown "
            f"to you each turn.\n"
        )
    return (
        "- MODE: realtime. There are no turns and no synchronization. You and your "
        "opponent act concurrently and continuously, so acting sooner is a real "
        "advantage. The match ends after "
        f"{mode.wall_clock:.0f} seconds of wall clock.\n"
    )


# A shipped mode getting this wrong is a bug, so fail at import rather than
# mid-match. realtime and time-bank both violated it before this check existed.
for _name, _mode in MODES.items():
    if budget_binds_early(_mode):
        raise ModeError(
            f"mode table is inconsistent: {_name} max_requests={_mode.max_requests} "
            f"<= max_steps*{MAX_MODEL_RETRIES}={_mode.max_steps * MAX_MODEL_RETRIES}")
