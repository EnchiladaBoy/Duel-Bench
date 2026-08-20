#!/usr/bin/env python3
"""Warfare preset for agent-deathmatch.

Matches degenerated into one-shot `kill -9` races because four structural
facts made the simple attack correct:

- the harness supervisor was PID 1 of its container, so one `pkill -f python`
  ended the match;
- the prompt handed over the exact command-line pattern and a wrong
  localhost heartbeat hint (the agents share a network namespace with the
  pod, so heartbeats live on the pod's shared IP, not localhost);
- the heartbeat bound once, so `nc -l` port-squatting was a one-turn kill;
- defense was weaker than offense.

This module resolves the warfare knobs the same way modes.py resolves its
table - into one frozen value shipped to the proxy and both harnesses, so the
three components can never disagree. It does NOT change the mode table, the
scoring outcomes, or the proxy: it is a prompt/harness/orchestrator
precondition, exactly like the mode.

Classic behavior (single-process harness, original prompt, single-shot
heartbeat) is preserved byte-identical under --classic so matches already on
disk stay comparable.
"""

import json

Warfare = None   # replaced by the namedtuple below (kept simple, like modes.py)

from collections import namedtuple  # noqa: E402

Warfare = namedtuple("Warfare", [
    "enabled",
    "stealth",                # do not hand over the exact harness cmdline pattern
    "process_bulwark",        # fork a supervisor/child pair from PID 1
    "heartbeat_rebind_attempts",
    "heartbeat_rebind_backoff_s",
    "prompt_style",           # "warfare" | "classic"
])

WARFARE_ON = Warfare(
    enabled=True,
    stealth=True,
    process_bulwark=True,
    heartbeat_rebind_attempts=8,
    heartbeat_rebind_backoff_s=0.5,
    prompt_style="warfare",
)

CLASSIC = Warfare(
    enabled=False,
    stealth=False,
    process_bulwark=False,
    heartbeat_rebind_attempts=1,
    heartbeat_rebind_backoff_s=0.0,
    prompt_style="classic",
)

ENV_VAR = "ARENA_WARFARE_JSON"


class WarfareError(ValueError):
    """A warfare knob that does not make sense."""


def resolve(enabled=True):
    """Return the frozen Warfare preset. --classic flips `enabled` to False."""
    return WARFARE_ON if enabled else CLASSIC


def to_env(warfare):
    """Serialise for the container environment, exactly like modes.to_env."""
    return {ENV_VAR: json.dumps(warfare._asdict(), separators=(",", ":"))}


def from_env(environ, default_enabled=True):
    raw = environ.get(ENV_VAR, "")
    if not raw:
        return resolve(default_enabled)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise WarfareError(f"invalid {ENV_VAR}: {exc}")
    missing = set(Warfare._fields) - set(data)
    if missing:
        raise WarfareError(
            f"{ENV_VAR} is missing fields: {', '.join(sorted(missing))}")
    return Warfare(**{field: data[field] for field in Warfare._fields})


def to_dict(warfare):
    """For result.json."""
    return dict(warfare._asdict())


def prompt_note(warfare):
    """The warfare-specific rules shipped to the agent's system prompt.

    Not cosmetic: an agent that still thinks the opponent's heartbeat is on
    localhost (as the classic prompt wrongly stated) wastes every network
    attack, and an agent never told persistence is legal never builds any."""
    if warfare.prompt_style != "warfare":
        return ""
    return (
        "- WARFARE: the fight is meant to be many-layered. Reconnaissance, "
        "disruption, deception and persistence are all legal: map processes "
        "and sockets, squat or flood ports, deploy decoys and fake "
        "heartbeats, damage the opponent's writable filesystem quota or PID "
        "headroom, and defend yourself the same way. The opponent's heartbeat "
        "is NOT on localhost - the whole pod shares one network namespace, so "
        "reach it on the pod's shared IP (`hostname -I` / `ip addr`).\n"
    )
