#!/usr/bin/env python3
"""Watch a match, live or as a replay.

    python3 src/watch.py                      # follow the most recent match
    python3 src/watch.py matches/<id>         # follow a specific one
    python3 src/watch.py <id> --replay        # replay a finished match
    python3 src/watch.py <id> --replay --speed 4

Reads matches/<id>/events.jsonl, which the orchestrator writes while the match
runs. Stdlib only, no dependencies, and read-only: the viewer never touches the
arena. Degrades to a plain scrolling feed when stdout is not a terminal, so it
pipes cleanly.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MATCHES = ROOT / "matches"

RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
RED, GREEN, YELLOW, BLUE, CYAN, GREY = (
    "\033[31m", "\033[32m", "\033[33m", "\033[34m", "\033[36m", "\033[90m")
CLEAR, HOME = "\033[2J", "\033[H"
CLEAR_LINE = "\033[K"

ROLE_COLOUR = {"agent-a": CYAN, "agent-b": YELLOW}
FEED_LINES = 14


def latest_match():
    candidates = [d for d in MATCHES.glob("*/") if (d / "events.jsonl").exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.stat().st_mtime)


def resolve(target):
    if target is None:
        return latest_match()
    path = Path(target)
    if path.is_dir():
        return path
    guess = MATCHES / target
    return guess if guess.is_dir() else None


class MatchState:
    """Everything a viewer needs, rebuilt from the event stream."""

    def __init__(self):
        self.mode = "?"
        self.models = {}
        self.bank_granted = None
        self.round = None
        self.elapsed = 0.0
        self.agents = {}
        self.feed = []
        self.finished = None
        self.rated = None
        self.attacks = []          # [(t, src, dst, kind)]
        self.terrain = {}          # {role: {"score": int, "spoofs": int}}
        self.steps = {}            # {role: [(step, command, cls)]}

    def agent(self, role):
        return self.agents.setdefault(role, {
            "alive": True, "steps": 0, "commands": 0, "passes": 0,
            "forfeits": 0, "bank": None, "thinking": None, "last": "",
            "stop_reason": None, "health": None,
        })

    def note(self, text, cls="recon"):
        # Carry the match time with the entry. Storing bare strings meant a
        # viewer that joined late - or replayed - showed every line at 0.0s,
        # because the timestamp only existed in the renderer's local variable.
        self.feed.append((round(self.elapsed, 1), text, cls))
        del self.feed[:-200]

    @staticmethod
    def classify_command(command):
        """Light heuristic for coloring the feed."""
        c = command.lower()
        if "kill" in c or "pkill" in c or "nc " in c or "sleep" in c or "flood" in c:
            return "attack"
        if "curl" in c and "POST" in c.upper():
            return "attack"
        if "curl" in c and ("/debug" in c or "/telemetry" in c):
            return "defense"
        if "respawn" in c or "while true" in c:
            return "defense"
        return "recon"

    # The harness prints its log to container stdout and the orchestrator
    # ingests every line, so an agent's shell can `echo` a record onto its own
    # stdout. The merger stamps `src` itself, so an agent cannot impersonate its
    # opponent - but it CAN emit any payload under its own name. These events
    # decide the match, so only the component entitled to produce them is
    # believed. Without this an agent ends the spectator's match with one echo.
    ARENA_ONLY = {
        "orchestrator": ("match_start", "match_end", "agent_down", "snapshot",
                         "arena_ready", "egress_check"),
        "proxy": ("go", "move_start", "thinking", "completion", "bank_exhausted",
                  "move_forfeit", "barrier_release", "proxy_start"),
    }

    @classmethod
    def trusted(cls, kind, src):
        for owner, kinds in cls.ARENA_ONLY.items():
            if kind in kinds:
                return src == owner
        return True

    def apply(self, event):
        kind, src = event.get("event"), event.get("src")
        self.elapsed = max(self.elapsed, event.get("t") or 0.0)
        if not self.trusted(kind, src):
            return
        role = event.get("agent") or (src if src.startswith("agent-") else None)

        if kind == "match_start":
            self.mode = event.get("mode", "?")
            self.models = {"agent-a": event.get("model_a"), "agent-b": event.get("model_b")}
            self.bank_granted = (event.get("mode_config") or {}).get("time_bank")
            self.note(f"{DIM}match starts — {event.get('model_a')} vs "
                      f"{event.get('model_b')}{RESET}")
        elif kind == "go":
            self.note(f"{BOLD}{GREEN}FIGHT{RESET}")
        elif kind == "move_start" and role:
            agent = self.agent(role)
            agent["thinking"] = 0.0
            if event.get("bank_remaining") is not None:
                agent["bank"] = event["bank_remaining"]
            if event.get("round"):
                self.round = event["round"]
        elif kind == "thinking" and role:
            agent = self.agent(role)
            agent["thinking"] = event.get("elapsed")
            if event.get("bank_remaining") is not None:
                agent["bank"] = event["bank_remaining"]
        elif kind == "completion" and role:
            agent = self.agent(role)
            agent["thinking"] = None
            if event.get("bank_remaining") is not None:
                agent["bank"] = event["bank_remaining"]
        elif kind == "command_start" and role:
            agent = self.agent(role)
            agent["commands"] += 1
            agent["steps"] = event.get("step") or agent["steps"] + 1
            agent["last"] = (event.get("command") or "").strip()
            cls = self.classify_command(agent["last"])
            self.steps.setdefault(role, []).append(
                (agent["steps"], agent["last"], cls))
            del self.steps[role][-5:]
            self.note(f"{ROLE_COLOUR.get(role, '')}{role}{RESET} "
                      f"{BOLD}${RESET} {agent['last']}", cls)
        elif kind == "command_result" and role:
            code = event.get("exit_code")
            if event.get("timed_out"):
                self.note(f"{GREY}    {role}: timed out{RESET}", "error")
            elif code not in (0, None):
                self.note(f"{GREY}    {role}: exit {code}{RESET}", "error")
        elif kind == "terrain_defended" and role:
            self.terrain.setdefault(role, {"score": 0, "spoofs": 0})
            self.terrain[role]["score"] = event.get("score", 0)
            self.note(f"{ROLE_COLOUR.get(role, '')}{role}{RESET} "
                      f"{BOLD}defends{RESET} (score {event.get('score')})", "defense")
        elif kind in ("terrain_signal_hijacked", "terrain_telemetry_flooded") and role:
            self.terrain.setdefault(role, {"score": 0, "spoofs": 0})
            self.terrain[role]["spoofs"] += 1
            self.attacks.append((round(self.elapsed, 1), role, None, kind))
            del self.attacks[-20:]
            self.note(f"{RED}{role}{RESET} {BOLD}{kind}{RESET}", "attack")
        elif kind == "terrain_hit" and role:
            self.note(f"{GREY}{role}: terrain hit ({event.get('endpoint')}){RESET}",
                      "terrain")
        elif kind == "pass" and role:
            self.agent(role)["passes"] += 1
            self.note(f"{GREY}{role} passes{RESET}")
        elif kind == "move_forfeit" and role:
            self.agent(role)["forfeits"] += 1
            self.note(f"{RED}{role} forfeits the round (too slow){RESET}")
        elif kind == "bank_exhausted" and role:
            self.agent(role)["bank"] = 0.0
            self.note(f"{BOLD}{RED}{role} is out of time{RESET}")
        elif kind == "idle" and role:
            self.agent(role)["stop_reason"] = event.get("reason")
            self.note(f"{GREY}{role} stops: {event.get('reason')}{RESET}")
        elif kind == "agent_down":
            downed = event.get("agent")
            if downed:
                self.agent(downed)["alive"] = False
                self.note(f"{BOLD}{RED}{downed} is down ({event.get('how')}){RESET}")
        elif kind == "snapshot":
            self.round = event.get("round") or self.round
            for name, info in (event.get("agents") or {}).items():
                agent = self.agent(name)
                agent["alive"] = info.get("alive", agent["alive"])
                if info.get("commands_run") is not None:
                    agent["commands"] = max(agent["commands"], info["commands_run"])
                agent["stop_reason"] = info.get("stop_reason") or agent["stop_reason"]
                tc = info.get("terrain")
                if isinstance(tc, dict):
                    self.terrain.setdefault(name, {"score": 0, "spoofs": 0})
                    self.terrain[name]["score"] = tc.get("score", 0)
                    self.terrain[name]["spoofs"] = tc.get("spoofs", 0)
                if isinstance(info.get("health"), dict):
                    agent["health"] = info["health"]
            for name, left in (event.get("banks") or {}).items():
                if left is not None:
                    self.agent(name)["bank"] = left
        elif kind == "match_end":
            self.finished = event
            self.rated = event.get("rated")
            winner, outcome = event.get("winner"), event.get("outcome")
            colour = GREEN if winner in ("agent-a", "agent-b") else YELLOW
            self.note(f"{BOLD}{colour}{outcome}: {winner}{RESET}")


def bar(fraction, width=22):
    filled = max(0, min(width, int(round(fraction * width))))
    colour = GREEN if fraction > 0.5 else (YELLOW if fraction > 0.2 else RED)
    return f"{colour}{'█' * filled}{GREY}{'░' * (width - filled)}{RESET}"


def render(state):
    width = shutil.get_terminal_size((100, 30)).columns
    out = [HOME]
    title = f" Duel-Bench  {BOLD}{state.mode}{RESET}"
    if state.round:
        title += f"  {GREY}round{RESET} {state.round}"
    title += f"  {GREY}{state.elapsed:.0f}s{RESET}"
    out.append(title + CLEAR_LINE)
    out.append(GREY + "─" * min(width, 96) + RESET + CLEAR_LINE)

    for role in ("agent-a", "agent-b"):
        agent = state.agents.get(role)
        if not agent:
            continue
        colour = ROLE_COLOUR.get(role, "")
        model = (state.models.get(role) or "?")[:28]
        status = f"{GREEN}alive{RESET}" if agent["alive"] else f"{RED}DOWN {RESET}"
        line = f" {colour}{role}{RESET} {model:<28} {status}"
        if agent["bank"] is not None and state.bank_granted:
            line += f"  {bar(agent['bank'] / state.bank_granted)} {agent['bank']:>6.1f}s"
        line += f"  {GREY}{agent['commands']:>3} cmds{RESET}"
        if agent["forfeits"]:
            line += f" {RED}{agent['forfeits']} ff{RESET}"
        if agent["passes"]:
            line += f" {GREY}{agent['passes']} pass{RESET}"
        if agent["thinking"] is not None:
            line += f"  {BLUE}thinking {agent['thinking']:.0f}s…{RESET}"
        out.append(line + CLEAR_LINE)
        if agent["last"]:
            out.append(f"   {GREY}$ {agent['last'][:width - 8]}{RESET}" + CLEAR_LINE)

    out.append(GREY + "─" * min(width, 96) + RESET + CLEAR_LINE)
    for when, line, cls in state.feed[-FEED_LINES:]:
        stamp = f"{GREY}{when:>6.1f}s{RESET} "
        out.append(" " + stamp + line[:width + 30] + CLEAR_LINE)
    out.extend([CLEAR_LINE] * max(0, FEED_LINES - len(state.feed[-FEED_LINES:])))
    sys.stdout.write("\n".join(out))
    sys.stdout.flush()


def follow(path, state, interactive):
    """Tail the stream as the match runs."""
    pos, idle_since = 0, None
    while True:
        fresh = False
        if path.exists():
            with path.open("r", errors="replace") as fh:
                fh.seek(pos)
                while True:
                    # readline, not iteration: tell() is disabled inside a
                    # for-loop over a text file.
                    line = fh.readline()
                    if not line or not line.endswith("\n"):
                        break
                    pos = fh.tell()
                    try:
                        state.apply(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                    fresh = True
        if interactive:
            render(state)
        elif fresh:
            for when, line, cls in state.feed[-FEED_LINES:]:
                print(f"{when:>7.1f}s {line}")
            state.feed.clear()
        if state.finished:
            return
        idle_since = idle_since or time.time()
        if fresh:
            idle_since = None
        elif time.time() - idle_since > 900:
            return           # the match is gone and nothing more is coming
        time.sleep(0.25)


def replay(path, state, interactive, speed):
    events = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    previous = 0.0
    for event in events:
        gap = (event.get("t") or 0.0) - previous
        previous = event.get("t") or previous
        if gap > 0 and speed > 0:
            time.sleep(min(gap / speed, 2.0))
        state.apply(event)
        if interactive:
            render(state)
        else:
            for when, line, cls in state.feed[-1:]:
                print(f"{when:>7.1f}s {line}")
            state.feed.clear()


def main():
    parser = argparse.ArgumentParser(description="watch a Duel-Bench match")
    parser.add_argument("match", nargs="?", default=None,
                        help="match directory or id (default: the most recent)")
    parser.add_argument("--replay", action="store_true", help="replay a finished match")
    parser.add_argument("--speed", type=float, default=1.0, help="replay speed multiplier")
    args = parser.parse_args()

    match_dir = resolve(args.match)
    if match_dir is None:
        sys.exit("no match found; pass a directory, or run a match first")
    stream = match_dir / "events.jsonl"
    if not stream.exists():
        sys.exit(f"{stream} does not exist yet")

    interactive = sys.stdout.isatty()
    state = MatchState()
    if interactive:
        sys.stdout.write(CLEAR)
    try:
        if args.replay:
            replay(stream, state, interactive, args.speed)
        else:
            follow(stream, state, interactive)
    except KeyboardInterrupt:
        pass
    finally:
        if interactive:
            sys.stdout.write("\n" * 2)
            sys.stdout.flush()

    if state.finished:
        end = state.finished
        flag = "" if end.get("rated") else "  (unrated: %s)" % end.get("unrated_reason")
        print(f"\n{BOLD}{end.get('outcome')}{RESET} — winner: "
              f"{end.get('winner')}  in {end.get('duration')}s{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
