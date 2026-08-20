#!/usr/bin/env python3
"""Run a tournament: every pair, both ways, across one or more modes.

A single match carries a side bias (agent-a's container is created first) and a
single result carries no statistical weight. A leaderboard worth citing needs
each pair played in BOTH directions, enough times, in each mode it is rated in.

    python3 src/tournament.py --models a,b,c --modes untimed,realtime --games 3
    python3 src/tournament.py --models a,b --estimate          # spend nothing
    python3 src/tournament.py --resume tournaments/<id>        # pick up where it stopped

Runs are resumable: the full schedule is written up front, and a re-run skips
matches that already finished. Long runs against real models must survive a
transient provider outage without discarding everything already paid for.
"""

import argparse
import itertools
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORCHESTRATOR = ROOT / "src" / "orchestrator.py"
TOURNAMENTS = ROOT / "tournaments"
MATCHES = ROOT / "matches"

# Used only by --estimate when no completed matches exist to learn from.
ASSUMED_TOKENS_PER_MATCH = 60000
LOGS_RE = re.compile(r"^\[result\] logs: (.+)$", re.M)


def schedule(models, modes, games):
    """Every unordered pair, played `games` times in BOTH directions, per mode.

    Both directions is the point: without it every rating carries the first-mover
    advantage of whichever model happened to be passed as --model-a."""
    plan = []
    for mode in modes:
        for left, right in itertools.combinations(sorted(set(models)), 2):
            for repeat in range(games):
                for model_a, model_b in ((left, right), (right, left)):
                    plan.append({
                        "id": f"{mode}:{model_a}:vs:{model_b}:{repeat}",
                        "mode": mode,
                        "model_a": model_a,
                        "model_b": model_b,
                        "repeat": repeat,
                        "status": "pending",
                        "attempts": 0,
                        "match_id": None,
                        "winner": None,
                        "rated": None,
                        "total_tokens": 0,
                    })
    return plan


def observed_tokens_per_match(matches_dir=MATCHES):
    """Learn the per-match cost from finished matches rather than guessing."""
    totals, count = 0, 0
    for path in Path(matches_dir).glob("*/result.json"):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("mock"):
            continue
        spent = sum((u or {}).get("total_tokens", 0)
                    for u in (data.get("usage") or {}).values())
        if spent:
            totals += spent
            count += 1
    return (totals // count) if count else None


def estimate(plan, matches_dir=MATCHES):
    per_match = observed_tokens_per_match(matches_dir)
    source = "observed from finished matches" if per_match else "assumed (no data yet)"
    per_match = per_match or ASSUMED_TOKENS_PER_MATCH
    modes = sorted({m["mode"] for m in plan})
    print(f"{len(plan)} matches across {len(modes)} mode(s): {', '.join(modes)}")
    print(f"~{per_match:,} tokens per match ({source})")
    print(f"~{per_match * len(plan):,} tokens total")
    print("\nNo API calls were made. Add --run to execute, and consider "
          "--max-total-tokens to bound the spend.")


def run_match(entry, args):
    """Invoke the orchestrator for one scheduled match."""
    cmd = [
        sys.executable, str(ORCHESTRATOR),
        "--mode", entry["mode"],
        "--model-a", entry["model_a"],
        "--model-b", entry["model_b"],
        # The tournament assigns sides deliberately, one pairing each way. If
        # the orchestrator re-randomised them the balance would be destroyed.
        "--no-shuffle-sides",
    ]
    if args.mock:
        cmd.append("--mock")
    for flag, value in (("--time-bank", args.time_bank),
                        ("--max-rounds", args.max_rounds),
                        ("--time-limit", args.time_limit)):
        if value is not None:
            cmd += [flag, str(value)]
    cmd += args.passthrough

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.match_timeout)
    found = LOGS_RE.search(proc.stdout or "")
    if not found:
        return None, (proc.stderr or proc.stdout or "").strip()[-500:]
    match_dir = Path(found.group(1).strip())
    try:
        return json.loads((match_dir / "result.json").read_text()), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"unreadable result.json: {exc}"


def save(manifest_path, manifest):
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def run(manifest_path, manifest, args):
    plan = manifest["plan"]
    spent = sum(m["total_tokens"] for m in plan)
    done = sum(1 for m in plan if m["status"] in ("done", "failed"))

    for entry in plan:
        if entry["status"] == "done":
            continue
        if entry["status"] == "failed" and entry["attempts"] > 1:
            continue
        if args.max_total_tokens and spent >= args.max_total_tokens:
            print(f"\n[budget] stopping: {spent:,} tokens spent, cap is "
                  f"{args.max_total_tokens:,}. Re-run with --resume to continue.")
            break

        done += 1
        entry["attempts"] += 1
        label = f"{entry['mode']}  {entry['model_a']} vs {entry['model_b']}"
        print(f"[{done}/{len(plan)}] {label} ...", flush=True)

        result, error = run_match(entry, args)
        if result is None:
            entry["status"] = "failed"
            entry["error"] = error
            # One retry: a transient provider outage should not thin a board.
            if entry["attempts"] <= 1:
                entry["status"] = "pending"
            print(f"          FAILED ({'will retry' if entry['attempts'] <= 1 else 'giving up'}): {error}")
        else:
            tokens = sum((u or {}).get("total_tokens", 0)
                         for u in (result.get("usage") or {}).values())
            entry.update({
                "status": "done", "match_id": result.get("match_id"),
                "winner": result.get("winner"), "rated": result.get("rated"),
                "outcome": result.get("outcome"), "total_tokens": tokens,
            })
            spent += tokens
            flag = "" if result.get("rated") else f"  UNRATED ({result.get('unrated_reason')})"
            print(f"          {result.get('outcome')} -> {result.get('winner')}{flag}")
        manifest["spent_tokens"] = spent
        save(manifest_path, manifest)

    return spent


def summarize(manifest):
    plan = manifest["plan"]
    by_status = {}
    for entry in plan:
        by_status[entry["status"]] = by_status.get(entry["status"], 0) + 1
    rated = sum(1 for e in plan if e.get("rated"))
    print(f"\n{len(plan)} scheduled: " +
          ", ".join(f"{n} {s}" for s, n in sorted(by_status.items())))
    print(f"{rated} rated, {manifest.get('spent_tokens', 0):,} tokens spent")
    unrated = [e for e in plan if e["status"] == "done" and not e.get("rated")]
    if unrated:
        # Never let discarded matches be invisible: a thinned board looks the
        # same as a complete one unless the losses are stated.
        print(f"{len(unrated)} completed match(es) were NOT rated:")
        for entry in unrated[:10]:
            print(f"  {entry['mode']}: {entry['model_a']} vs {entry['model_b']}"
                  f" - {entry.get('outcome')}")


def parse_args():
    p = argparse.ArgumentParser(description="agent-deathmatch tournament runner")
    p.add_argument("--models", default="", help="comma-separated model ids")
    p.add_argument("--modes", default="time-bank", help="comma-separated modes")
    p.add_argument("--games", type=int, default=3,
                   help="repeats per pair PER DIRECTION (so 3 means 6 matches per pair)")
    p.add_argument("--estimate", action="store_true",
                   help="print the schedule and projected spend, then exit")
    p.add_argument("--run", action="store_true", help="actually execute the schedule")
    p.add_argument("--resume", default=None, help="continue an existing tournament directory")
    p.add_argument("--max-total-tokens", type=int, default=0,
                   help="stop once this many tokens have been spent (0 = no cap)")
    p.add_argument("--match-timeout", type=int, default=3600)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--time-bank", type=float, default=None)
    p.add_argument("--max-rounds", type=int, default=None)
    p.add_argument("--time-limit", type=int, default=None)
    p.add_argument("passthrough", nargs="*", default=[],
                   help="extra flags forwarded to the orchestrator")
    return p.parse_args()


def main():
    args = parse_args()

    if args.resume:
        manifest_path = Path(args.resume) / "manifest.json"
        if not manifest_path.exists():
            sys.exit(f"no manifest at {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        print(f"Resuming {manifest['id']}: "
              f"{sum(1 for m in manifest['plan'] if m['status'] == 'done')}"
              f"/{len(manifest['plan'])} already done")
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        if len(models) < 2:
            sys.exit("need at least two --models")
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
        plan = schedule(models, modes, args.games)
        if args.estimate:
            estimate(plan)
            return 0
        started = datetime.now(timezone.utc)
        tournament_id = started.strftime("%Y%m%d-%H%M%SZ")
        directory = TOURNAMENTS / tournament_id
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {"id": tournament_id, "started_at_utc": started.isoformat(),
                    "models": models, "modes": modes, "games": args.games,
                    "spent_tokens": 0, "plan": plan}
        manifest_path = directory / "manifest.json"
        save(manifest_path, manifest)
        print(f"Scheduled {len(plan)} matches -> {manifest_path}")

    if not (args.run or args.resume):
        print("Nothing executed. Add --run to start, or --estimate to price it first.")
        return 0

    run(manifest_path, manifest, args)
    summarize(manifest)
    print(f"\nLeaderboards:  python3 src/elo.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
