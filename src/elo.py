#!/usr/bin/env python3
"""ELO leaderboard over matches/*/result.json.

Only matches that were actually contested in an intact arena are rated. A
result is skipped when it was a --mock run, when the arena was silently
degraded (no shared PID namespace, or no internal network), or when the
orchestrator marked it unrated because neither model ever executed a command.

Usage:
    python3 src/elo.py [--matches-dir matches] [--k 32] [--min-games 5]
                       [--include-degraded] [--include-mock] [--quiet]
"""

import argparse
import json
import random
from pathlib import Path

INITIAL_RATING = 1500.0
BOOTSTRAP_SAMPLES = 2000


def expected_score(ra, rb):
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


LEGACY_POOL = "legacy"


def pool_of(data):
    """Which leaderboard a result belongs on. Modes are never pooled: they
    impose different rules, so their ratings are not comparable."""
    return data.get("mode") or LEGACY_POOL


def partition(results):
    """Group ordered results by mode. rate() and bootstrap_interval() are pure
    functions of a result list, so each pool is simply rated independently."""
    pools = {}
    for data in results:
        pools.setdefault(pool_of(data), []).append(data)
    return pools


def eligibility(data, include_degraded=False, include_mock=False, include_legacy=False):
    """Return None if the match is ratable, else a reason string."""
    if not data.get("model_a") or not data.get("model_b"):
        return "missing model ids"
    if data.get("winner") not in ("agent-a", "agent-b", "draw"):
        return f"non-contest outcome ({data.get('winner')})"
    if data.get("mock") and not include_mock:
        return "mock match (scripted agents, no model calls)"
    if data.get("rated") is False:
        return f"orchestrator marked unrated ({data.get('unrated_reason', 'no reason given')})"
    if not data.get("mode") and not include_legacy:
        # Unlike a missing arena field ("not recorded, probably fine"), a missing
        # mode means we cannot know which leaderboard this belongs on, and
        # guessing would recreate the pooling this partitioning exists to stop.
        return "no mode recorded (pre-mode-system result); use --include-legacy"
    if not include_degraded:
        # Absent fields mean "written by a build that did not record this";
        # only an explicit False is treated as a degraded arena.
        if data.get("pid_shared") is False:
            return "degraded arena (no shared PID namespace: agents could not reach each other)"
        if data.get("network_internal") is False:
            return "degraded arena (agents had unrestricted internet egress)"
    return None


def sort_key(item):
    """Order matches chronologically. ELO is order-dependent, so this must not
    depend on filesystem layout: prefer the recorded UTC timestamp, fall back to
    the naive local timestamp, and only then to the path."""
    path, data = item
    return (
        str(data.get("started_at_utc") or ""),
        str(data.get("date") or ""),
        str(path),
    )


def load_results(matches_dir, include_degraded=False, include_mock=False,
                 include_legacy=False):
    rated, skipped = [], []
    for path in sorted(Path(matches_dir).glob("*/result.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            skipped.append((path, f"unreadable: {exc}"))
            continue
        reason = eligibility(data, include_degraded, include_mock, include_legacy)
        if reason:
            skipped.append((path, reason))
        else:
            rated.append((path, data))
    rated.sort(key=sort_key)
    return [data for _, data in rated], skipped


def rate(results, k):
    """Run ELO over an ordered list of results. Both updates are computed from
    the pre-update ratings, so a self-play match (model_a == model_b) nets to
    zero instead of the winner's update being overwritten by the loser's."""
    ratings, games = {}, {}
    for data in results:
        ma, mb = data["model_a"], data["model_b"]
        winner = data["winner"]
        score_a = 1.0 if winner == "agent-a" else (0.0 if winner == "agent-b" else 0.5)

        ra = ratings.setdefault(ma, INITIAL_RATING)
        rb = ratings.setdefault(mb, INITIAL_RATING)
        ea = expected_score(ra, rb)
        delta_a = k * (score_a - ea)
        delta_b = k * ((1.0 - score_a) - (1.0 - ea))

        ratings[ma] = ratings[ma] + delta_a
        ratings[mb] = ratings[mb] + delta_b
        games[ma] = games.get(ma, 0) + 1
        games[mb] = games.get(mb, 0) + 1
    return ratings, games


def bootstrap_interval(results, k, samples=BOOTSTRAP_SAMPLES, seed=0):
    """Resample matches with replacement to get a 95% interval per model. With
    a handful of matches this interval is very wide, which is the honest
    answer rather than a bug."""
    if len(results) < 2:
        return {}
    rng = random.Random(seed)
    draws = {}
    n = len(results)
    for _ in range(samples):
        sample = [results[rng.randrange(n)] for _ in range(n)]
        sample_ratings, _ = rate(sample, k)
        for model, rating in sample_ratings.items():
            draws.setdefault(model, []).append(rating)
    intervals = {}
    for model, values in draws.items():
        values.sort()
        lo = values[int(0.025 * (len(values) - 1))]
        hi = values[int(0.975 * (len(values) - 1))]
        intervals[model] = (lo, hi)
    return intervals


def usage_per_model(results):
    """Mean total tokens per match, reported beside ELO but never scored."""
    totals, counts = {}, {}
    for data in results:
        usage = data.get("usage") or {}
        for role, model in (("agent-a", data.get("model_a")),
                            ("agent-b", data.get("model_b"))):
            spent = (usage.get(role) or {}).get("total_tokens")
            if not model or not spent:
                continue
            totals[model] = totals.get(model, 0) + spent
            counts[model] = counts.get(model, 0) + 1
    return {m: totals[m] / counts[m] for m in totals if counts.get(m)}


def main():
    parser = argparse.ArgumentParser(description="agent-deathmatch ELO leaderboard")
    parser.add_argument("--matches-dir", default="matches")
    parser.add_argument("--k", type=float, default=32.0)
    parser.add_argument("--min-games", type=int, default=5,
                        help="ratings below this many games are marked provisional")
    parser.add_argument("--include-degraded", action="store_true",
                        help="rate matches whose arena was silently degraded (not recommended)")
    parser.add_argument("--include-mock", action="store_true",
                        help="rate --mock pipeline tests as if they were real matches")
    parser.add_argument("--include-legacy", action="store_true",
                        help="rate results written before modes existed, in a 'legacy' pool")
    parser.add_argument("--mode", default=None,
                        help="show only this mode's leaderboard")
    parser.add_argument("--quiet", action="store_true",
                        help="do not list skipped matches")
    args = parser.parse_args()

    results, skipped = load_results(
        args.matches_dir, args.include_degraded, args.include_mock, args.include_legacy
    )

    if skipped and not args.quiet:
        print(f"Skipped {len(skipped)} match(es) as unratable:")
        for path, reason in skipped:
            print(f"  {path.parent.name}: {reason}")
        print()

    if not results:
        print(f"No ratable matches found in {args.matches_dir}/")
        if skipped:
            print("Every match on disk was skipped for the reasons above.")
        return 0

    pools = partition(results)
    if args.mode:
        pools = {k: v for k, v in pools.items() if k == args.mode}
        if not pools:
            print(f"No rated matches in mode {args.mode!r}")
            return 0

    for pool_name in sorted(pools):
        pool = pools[pool_name]
        ratings, games = rate(pool, args.k)
        intervals = bootstrap_interval(pool, args.k)
        usage = usage_per_model(pool)

        print(f"\n=== {pool_name} ({len(pool)} rated matches, K={args.k}) ===")
        print(f"{'MODEL':<40} {'ELO':>7} {'95% CI':>17} {'GAMES':>6} {'TOK/MATCH':>10}")
        print("-" * 84)
        for model, rating in sorted(ratings.items(), key=lambda kv: -kv[1]):
            lo, hi = intervals.get(model, (float("nan"), float("nan")))
            ci = f"{lo:.0f} - {hi:.0f}" if lo == lo else "n/a"
            flag = "  *" if games[model] < args.min_games else ""
            tokens = usage.get(model)
            tok = f"{tokens:,.0f}" if tokens else "-"
            print(f"{model:<40} {rating:>7.0f} {ci:>17} {games[model]:>6} {tok:>10}{flag}")
        if any(games[m] < args.min_games for m in ratings):
            print(f"* provisional: fewer than {args.min_games} games; the interval "
                  f"spans most of the table, so ordering here is not meaningful.")

    if len(pools) > 1:
        print("\nModes are rated separately and are NOT comparable: each imposes "
              "different rules, so a rating in one says nothing about another.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
