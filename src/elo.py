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


def ranked_order(ratings, games, min_games):
    """Models eligible to be RANKED, best first.

    The gate is on display and ordering only. Matches played by an unranked
    model still update its opponents' ratings - discarding those games would
    distort every ranked model's number."""
    eligible = [(m, r) for m, r in ratings.items() if games.get(m, 0) >= min_games]
    return [m for m, _ in sorted(eligible, key=lambda kv: -kv[1])]


def spearman(rank_a, rank_b):
    """Rank correlation between two orderings of the same models.

    The single number that answers the project's actual question: does the time
    regime change the answer? 1.0 means the boards agree completely, 0 means
    they are unrelated, negative means they invert."""
    shared = sorted(set(rank_a) & set(rank_b))
    n = len(shared)
    if n < 2:
        return None
    a = {m: i for i, m in enumerate(rank_a) if m in set(shared)}
    b = {m: i for i, m in enumerate(rank_b) if m in set(shared)}
    # Re-rank within the intersection so a model present in only one pool
    # cannot shift everyone else's number and manufacture a correlation.
    ra = {m: i for i, m in enumerate(sorted(shared, key=lambda m: a[m]))}
    rb = {m: i for i, m in enumerate(sorted(shared, key=lambda m: b[m]))}
    d2 = sum((ra[m] - rb[m]) ** 2 for m in shared)
    return round(1 - (6 * d2) / (n * (n * n - 1)), 3)


def compare_pools(pools, names, k, min_games):
    """Cross-mode rank comparison.

    Ratings are NOT comparable across pools - each is anchored independently -
    so this compares RANKS, and only over models ranked in every pool named."""
    orders = {}
    for name in names:
        if name not in pools:
            print(f"No rated matches in mode {name!r}")
            return
        ratings, games = rate(pools[name], k)
        orders[name] = ranked_order(ratings, games, min_games)

    shared = set(orders[names[0]])
    for name in names[1:]:
        shared &= set(orders[name])
    if not shared:
        print("No model is ranked in every mode named, so ranks are not comparable.")
        return

    width = max(len(m) for m in shared) + 2
    header = f"{'MODEL':<{width}}" + "".join(f"{n:>13}" for n in names)
    if len(names) == 2:
        header += f"{'D-rank':>8}"
    print(header)
    print("-" * len(header))

    within = {n: [m for m in orders[n] if m in shared] for n in names}
    for model in sorted(shared, key=lambda m: within[names[0]].index(m)):
        row = f"{model:<{width}}"
        positions = []
        for name in names:
            pos = within[name].index(m := model) + 1
            positions.append(pos)
            row += f"{('#' + str(pos)):>13}"
        if len(names) == 2:
            delta = positions[1] - positions[0]
            row += f"{delta:>+8}"
        print(row)

    if len(names) == 2:
        rho = spearman(within[names[0]], within[names[1]])
        if rho is not None:
            print(f"\nSpearman rank correlation ({names[0]} vs {names[1]}): "
                  f"{rho} over {len(shared)} models")
            if rho < 0.5:
                print("The two regimes disagree substantially: the time rule is "
                      "changing who wins, which is what these modes exist to expose.")


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
    parser.add_argument("--compare", default=None,
                        help="compare RANKS across two or more modes, "
                             "e.g. --compare untimed,realtime")
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
    if args.compare:
        names = [n.strip() for n in args.compare.split(",") if n.strip()]
        compare_pools(pools, names, args.k, args.min_games)
        return 0
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
        print(f"{'RANK':>4}  {'MODEL':<38} {'ELO':>7} {'95% CI':>17} {'GAMES':>6} {'TOK/MATCH':>10}")
        print("-" * 90)

        def row(model, rank_label):
            lo, hi = intervals.get(model, (float("nan"), float("nan")))
            ci = f"{lo:.0f} - {hi:.0f}" if lo == lo else "n/a"
            tokens = usage.get(model)
            tok = f"{tokens:,.0f}" if tokens else "-"
            print(f"{rank_label:>4}  {model:<38} {ratings[model]:>7.0f} {ci:>17} "
                  f"{games[model]:>6} {tok:>10}")

        ranked = ranked_order(ratings, games, args.min_games)
        for position, model in enumerate(ranked, 1):
            row(model, str(position))

        unranked = sorted((m for m in ratings if m not in set(ranked)),
                          key=lambda m: -ratings[m])
        if unranked:
            print(f"UNRANKED (fewer than {args.min_games} games; rating shown, "
                  f"not ordered - their matches still count toward ranked models)")
            for model in unranked:
                row(model, "-")

    if len(pools) > 1:
        print("\nModes are rated separately and are NOT comparable: each imposes "
              "different rules, so a rating in one says nothing about another.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
