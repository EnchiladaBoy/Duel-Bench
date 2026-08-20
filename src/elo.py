#!/usr/bin/env python3
"""Minimal ELO leaderboard over matches/*/result.json.

Usage:
    python3 src/elo.py [--matches-dir matches] [--k 32]
"""

import argparse
import json
from pathlib import Path


def expected_score(ra, rb):
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def load_results(matches_dir):
    results = []
    for path in sorted(Path(matches_dir).glob("*/result.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[warn] skipping {path}: {exc}")
            continue
        if not data.get("model_a") or not data.get("model_b"):
            continue
        if data.get("winner") not in ("agent-a", "agent-b", "draw"):
            continue
        results.append(data)
    return results


def main():
    parser = argparse.ArgumentParser(description="agent-deathmatch ELO leaderboard")
    parser.add_argument("--matches-dir", default="matches")
    parser.add_argument("--k", type=float, default=32.0)
    args = parser.parse_args()

    results = load_results(args.matches_dir)
    if not results:
        print(f"No finished matches found in {args.matches_dir}/")
        return 0

    ratings = {}
    games = {}
    for data in results:
        ma, mb = data["model_a"], data["model_b"]
        winner = data["winner"]
        score_a = 1.0 if winner == "agent-a" else (0.0 if winner == "agent-b" else 0.5)

        ra = ratings.setdefault(ma, 1500.0)
        rb = ratings.setdefault(mb, 1500.0)
        ea = expected_score(ra, rb)
        eb = 1.0 - ea
        ratings[ma] = ra + args.k * (score_a - ea)
        ratings[mb] = rb + args.k * ((1.0 - score_a) - eb)
        games[ma] = games.get(ma, 0) + 1
        games[mb] = games.get(mb, 0) + 1

    print(f"{'MODEL':<45} {'ELO':>7} {'GAMES':>6}")
    print("-" * 60)
    for model, rating in sorted(ratings.items(), key=lambda kv: -kv[1]):
        print(f"{model:<45} {rating:>7.0f} {games[model]:>6}")
    print(f"\n({len(results)} matches processed, K={args.k})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
