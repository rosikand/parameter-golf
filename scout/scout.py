#!/usr/bin/env python3
"""
Parameter Golf Scout — Daily competition intelligence agent.

Usage:
    python scout.py              # full run: fetch + analyze + briefing
    python scout.py --fetch-only # just fetch, no Claude analysis
    python scout.py --brief-only # regenerate briefing from existing state (no fetch)

Environment variables:
    GITHUB_TOKEN       — optional, raises GitHub API rate limit (60 → 5000/hr)
    ANTHROPIC_API_KEY  — required for analysis/briefing steps
"""

import argparse
import builtins
import sys
import os
from functools import partial

# Force flush on all prints so progress is visible when output is piped
print = partial(builtins.print, flush=True)

# Ensure scout/ is on the path so imports work when run from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fetch import fetch_all, fetch_diff
from analyze import (
    analyze_all,
    generate_briefing,
    _load_json,
    update_history,
)
from config import STATE_DIR


def main():
    parser = argparse.ArgumentParser(description="Parameter Golf Scout")
    parser.add_argument("--fetch-only", action="store_true", help="Only fetch data, skip analysis")
    parser.add_argument("--brief-only", action="store_true", help="Regenerate briefing from existing state")
    parser.add_argument("--diff", type=int, nargs="+", metavar="PR", help="Fetch diff for specific PR(s)")
    args = parser.parse_args()

    if args.diff:
        for pr_num in args.diff:
            fetch_diff(pr_num)
        return

    if args.brief_only:
        print("=== Parameter Golf Scout (briefing only) ===\n")
        leaderboard = _load_json("leaderboard.json", default=[])
        techniques = _load_json("techniques.json", default={})
        issues = _load_json("issues.json", default=[])
        history = _load_json("history.json", default=[])
        if not leaderboard:
            print("No leaderboard data found. Run a full scout first.")
            sys.exit(1)
        briefing = generate_briefing(leaderboard, techniques, issues, history, 0, 0)
        print("\n" + briefing)
        return

    print("=== Parameter Golf Scout ===\n")

    # Step 1: Fetch
    print("--- Fetch ---")
    fetched = fetch_all()

    if args.fetch_only:
        print("\nFetch complete. Skipping analysis (--fetch-only).")
        return

    # Step 2: Analyze + Brief
    if not fetched["prs"] and not fetched["issues"]:
        print("\nNo new data to analyze. Regenerating briefing from existing state...")
        leaderboard = _load_json("leaderboard.json", default=[])
        techniques = _load_json("techniques.json", default={})
        issues = _load_json("issues.json", default=[])
        history = _load_json("history.json", default=[])
        if leaderboard:
            generate_briefing(leaderboard, techniques, issues, history, 0, 0)
        else:
            print("No existing state found either. Nothing to do.")
        return

    result = analyze_all(fetched)

    # Print summary
    top5 = result["leaderboard"][:5]
    print("\n=== Top 5 ===")
    for i, entry in enumerate(top5, 1):
        bpb = entry.get("bpb_score", "?")
        techs = ", ".join(entry.get("techniques", [])[:4])
        ttt = " [TTT]" if entry.get("uses_ttt") else ""
        print(f"  {i}. PR#{entry['number']} @{entry.get('user', '?')} — {bpb} BPB{ttt} ({techs})")

    print(f"\nDone. Briefing saved to briefings/ directory.")


if __name__ == "__main__":
    main()
