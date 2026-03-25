#!/usr/bin/env python3
"""
Local leaderboard for Parameter Golf experiments.

    python leaderboard.py              # show leaderboard
    python leaderboard.py --sync       # pull latest results from Modal Volume
    python leaderboard.py --full       # show all columns
    python leaderboard.py --log RUN_ID # fetch full training log for a run
    python leaderboard.py --compare A B
    python leaderboard.py --delete RUN_ID
"""

import argparse
import json
import sys
from pathlib import Path

LB_PATH = Path(__file__).parent / "leaderboard.json"


def load():
    if not LB_PATH.exists():
        return []
    return json.loads(LB_PATH.read_text())


def save(entries):
    LB_PATH.write_text(json.dumps(entries, indent=2))


def sync():
    """Pull all results from Modal Volume into local leaderboard.json."""
    print("Syncing results from Modal Volume...")

    from modal_train_frozen import app, sync_results

    with app.run():
        remote_results = sync_results.remote()
    local_entries = load()

    # Merge: update existing runs, add new ones
    local_by_id = {e["run_id"]: e for e in local_entries}
    for r in remote_results:
        run_id = r["run_id"]
        if run_id in local_by_id:
            # Update with latest (might have been running, now done)
            local_by_id[run_id].update(r)
        else:
            local_by_id[run_id] = r

    merged = list(local_by_id.values())
    save(merged)

    done = [e for e in remote_results if e.get("status") == "done"]
    running = [e for e in remote_results if e.get("status") == "running"]
    print(f"Synced: {len(done)} done, {len(running)} running, {len(merged)} total in leaderboard")


def fetch_log(run_id: str):
    """Fetch and print the full training log for a run."""
    try:
        from modal_train_frozen import app, get_run_log
        with app.run():
            log = get_run_log.remote(run_id)
        print(log)
    except Exception as e:
        print(f"Error fetching log: {e}")


def show(entries, full=False):
    done = [e for e in entries if e.get("val_bpb_int8") is not None]
    running = [e for e in entries if e.get("status") == "running" and e.get("val_bpb_int8") is None]
    failed = [e for e in entries if e.get("val_bpb_int8") is None and e.get("status") != "running"]

    if not done and not running and not failed:
        print("No runs recorded yet. Run: python leaderboard.py --sync")
        return

    done.sort(key=lambda x: x["val_bpb_int8"])

    if done:
        best_bpb = done[0]["val_bpb_int8"]
        print()
        print("=" * 76)
        print("  PARAMETER GOLF — LOCAL LEADERBOARD")
        print("=" * 76)

        if full:
            print(f"  {'#':>3}  {'BPB (int8)':>11}  {'Loss (int8)':>12}  {'BPB (f32)':>10}  {'Size MB':>8}  {'Steps':>6}  {'Time':>6}  {'Run ID'}")
            print("-" * 76)
        else:
            print(f"  {'#':>3}  {'BPB (int8)':>11}  {'Δ best':>8}  {'Size MB':>8}  {'Steps':>6}  {'Run ID'}")
            print("-" * 76)

        for i, e in enumerate(done, 1):
            bpb = e["val_bpb_int8"]
            delta = bpb - best_bpb
            size_mb = f"{e['total_bytes'] / 1_000_000:.2f}" if e.get("total_bytes") else "?"
            steps = e.get("steps_completed", "?")
            elapsed = f"{e.get('elapsed_seconds', 0):.0f}s" if e.get("elapsed_seconds") else "?"
            rid = e["run_id"]
            marker = " *" if i == 1 else "  "

            if full:
                loss_int8 = f"{e.get('val_loss_int8', 0):.6f}" if e.get("val_loss_int8") else "?"
                bpb_f32 = f"{e.get('val_bpb', 0):.6f}" if e.get("val_bpb") else "?"
                print(f"{marker}{i:>2}  {bpb:>11.6f}  {loss_int8:>12}  {bpb_f32:>10}  {size_mb:>8}  {steps:>6}  {elapsed:>6}  {rid}")
            else:
                delta_str = f"+{delta:.6f}" if delta > 0 else "  best"
                print(f"{marker}{i:>2}  {bpb:>11.6f}  {delta_str:>8}  {size_mb:>8}  {steps:>6}  {rid}")

        print()
        print(f"  Challenge baseline (8xH100): 1.2244 BPB")
        print(f"  Your best (1xH100):          {done[0]['val_bpb_int8']:.4f} BPB")

    if running:
        print(f"\n  Running ({len(running)}):")
        for e in running:
            print(f"    - {e['run_id']}")

    if failed:
        print(f"\n  No result ({len(failed)}):")
        for e in failed:
            print(f"    - {e['run_id']}  status={e.get('status', '?')}")


def compare(entries, run_a, run_b):
    a = next((e for e in entries if e["run_id"] == run_a), None)
    b = next((e for e in entries if e["run_id"] == run_b), None)
    if not a or not b:
        print(f"Run not found: {run_a if not a else run_b}")
        return

    print(f"\n{'':>20}  {'Run A':>14}  {'Run B':>14}  {'Delta':>10}")
    print("-" * 65)
    for key in ["val_bpb_int8", "val_loss_int8", "val_bpb", "val_loss",
                 "total_bytes", "model_bytes_int8_zlib", "steps_completed", "elapsed_seconds"]:
        va, vb = a.get(key), b.get(key)
        delta = ""
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            d = vb - va
            delta = f"{d:+.6f}" if isinstance(va, float) else f"{d:+d}"
        print(f"  {key:>20}  {str(va):>14}  {str(vb):>14}  {delta:>10}")


def main():
    parser = argparse.ArgumentParser(description="Parameter Golf leaderboard")
    parser.add_argument("--sync", action="store_true", help="Pull results from Modal Volume")
    parser.add_argument("--full", action="store_true", help="Show all columns")
    parser.add_argument("--log", metavar="RUN_ID", help="Fetch training log for a run")
    parser.add_argument("--delete", metavar="RUN_ID", help="Delete a run")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"), help="Compare two runs")
    args = parser.parse_args()

    if args.sync:
        sync()
        show(load())
    elif args.log:
        fetch_log(args.log)
    elif args.delete:
        entries = load()
        before = len(entries)
        entries = [e for e in entries if e["run_id"] != args.delete]
        if len(entries) < before:
            save(entries)
            print(f"Deleted: {args.delete}")
        else:
            print(f"Not found: {args.delete}")
    elif args.compare:
        compare(load(), *args.compare)
    else:
        show(load(), full=args.full)


if __name__ == "__main__":
    main()