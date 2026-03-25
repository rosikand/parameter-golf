#!/usr/bin/env python3
"""Fan out parallel Parameter Golf training runs on Modal.

Usage:
    python infra/sweep.py --script experiments/baseline.py --name "lr_sweep" \\
        --vary LEARNING_RATE=0.001,0.003,0.01 --vary NUM_LAYERS=6,9,12 --gpus 1

    python infra/sweep.py --script "experiments/*.py" --gpus 1

    python infra/sweep.py --script experiments/baseline.py --name "lr_sweep" \\
        --vary LEARNING_RATE=0.001,0.003 --gpus 1 --yes
"""

import argparse
import glob
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEFAULT_ENV, H100_COST_PER_HOUR


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep Parameter Golf experiments on Modal")
    parser.add_argument(
        "--script", type=str, nargs="+", required=True,
        help="Path(s) to training script(s). Supports glob patterns.",
    )
    parser.add_argument(
        "--name", type=str, default="sweep",
        help="Sweep name prefix (default: 'sweep').",
    )
    parser.add_argument(
        "--gpus", type=int, default=1, choices=[1, 2, 4, 8],
        help="Number of H100 GPUs per run (default: 1).",
    )
    parser.add_argument(
        "--vary", type=str, action="append", default=[],
        help="Sweep variable as KEY=v1,v2,v3. Repeatable. Cartesian product of all --vary args.",
    )
    parser.add_argument(
        "--env", type=str, action="append", default=[],
        help="Fixed environment variable override as KEY=VALUE. Repeatable.",
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable W&B logging.",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip confirmation prompt.",
    )
    return parser.parse_args()


def expand_scripts(patterns: list[str]) -> list[str]:
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            print(f"ERROR: No files match pattern: {pattern}", file=sys.stderr)
            sys.exit(1)
        paths.extend(matches)
    return paths


def parse_vary_args(vary_args: list[str]) -> list[tuple[str, list[str]]]:
    """Parse ['KEY=v1,v2,v3', ...] into [(KEY, [v1, v2, v3]), ...]."""
    result = []
    for arg in vary_args:
        if "=" not in arg:
            print(f"ERROR: --vary must be KEY=v1,v2,v3, got: {arg}", file=sys.stderr)
            sys.exit(1)
        key, values_str = arg.split("=", 1)
        values = [v.strip() for v in values_str.split(",")]
        result.append((key, values))
    return result


def parse_env_args(env_args: list[str]) -> dict:
    result = {}
    for arg in env_args:
        if "=" not in arg:
            print(f"ERROR: --env must be KEY=VALUE, got: {arg}", file=sys.stderr)
            sys.exit(1)
        key, value = arg.split("=", 1)
        result[key] = value
    return result


def build_sweep_matrix(vary_parsed: list[tuple[str, list[str]]]) -> list[dict]:
    """Build cartesian product of sweep variables."""
    if not vary_parsed:
        return [{}]

    keys = [k for k, _ in vary_parsed]
    value_lists = [v for _, v in vary_parsed]

    matrix = []
    for combo in itertools.product(*value_lists):
        matrix.append(dict(zip(keys, combo)))
    return matrix


def make_run_name(prefix: str, script_basename: str, env_combo: dict, num_scripts: int) -> str:
    """Generate a unique run name from sweep parameters."""
    parts = [prefix]
    if num_scripts > 1:
        parts.append(os.path.splitext(script_basename)[0])
    for key, value in sorted(env_combo.items()):
        # Abbreviate key names for readability
        short_key = "".join(c for c in key if c.isupper() or c.isdigit()) or key[:4]
        parts.append(f"{short_key}{value}")
    return "_".join(parts)


def main():
    args = parse_args()

    scripts = expand_scripts(args.script)
    vary_parsed = parse_vary_args(args.vary)
    fixed_env = parse_env_args(args.env)
    matrix = build_sweep_matrix(vary_parsed)
    wandb_enabled = not args.no_wandb

    # Build full run list
    runs = []
    for script_path in scripts:
        script_basename = os.path.basename(script_path)
        with open(script_path) as f:
            script_content = f.read()

        for env_combo in matrix:
            run_name = make_run_name(args.name, script_basename, env_combo, len(scripts))
            merged_env = {**fixed_env, **env_combo}
            runs.append({
                "script_path": script_path,
                "script_basename": script_basename,
                "script_content": script_content,
                "run_name": run_name,
                "env_vars": merged_env,
            })

    # Print sweep summary
    total_runs = len(runs)
    est_cost = total_runs * args.gpus * (600 / 3600) * H100_COST_PER_HOUR

    print(f"Sweep: {total_runs} runs on {args.gpus}x H100 each")
    print(f"Estimated cost (10min/run): ${est_cost:.2f}")
    print()
    for i, run in enumerate(runs, 1):
        env_str = " ".join(f"{k}={v}" for k, v in run["env_vars"].items())
        print(f"  [{i}] {run['run_name']}  ({run['script_basename']})  {env_str}")
    print()

    if not args.yes:
        confirm = input("Proceed? [y/N] ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Aborted.")
            return

    # Launch all runs in parallel via Modal starmap
    from modal_app import make_train_app
    from leaderboard import add_result, print_summary

    train_app, train_fn = make_train_app(args.gpus)

    starmap_args = [
        (
            run["script_content"],
            run["env_vars"],
            run["run_name"],
            wandb_enabled,
        )
        for run in runs
    ]

    print(f"\nLaunching {total_runs} runs on Modal...")
    results = []
    with train_app.run():
        for result in train_fn.starmap(starmap_args):
            results.append(result)

    # Process results
    print("\n" + "=" * 70)
    print("  SWEEP RESULTS")
    print("=" * 70)

    # Sort by best score
    def score(r):
        for field in ("final_int8_val_bpb", "val_bpb"):
            v = r.get(field)
            if v is not None:
                return float(v)
        return float("inf")

    results.sort(key=score)

    print(f"\n{'Rank':<5} {'Run':<30} {'val_bpb':<10} {'int8_bpb':<10} {'Exit':<6}")
    print("-" * 65)
    for i, result in enumerate(results, 1):
        print(
            f"{i:<5} "
            f"{result.get('run_name', '?'):<30} "
            f"{result.get('val_bpb', 'N/A'):<10} "
            f"{result.get('final_int8_val_bpb', 'N/A'):<10} "
            f"{result.get('exit_code', '?'):<6}"
        )

    # Save to leaderboard
    for result in results:
        run_info = next((r for r in runs if r["run_name"] == result.get("run_name")), {})
        add_result(
            result,
            script_name=run_info.get("script_basename", ""),
            gpus=args.gpus,
        )

    print(f"\n{len(results)} results saved to leaderboard.")


if __name__ == "__main__":
    main()
