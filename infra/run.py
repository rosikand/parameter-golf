#!/usr/bin/env python3
"""Launch a single Parameter Golf training run on Modal.

Usage:
    python infra/run.py --script experiments/baseline.py --name "baseline_v1" --gpus 1
    python infra/run.py --script experiments/baseline.py --name "test_lr" --gpus 1 --env LEARNING_RATE=0.003
    python infra/run.py --script "experiments/*.py" --gpus 1
    python infra/run.py --ensure-data
"""

import argparse
import glob
import os
import sys

# Ensure infra/ is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEFAULT_ENV, H100_COST_PER_HOUR, PROJECT_ROOT


def parse_args():
    parser = argparse.ArgumentParser(description="Launch Parameter Golf training on Modal")
    parser.add_argument(
        "--script", type=str, nargs="+",
        help="Path(s) to training script(s). Supports glob patterns.",
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Run name (used as RUN_ID and W&B run name). Auto-generated from script name if omitted.",
    )
    parser.add_argument(
        "--gpus", type=int, default=1, choices=[1, 2, 4, 8],
        help="Number of H100 GPUs (default: 1).",
    )
    parser.add_argument(
        "--env", type=str, action="append", default=[],
        help="Environment variable override as KEY=VALUE. Repeatable.",
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable W&B logging.",
    )
    parser.add_argument(
        "--ensure-data", action="store_true",
        help="Download dataset to Modal volume and exit.",
    )
    parser.add_argument(
        "--train-shards", type=int, default=80,
        help="Number of training shards to download (for --ensure-data).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be executed without launching.",
    )
    return parser.parse_args()


def parse_env_args(env_args: list[str]) -> dict:
    result = {}
    for arg in env_args:
        if "=" not in arg:
            print(f"ERROR: --env argument must be KEY=VALUE, got: {arg}", file=sys.stderr)
            sys.exit(1)
        key, value = arg.split("=", 1)
        result[key] = value
    return result


def expand_scripts(patterns: list[str]) -> list[str]:
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            print(f"ERROR: No files match pattern: {pattern}", file=sys.stderr)
            sys.exit(1)
        paths.extend(matches)
    return paths


def run_ensure_data(train_shards: int):
    """Download dataset to Modal volume."""
    from modal_app import data_app, ensure_data

    data_script_path = os.path.join(PROJECT_ROOT, "data", "cached_challenge_fineweb.py")
    with open(data_script_path) as f:
        data_script = f.read()

    print(f"Downloading dataset to Modal volume (train_shards={train_shards})...")
    print("This may take several minutes on first run.")

    with data_app.run():
        ensure_data.remote(data_download_script=data_script, train_shards=train_shards)

    print("Done.")


def run_single(script_path: str, run_name: str, gpus: int, env_vars: dict, wandb_enabled: bool, dry_run: bool):
    """Launch a single training run on Modal."""
    with open(script_path) as f:
        script_content = f.read()

    script_basename = os.path.basename(script_path)

    if dry_run:
        print(f"[DRY RUN] Would launch:")
        print(f"  Script:  {script_path}")
        print(f"  Name:    {run_name}")
        print(f"  GPUs:    {gpus}")
        print(f"  W&B:     {'yes' if wandb_enabled else 'no'}")
        print(f"  Env overrides: {env_vars}")
        est_cost = gpus * (600 / 3600) * H100_COST_PER_HOUR
        print(f"  Est. cost (10min): ${est_cost:.2f}")
        return

    print(f"Launching: {script_basename} as '{run_name}' on {gpus}x H100")
    est_cost = gpus * (600 / 3600) * H100_COST_PER_HOUR
    print(f"Estimated cost (10min): ${est_cost:.2f}")

    from modal_app import train_app, get_train_fn
    train_fn = get_train_fn(gpus)

    with train_app.run():
        result = train_fn.remote(
            script_content=script_content,
            env_vars=env_vars,
            run_name=run_name,
            wandb_enabled=wandb_enabled,
        )

    from leaderboard import add_result, print_summary
    print_summary(result, gpus=gpus)
    add_result(result, script_name=script_basename, gpus=gpus)
    print(f"Results saved to leaderboard.")


def main():
    args = parse_args()

    if args.ensure_data:
        run_ensure_data(args.train_shards)
        return

    if not args.script:
        print("ERROR: --script is required (unless using --ensure-data)", file=sys.stderr)
        sys.exit(1)

    env_vars = parse_env_args(args.env)
    scripts = expand_scripts(args.script)
    wandb_enabled = not args.no_wandb

    if len(scripts) == 1:
        name = args.name or os.path.splitext(os.path.basename(scripts[0]))[0]
        run_single(scripts[0], name, args.gpus, env_vars, wandb_enabled, args.dry_run)
    else:
        print(f"Launching {len(scripts)} scripts...")
        for script_path in scripts:
            name = os.path.splitext(os.path.basename(script_path))[0]
            if args.name:
                name = f"{args.name}_{name}"
            run_single(script_path, name, args.gpus, env_vars, wandb_enabled, args.dry_run)


if __name__ == "__main__":
    main()
