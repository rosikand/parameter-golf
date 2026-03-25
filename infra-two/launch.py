#!/usr/bin/env python3
"""
Launch Parameter Golf experiments (fire-and-forget).

    python launch.py my_train.py --name idea_1
    python launch.py my_train.py --name idea_2
    python launch.py my_train.py --name idea_3
    # all return instantly — close laptop, go to sleep
    # check W&B live, or sync results later with: python leaderboard.py --sync
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Launch a Parameter Golf run on Modal (fire-and-forget)")
    parser.add_argument("train_script", help="Path to your training script")
    parser.add_argument("--name", "-n", default="", help="Run name (auto-generated if empty)")
    parser.add_argument("--variant", default="sp1024")
    parser.add_argument("--wandb-project", default="parameter-golf")

    # Presets
    parser.add_argument("--smoke", action="store_true", help="Quick 2-min test")
    parser.add_argument("--fast", action="store_true", help="Skip data cache check")

    # Architecture
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=9)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=4)

    # Training
    parser.add_argument("--max-wallclock", type=int, default=600)
    parser.add_argument("--val-every", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=20000)

    args = parser.parse_args()

    script = Path(args.train_script)
    if not script.exists():
        print(f"Error: {script} not found")
        sys.exit(1)

    # Generate run ID
    if args.name:
        run_id = f"{args.name}_{datetime.now().strftime('%m%d_%H%M')}"
    else:
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if args.smoke:
        args.max_wallclock = 120
        args.val_every = 50
        args.iterations = 5000
        run_id = f"smoke_{run_id}"

    # Build modal run command
    cmd = [
        "modal", "run", "--detach", "modal_train_frozen.py",
        "--train-script", str(script),
        "--run-id", run_id,
        "--variant", args.variant,
        "--wandb-project", args.wandb_project,
        "--max-wallclock", str(args.max_wallclock),
        "--val-loss-every", str(args.val_every),
        "--vocab-size", str(args.vocab_size),
        "--num-layers", str(args.num_layers),
        "--model-dim", str(args.model_dim),
        "--num-heads", str(args.num_heads),
        "--num-kv-heads", str(args.num_kv_heads),
        "--iterations", str(args.iterations),
    ]

    if args.fast:
        cmd.append("--skip-data-check")

    print(f"  Launching: {run_id}")
    print(f"  Script:    {script}")
    print(f"  Config:    {args.num_layers}L x {args.model_dim}D, vocab={args.vocab_size}")
    print(f"  Wallclock: {args.max_wallclock}s")
    print()

    subprocess.run(cmd, cwd=str(Path(__file__).parent))


if __name__ == "__main__":
    main()