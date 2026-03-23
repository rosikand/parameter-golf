# Parameter Golf

OpenAI's "Parameter Golf" challenge: train the best LM under 16MB artifact + 10min on 8xH100s.
Metric: val_bpb (bits per byte) on FineWeb validation — lower is better.
Repo: https://github.com/openai/parameter-golf | Deadline: April 30, 2026

## Project structure

- `train_gpt.py` — main CUDA training script (torchrun). All hyperparams via env vars.
- `train_gpt_mlx.py` — Apple Silicon (MLX) variant for local iteration.
- `data/` — dataset download scripts, tokenizer specs. Datasets/tokenizers are gitignored.
- `records/` — leaderboard submissions (each in its own dated folder).
- `scout/` — custom competition intelligence agent. Fetches PRs/issues from GitHub, analyzes with Claude API, generates daily briefings. Run: `python scout/scout.py`
- `experiments/` — training script variants. Each is a self-contained copy/fork of train_gpt.py.
- `infra/` — Modal-based infrastructure for launching runs on remote H100s.
- `context/` — LLM handoff prompts summarizing the challenge for AI assistants.
- `notes/` — working notes (compute credit request, etc.).
- `LEADERBOARD.md` — auto-generated local leaderboard of all experiment runs.

## Experiment workflow

1. Create/edit a script in `experiments/` (fork from `experiments/baseline.py`)
2. Launch on Modal: `python infra/run.py --script experiments/my_idea.py --name "my_idea_v1" --gpus 1`
3. Logs stream to terminal, metrics go to W&B, results saved to local leaderboard
4. For sweeps: `python infra/sweep.py --script experiments/baseline.py --vary LEARNING_RATE=0.001,0.003 --vary NUM_LAYERS=6,9`

## Commands

```bash
# Infrastructure setup (one-time)
pip install modal wandb
modal setup
modal secret create wandb-secret WANDB_API_KEY=<key>
python infra/run.py --ensure-data  # download dataset to Modal volume

# Launch a single run on 1x H100
python infra/run.py --script experiments/baseline.py --name "baseline" --gpus 1

# Launch with env var overrides
python infra/run.py --script experiments/baseline.py --name "test_lr" --gpus 1 --env LEARNING_RATE=0.003

# Launch multiple scripts
python infra/run.py --script "experiments/*.py" --gpus 1

# Sweep (cartesian product, parallel)
python infra/sweep.py --script experiments/baseline.py --name "lr_sweep" \
    --vary LEARNING_RATE=0.001,0.003,0.01 --vary NUM_LAYERS=6,9,12 --gpus 1

# Dry run (show what would execute)
python infra/run.py --script experiments/baseline.py --name "test" --dry-run

# Final submission validation on 8x H100
python infra/run.py --script experiments/my_best.py --name "submission" --gpus 8

# Local smoke test (MLX, Apple Silicon)
RUN_ID=mlx_smoke ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 python3 train_gpt_mlx.py

# Scout
python scout/scout.py              # full run: fetch + analyze + briefing
python scout/scout.py --fetch-only # just fetch GitHub data
python scout/scout.py --brief-only # regenerate briefing from cached state
```

## Environment variables

- `GITHUB_TOKEN` — raises GitHub API rate limit (60 → 5000/hr) for scout
- `ANTHROPIC_API_KEY` — required for scout analysis/briefing
- `WANDB_API_KEY` — for W&B logging (set as Modal secret too)

## Key constraints

- Artifact = code bytes + int8-quantized zlib-compressed weights, must be < 16,000,000 bytes
- New SOTA must beat current best by ≥ 0.005 nats at p < 0.01
- No network calls allowed during evaluation; artifact must be self-contained
