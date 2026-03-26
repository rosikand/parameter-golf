# Infra

Run Parameter Golf experiments on remote H100s via [Modal](https://modal.com). Edit training scripts locally, launch with one command, get results in W&B and a local leaderboard.

## Setup (one-time)

```bash
pip install modal wandb
modal setup
modal secret create wandb-secret WANDB_API_KEY=<your_key>
python infra/run.py --ensure-data   # downloads dataset to Modal volume (~5 min)
```

## Usage

```bash
# Run a single experiment on 1x H100
python infra/run.py --script experiments/baseline.py --name "baseline" --gpus 1

# Override env vars
python infra/run.py --script experiments/baseline.py --name "test_lr" --gpus 1 \
    --env LEARNING_RATE=0.003 --env NUM_LAYERS=12

# Fire off multiple scripts at once
python infra/run.py --script "experiments/*.py" --gpus 1

# Sweep hyperparameters (cartesian product, runs in parallel)
python infra/sweep.py --script experiments/baseline.py --name "lr_sweep" \
    --vary LEARNING_RATE=0.001,0.003,0.01 --vary NUM_LAYERS=6,9,12 --gpus 1

# Dry run (see what would execute without launching)
python infra/run.py --script experiments/baseline.py --name "test" --dry-run

# Final submission validation on 8x H100
python infra/run.py --script experiments/my_best.py --name "submission" --gpus 8
```

## Workflow

1. Copy `experiments/baseline.py` and implement your idea
2. Run it: `python infra/run.py --script experiments/my_idea.py --name "my_idea_v1" --gpus 1`
3. Logs stream to your terminal, metrics log to W&B, results append to `infra/LEADERBOARD.md`
4. Iterate

## Files

| File | Purpose |
|------|---------|
| `config.py` | Shared constants (Modal volume, W&B project, default env vars) |
| `modal_app.py` | Modal app definition (image, volume, GPU functions) |
| `run.py` | CLI for single runs |
| `sweep.py` | CLI for parallel sweeps |
| `log_parser.py` | Parses `train_gpt.py` structured output for W&B + leaderboard |
| `leaderboard.py` | Manages `leaderboard.csv` and `LEADERBOARD.md` |
| `DESIGN.md` | Full design doc |

## Cost

| Config | ~Cost per 10min run |
|--------|---------------------|
| 1x H100 | $0.66 |
| 2x H100 | $1.32 |
| 8x H100 | $5.27 |
