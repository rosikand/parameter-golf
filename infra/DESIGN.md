# Infrastructure Design Doc: Parameter Golf Experiment Runner

## Context

We're competing in OpenAI's Parameter Golf challenge (train best LM under 16MB + 10min on 8xH100s). We need infrastructure to: edit training scripts locally, fire off runs on remote H100s, fan out sweeps, track results. The goal is fast iteration on 1xH100, final validation on 8xH100.

## User Workflow

```
1. Create/edit a script in experiments/    (e.g. experiments/depth_recurrence.py)
2. Launch:  python infra/run.py --script experiments/depth_recurrence.py --name "dr_v1" --gpus 1
3. Logs stream to terminal, metrics go to W&B, results saved to local leaderboard
4. For sweeps: python infra/sweep.py --script experiments/baseline.py --vary LEARNING_RATE=0.001,0.003 --vary NUM_LAYERS=6,9
5. For multi-script: python infra/run.py --script "experiments/*.py" --gpus 1
```

## File Structure

```
parameter-golf/
├── experiments/
│   └── baseline.py              # Copy of train_gpt.py as starting point
├── infra/
│   ├── __init__.py
│   ├── config.py                # Constants: volume names, image def, default env vars
│   ├── modal_app.py             # Modal app: image, volume, train_run(), ensure_data()
│   ├── run.py                   # CLI: single run launcher
│   ├── sweep.py                 # CLI: parallel sweep launcher
│   ├── log_parser.py            # Parse train_gpt.py structured output
│   ├── leaderboard.py           # Result tracking (CSV + markdown)
│   └── leaderboard.csv          # Machine-readable results log
└── LEADERBOARD.md               # Human-readable: runs, techniques, scores, notes
```

## Component Design

### `infra/config.py` — Shared constants

- `MODAL_APP_NAME = "parameter-golf"`
- `MODAL_VOLUME_NAME = "parameter-golf-data"` — persistent volume for dataset
- `MODAL_VOLUME_MOUNT = "/data"`
- `WANDB_PROJECT = "parameter-golf"`
- `DEFAULT_ENV` — data paths pointing to `/data/...` on the volume, plus `VOCAB_SIZE=1024`, `MAX_WALLCLOCK_SECONDS=600`
- `IMAGE_PACKAGES` — torch deps + wandb

### `infra/modal_app.py` — Modal function definitions

**Image:** Based on NGC PyTorch image (`nvcr.io/nvidia/pytorch:24.04-py3`) with `wandb`, `sentencepiece`, `kernels` added. Avoids building CUDA/cuDNN from scratch.

**Volume:** `modal.Volume.from_name("parameter-golf-data", create_if_missing=True)` — persistent storage for the dataset (~16GB). Downloaded once, reused across all runs.

**`ensure_data()`** — Modal function that runs `cached_challenge_fineweb.py` to download dataset into the volume. One-time setup: `python infra/run.py --ensure-data`.

**`train_run(script_content, env_vars, run_name, gpus)`** — Core function:
1. Writes script to `/tmp/train_script.py` in the container
2. Merges `DEFAULT_ENV` with caller's env vars, sets `RUN_ID`
3. Inits W&B run
4. Launches `torchrun --standalone --nproc_per_node={gpus} /tmp/train_script.py` via subprocess
5. Streams stdout line-by-line: prints to terminal + parses metrics for W&B
6. Returns result dict: `{run_name, exit_code, val_loss, val_bpb, final_int8_val_bpb, model_bytes, wallclock_s, ...}`

Key decisions:
- **Script uploaded as string argument**, not mounted. Simple, works for rapid iteration.
- **W&B logging in the wrapper, not the training script.** Experiment scripts stay identical to upstream `train_gpt.py` — no W&B imports needed. The wrapper parses structured stdout.
- **`PYTHONUNBUFFERED=1`** in subprocess env for real-time log streaming.
- **Timeout: 1200s** (20 min). Training caps at 600s but there's startup/eval overhead.
- GPU: `modal.gpu.H100(count=gpus)`. Valid counts: 1, 2, 4, 8.

### `infra/log_parser.py` — Structured log parsing

Regex-based parser for `train_gpt.py` output format:
- `step:N/M train_loss:F` → train metrics for W&B
- `step:N/M val_loss:F val_bpb:F` → val metrics for W&B
- `final_int8_zlib_roundtrip val_loss:F val_bpb:F` → final score
- `final_int8_ttt_lora val_loss:F val_bpb:F` → TTT score
- `Serialized model int8+zlib: N bytes` → model size
- `Total submission size int8+zlib: N bytes` → submission size

Two main functions:
- `parse_line(line) -> dict | None` — per-line parsing for streaming W&B logs
- `extract_final_metrics(lines) -> dict` — scan all lines for final results

### `infra/run.py` — Single run CLI

```
python infra/run.py --script experiments/my_idea.py --name "test1" --gpus 1
python infra/run.py --script experiments/my_idea.py --name "test1" --gpus 1 --env LEARNING_RATE=0.003
python infra/run.py --script "experiments/*.py" --gpus 1   # multiple scripts
python infra/run.py --ensure-data                          # one-time data setup
```

Args: `--script`, `--name`, `--gpus` (default 1), `--env KEY=VALUE` (repeatable), `--no-wandb`, `--ensure-data`, `--dry-run`

After run completes: calls `leaderboard.add_result(result)`, prints summary with estimated cost (~$0.66 per 10min on 1xH100).

### `infra/sweep.py` — Parallel sweep CLI

```
python infra/sweep.py --script experiments/baseline.py --name "lr_sweep" \
    --vary LEARNING_RATE=0.001,0.003,0.01 --vary NUM_LAYERS=6,9,12 --gpus 1
```

- Builds cartesian product of `--vary` args (3 LR × 3 layers = 9 runs)
- Auto-names: `lr_sweep_LR0.001_NL6`, etc.
- Shows matrix + cost estimate, asks for confirmation (unless `--yes`)
- Launches all via `train_run.starmap()` — Modal handles parallel scheduling
- Collects all results, updates leaderboard, prints summary table sorted by val_bpb

### `infra/leaderboard.py` — Results tracking

**CSV** (`infra/leaderboard.csv`): append-only log
```
timestamp,run_name,script,gpus,val_bpb,final_int8_val_bpb,final_ttt_val_bpb,submission_bytes,wallclock_s,exit_code,notes
```

**Markdown** (`LEADERBOARD.md`): auto-regenerated from CSV, sorted by best val_bpb
```markdown
# Parameter Golf — Local Leaderboard

| # | Run | Script | val_bpb | int8_bpb | Size (KB) | GPUs | Time | Date | Notes |
|---|-----|--------|---------|----------|-----------|------|------|------|-------|
| 1 | depth_rec_v3 | depth_recurrence.py | 1.1748 | 1.1801 | 4521 | 1 | 9m42s | 2026-03-23 | |
```

### `experiments/` — Training script variants

- Start with `experiments/baseline.py` (copy of `train_gpt.py`)
- Each script is self-contained (model + optimizer + training loop in one file)
- All config via env vars (same `Hyperparameters` class pattern)
- Must print structured logs to stdout so `log_parser.py` can parse them

## Data Flow

```
run.py reads script → calls modal train_run.remote(script_content, env, name, gpus)
  → Modal provisions H100 container
  → writes script to /tmp, mounts data volume at /data
  → torchrun launches training
  → stdout streams back: terminal + W&B logging via log_parser
  → training finishes, final metrics extracted
  → result dict returned
run.py calls leaderboard.add_result(result)
```

## Error Handling

| Failure | Handling |
|---------|----------|
| OOM | Non-zero exit code captured. Leaderboard shows "OOM". W&B run marked crashed. |
| Modal timeout (>20min) | `TimeoutError` caught, logged as failed. |
| Script syntax error | torchrun fails immediately. Stderr streamed to user. |
| Data missing on volume | Check at start of `train_run`. Auto-download if missing. |
| W&B auth failure | Graceful fallback: warning printed, training proceeds without W&B. |

## Prerequisites / Setup

1. `pip install modal wandb` locally
2. `modal setup` (one-time Modal auth)
3. `modal secret create wandb-secret WANDB_API_KEY=<key>`
4. `python infra/run.py --ensure-data` (one-time dataset download to Modal volume)

## Implementation Order

1. `infra/config.py`
2. `infra/log_parser.py` (pure logic, no deps)
3. `infra/modal_app.py` (core Modal functions)
4. `infra/leaderboard.py` (can test locally with mock data)
5. `infra/run.py` (ties everything together)
6. `infra/sweep.py` (builds on run.py)
7. `experiments/baseline.py` (copy of train_gpt.py)

## Verification

1. `python infra/run.py --ensure-data` — confirm dataset appears on Modal volume
2. `python infra/run.py --script experiments/baseline.py --name "smoke" --gpus 1 --env ITERATIONS=50` — short smoke test
3. Confirm W&B dashboard shows step-level metrics
4. Confirm `LEADERBOARD.md` and `infra/leaderboard.csv` updated
5. `python infra/sweep.py --script experiments/baseline.py --vary LEARNING_RATE=0.01,0.03 --gpus 1` — confirm parallel runs + results collection
6. Full baseline: `python infra/run.py --script experiments/baseline.py --name "baseline_1gpu" --gpus 1` — confirm ~1.2244 val_bpb
