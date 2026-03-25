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


---

# OpenAI Parameter Golf — Competition Rules & Constraints

**Read this before writing or modifying ANY code. Violating these rules will get the submission disqualified.**

## Hard Constraints

- **Artifact size limit:** Total of compressed model weights + training code ≤ 16,000,000 bytes (decimal 16MB, NOT 16 MiB).
- **Training time limit:** Must run reproducibly in under 10 minutes on 8×H100 GPUs.
- **Evaluation time limit:** Must evaluate in under 10 minutes on 8×H100 (this is IN ADDITION to training time).
- **No external downloads or network calls during evaluation.** The artifact must be fully self-contained.
- **No access to training data during evaluation**, unless those bits are paid for within the 16MB limit.
- **Metric:** val_bpb (bits per byte) on the FineWeb validation set — lower is better.
- **Evaluation set:** 50,000 randomly-shuffled documents from FineWeb, concatenated into one token stream with BOS tokens at document boundaries.

## Test-Time Training (TTT) Rules — CRITICAL

TTT means updating model weights during evaluation using gradient descent on eval tokens. There are valid and invalid ways to do this. **Getting this wrong = immediate disqualification.**

### ALLOWED — Token-stream TTT (autoregressive)

You process the eval token stream left-to-right. For each token position t:
1. **First** score/predict token t (record the loss)
2. **Then** update weights using tokens ≤ t

The key rule: **you may only use preceding tokens that you have already been scored on.** You must never look ahead. You must not reorder the evaluation set.

Pseudocode of what is allowed:
```python
ttt_model = deepcopy(trained_model)
for chunk in chunks(eval_tokens, chunk_size):
    context = eval_tokens[max(0, chunk.end - context_len) : chunk.end]
    loss_per_token = ttt_model(context)
    accumulate(loss_per_token[chunk.start_in_context:])  # score FIRST
    sgd_step(ttt_model, mean(loss_per_token))             # train AFTER
```

### ALLOWED — Per-document TTT

Reset model to original checkpoint at each document boundary. Do TTT within each document independently. This is the most conservative and cleanest approach.

```python
for doc in split_at_BOS(eval_tokens):
    doc_model = deepcopy(trained_model)  # fresh copy per doc
    # do token-stream TTT within this doc only
```

### FORBIDDEN — Training on eval before scoring

**DO NOT** train on eval tokens and then score them afterward. This is training on the test set.

```python
# THIS IS CHEATING — DO NOT DO THIS
for epoch in range(num_epochs):
    for batch in batches(eval_tokens):
        sgd_step(model, loss(model, batch))   # train first
# then measure loss — INVALID, model has seen all answers
return mean(loss(model, batch) for batch in batches(eval_tokens))
```

### FORBIDDEN — Multi-epoch TTT on eval

Running multiple passes over eval tokens before scoring is equivalent to training on the test set. Do not do this.

### FORBIDDEN — Reordering the evaluation set

You must process the eval token stream in its original order. No shuffling, sorting, or reordering documents.

### Quick self-check for TTT validity

Ask yourself: "When my model predicts token t, has it ever seen token t or any token after t during training/adaptation?" If yes → INVALID. If no → valid.

## Submission Requirements

Every submission (PR to `openai/parameter-golf`) must include:

1. **README.md** — explains the approach in reasonable detail
2. **submission.json** — name, GitHub ID, val_bpb, metadata (see existing examples)
3. **train.log** — automatically produced training log
4. **train_gpt.py** — training script + dependencies; must compile and run within the records folder

Place submissions in `/records/track_10min_16mb/` for record attempts, or `/records/track_non_record_16mb/` for non-record/interesting approaches.

### SOTA Record Requirements

- Must beat existing SOTA by ≥ 0.005 nats
- Must show statistical significance at p < 0.01 (provide enough run logs)
- If you change the tokenizer, you must prove val_bpb is correctly calculated
- Must reproducibly run in under 10 minutes on 8×H100

## What Counts Toward the 16MB Limit

- Code bytes (the `train_gpt.py` script and dependencies)
- Compressed model bytes (int8 quantized + zlib compressed weights)
- Total = code bytes + compressed model bytes ≤ 16,000,000

## External Compute / Hyperparameter Tuning

- Tuning hyperparameters (learning rate, Adam params, etc.) across runs offline is fine.
- Brute-forcing seeds or sneaking in additional compute unfairly is not allowed.
- Use your best judgment. OpenAI reserves the right to disqualify anything not in the spirit of the challenge.

## Encouraged Techniques

The competition explicitly encourages:
- Test-time compute (done legally — see TTT rules above)
- Aggressive parameter tying / weight sharing
- Depth recurrence / looped transformers
- Low-rank training
- Quantization-aware training (QAT), bitnets, low precision
- Novel tokenizers (but be extra careful about val_bpb correctness)
- Long context evaluation
- Custom CUDA kernels
- EMA (Exponential Moving Average)
- Any creative approach that fits the constraints

## Common Mistakes to Avoid

1. **Illegal TTT** — the #1 cause of disqualification. Always score before you train on each chunk.
2. **Artifact too large** — check your int8+zlib compressed size + code size < 16MB.
3. **Script doesn't run** — test that train_gpt.py compiles and runs from the records folder.
4. **Missing submission files** — you need ALL of: README.md, submission.json, train.log, train_gpt.py.
5. **Non-reproducible results** — use fixed seeds, log everything, ensure consistency across runs.

## Reference Links

- GitHub repo: https://github.com/openai/parameter-golf
- Discord: OpenAI Discord → #parameter-golf-discussions
- TTT rules discussion: https://github.com/openai/parameter-golf/issues/402
