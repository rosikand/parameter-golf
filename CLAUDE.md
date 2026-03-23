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
- `context/` — LLM handoff prompts summarizing the challenge for AI assistants.
- `notes/` — working notes (compute credit request, etc.).

## Commands

```bash
# Local smoke test (MLX, Apple Silicon)
RUN_ID=mlx_smoke ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 python3 train_gpt_mlx.py

# Remote training (CUDA, single GPU)
RUN_ID=baseline DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 \
  torchrun --standalone --nproc_per_node=1 train_gpt.py

# Scout
python scout/scout.py              # full run: fetch + analyze + briefing
python scout/scout.py --fetch-only # just fetch GitHub data
python scout/scout.py --brief-only # regenerate briefing from cached state
```

## Environment variables

- `GITHUB_TOKEN` — raises GitHub API rate limit (60 → 5000/hr) for scout
- `ANTHROPIC_API_KEY` — required for scout analysis/briefing

## Key constraints

- Artifact = code bytes + int8-quantized zlib-compressed weights, must be < 16,000,000 bytes
- New SOTA must beat current best by ≥ 0.005 nats at p < 0.01
- No network calls allowed during evaluation; artifact must be self-contained
