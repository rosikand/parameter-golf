# Context Prompt: OpenAI Parameter Golf Challenge

## The Challenge

I'm competing in OpenAI's "Parameter Golf" challenge (https://github.com/openai/parameter-golf). The goal is to train the best language model under these constraints:

- **16 MB artifact limit** (weights + training code combined, int8 quantized + zlib compressed)
- **10-minute training budget** on 8×H100 GPUs
- **10-minute evaluation budget** (separate from training)
- **Metric:** val_bpb (bits per byte) on a fixed FineWeb validation set — lower is better
- **Tokenizer-agnostic:** you can bring your own tokenizer; scoring is on raw byte compression
- **Deadline:** April 30, 2026

The current leaderboard baseline is **1.2244 val_bpb** (naive 9-layer, 512-dim, 1024-vocab GPT with tied embeddings). A 4-hour unlimited-compute run achieved **1.2074 val_bpb**.

New SOTA submissions must beat the current best by at least 0.005 nats at p < 0.01 significance.

## The Baseline Architecture

The default `train_gpt.py` uses:
- 9 transformer blocks, width 512, 8 attention heads, 4 KV heads (GQA)
- 2× MLP expansion with relu² activation
- Vocab size 1024 (SentencePiece BPE), sequence length 1024
- Tied embeddings (tok_emb shared with lm_head)
- RoPE positional encoding
- RMSNorm, logit softcapping at 30.0
- Skip connections with learned residual mixing (encoder-decoder style, similar to U-Net)
- Muon optimizer for matrix params, Adam for embeddings/scalars
- 524,288 tokens per batch, 20,000 max iterations with 600s wallclock cap
- Post-training: int8 quantization (per-row for matrices, per-tensor for vectors) + zlib compression

All hyperparameters are configurable via environment variables (NUM_LAYERS, MODEL_DIM, NUM_HEADS, LEARNING_RATE, etc.).

## The Repo Structure

```
parameter-golf/
├── train_gpt.py                    # main training script (~800 lines)
├── train_gpt_mlx.py                # Apple Silicon variant
├── requirements.txt                # numpy, torch==2.10, sentencepiece, kernels, etc.
├── data/
│   ├── cached_challenge_fineweb.py # downloads dataset from HuggingFace
│   └── tokenizer_specs.json
└── records/
    ├── track_10min_16mb/           # official leaderboard submissions
    └── track_non_record_16mb/      # unlimited compute / experimental submissions
```

Training is launched via torchrun:
```bash
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
RUN_ID=my_run \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The dataset is ~80 shards / ~8B tokens, downloaded once via `python3 data/cached_challenge_fineweb.py --variant sp1024`.

## Key Challenge Details

- Submissions are GitHub PRs adding a folder to `/records/track_10min_16mb/`
- Must include: README.md, submission.json, train log, train_gpt.py
- The 16MB limit is decimal (16,000,000 bytes), not 16 MiB
- Evaluation can use any sequence length; no training data access during eval unless you pay for those bits in the <16MB limit
- OpenAI is offering $1M in compute credits via RunPod to non-employee participants
- The official RunPod template has all dependencies pre-installed: https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th
- `torch==2.10` is the required PyTorch version

## My Intended Workflow

I want to iterate fast by making changes locally and running them remotely on H100s. My plan:

1. **Compute:** Use Modal (modal.com) or RunPod for GPU access. I want to edit `train_gpt.py` locally and fire off runs remotely with a single command — no SSH, no managing instances.
2. **Parallelism:** I want to launch 50+ variant runs simultaneously (different hyperparams, architectures) and collect all results.
3. **Tracking:** W&B for step-level metrics (train_loss curves, val_bpb over time). Also an internal leaderboard (markdown table) to rank all my experiments.
4. **Cost efficiency:** Do fast iteration on 1×H100 (~$0.50/run), only use 8×H100 (~$4/run) for final submission-grade timing.

## Current State

I haven't set up any infrastructure yet. No Modal account configured, no W&B project, no RunPod pods. I have the repo cloned locally and have read through the rules, baseline code, and training script.

## What I Need Help With

[INSERT YOUR SPECIFIC QUESTION HERE]
