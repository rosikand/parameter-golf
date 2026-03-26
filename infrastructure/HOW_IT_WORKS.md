# How This Works

## The Problem

You're competing in OpenAI's Parameter Golf challenge. You need to train lots of small language models on a GPU, trying different ideas to get the lowest "bits per byte" score. Each experiment takes ~20 minutes (10 min training + 10 min eval) on an H100 GPU. You want to launch a bunch of experiments before bed and see the results in the morning.

## The Setup

You have 5 files in a folder on your laptop:

```
parameter-golf-infra/
  modal_train.py        # the Modal app (runs in the cloud)
  launch.py             # CLI to kick off runs
  leaderboard.py        # view results
  AGENT_INSTRUCTIONS.md # instructions for AI agents writing experiments
  README.md
```

And you write training scripts — Python files that define a model, train it, and evaluate it. These can be forks of the baseline `train_gpt.py` or completely custom.

## The Workflow

```bash
python launch.py my_idea.py --name cool_idea
```

This returns in about 5 seconds. The GPU job is now running in the cloud. You can launch more:

```bash
python launch.py idea_2.py --name deeper_model
python launch.py idea_3.py --name bigger_vocab
```

Close your laptop. Go to sleep.

In the morning:

```bash
python leaderboard.py --sync    # pull results from the cloud
python leaderboard.py           # see which idea won
```

While runs are active, you can also watch them live on W&B (Weights & Biases) from your phone or any browser.

## What Happens Under the Hood

Here's the full chain of events when you run `python launch.py my_idea.py --name cool_idea`:

### Step 1: launch.py (your laptop, ~1 second)

`launch.py` is just a wrapper that builds a `modal run modal_train.py` command with your arguments and executes it. It translates nice flags like `--name`, `--smoke`, `--num-layers` into the flags that `modal_train.py` expects.

### Step 2: modal_train.py local entrypoint (your laptop, ~3 seconds)

Modal runs the `main()` function locally on your machine. This function:

1. Reads your training script file into a string (the raw Python source code)
2. Reads your `WANDB_API_KEY` from your shell environment
3. Calls `setup_data.remote()` — this checks if the FineWeb dataset is already cached on the Modal Volume (it is, after your first run), so it returns instantly
4. Calls `train.spawn(...)` — this is the key part

### Step 3: train.spawn() (your laptop → Modal cloud, instant)

`.spawn()` is Modal's fire-and-forget API. It sends a message to Modal saying "run this function with these arguments on an H100" and immediately returns a function call ID. Your laptop is now done — the terminal prints the ID and exits. Even if you close your laptop, the cloud job continues.

What gets sent to Modal: your training script as a string, the run ID, hyperparameters, your W&B key, and other config. Not the dataset — that's already on the Volume.

### Step 4: Modal spins up a container (Modal cloud, ~30 seconds)

Modal finds an available H100 GPU and boots a container with the image we defined: Ubuntu + Python 3.11 + PyTorch (CUDA-enabled) + numpy + sentencepiece + wandb. It mounts the Modal Volume at `/data/`, which contains the pre-downloaded FineWeb dataset shards.

### Step 5: train() runs (Modal cloud, ~20 minutes)

The `train()` function executes inside the container:

1. **Writes a status file** to the Volume: `/data/runs/cool_idea_0324_2130/status` → "running"
2. **Writes your training script** to `/tmp/train_run/train_gpt.py`
3. **Sets environment variables** — `DATA_PATH=/data/datasets/fineweb10B_sp1024`, `TOKENIZER_PATH=...`, `MAX_WALLCLOCK_SECONDS=600`, all the hyperparameters
4. **Initializes W&B** with your API key and starts a new run
5. **Launches torchrun** — `torchrun --standalone --nproc_per_node=1 /tmp/train_run/train_gpt.py`. This is the same command you'd run if you SSH'd into a GPU box manually
6. **Streams stdout** — every line your script prints gets captured. Lines matching `step:100 train_loss:3.45` get parsed and sent to W&B in real-time. This is how you get live training curves
7. **Waits for training to finish** — your script runs until it hits the wallclock limit or max iterations
8. **Parses final results** — scans the log for lines like `final_int8_zlib_roundtrip_exact val_bpb:1.2244` and extracts the official metric
9. **Saves everything to the Volume:**
   - `/data/runs/cool_idea_0324_2130/result.json` — the parsed metrics
   - `/data/runs/cool_idea_0324_2130/train.log` — the full stdout log
   - `/data/runs/cool_idea_0324_2130/status` → "done"
10. **Finishes the W&B run** with final summary metrics

The container shuts down. You stop paying for the GPU.

### Step 6: You sync results (your laptop, next morning)

```bash
python leaderboard.py --sync
```

This calls `sync_results.remote()`, which runs a small (no-GPU) function on Modal that reads every `/data/runs/*/result.json` from the Volume and returns them as a list. Back on your laptop, `leaderboard.py` merges these into `leaderboard.json` and prints the standings.

## The Data Flow

```
Your laptop                          Modal cloud
───────────                          ───────────

my_idea.py (source code)  ─────→    Written to /tmp/ on container
                                     │
                                     ▼
                                    torchrun train_gpt.py
                                     │
                                     ├──→ Reads /data/datasets/*.bin (from Volume)
                                     ├──→ Prints step:N train_loss:X to stdout
                                     │     │
                                     │     └──→ Parsed → sent to W&B (live)
                                     │
                                     └──→ Prints final_int8_zlib_roundtrip_exact ...
                                           │
                                           ▼
                                    result.json + train.log
                                    written to /data/runs/RUN_ID/ (Volume)

leaderboard.py --sync     ←─────    Reads /data/runs/*/result.json
       │
       ▼
leaderboard.json (local)
```

## The Modal Volume

The Volume is a persistent network disk that survives across function calls. It contains:

```
/data/
  datasets/
    fineweb10B_sp1024/
      fineweb_train_000000.bin   # ~100M tokens each, 80 shards
      fineweb_train_000001.bin
      ...
      fineweb_val_000000.bin     # validation split
      ...
  tokenizers/
    fineweb_1024_bpe.model       # SentencePiece tokenizer
  runs/
    cool_idea_0324_2130/
      status                     # "running" or "done"
      result.json                # parsed metrics
      train.log                  # full stdout
    deeper_model_0324_2131/
      ...
```

The dataset shards are downloaded once (`modal run modal_train.py::setup_data`) and reused by every training run. Results accumulate in `/data/runs/`.

## W&B Integration

Your training script doesn't need to know about W&B. The Modal wrapper does it all:

1. Your script prints `step:100 train_loss:3.45` to stdout (which it already does naturally)
2. The wrapper regex-matches these lines and calls `wandb.log({"train/loss": 3.45}, step=100)`
3. W&B receives the data and updates the dashboard in real-time

This means you can watch training curves live from your phone at wandb.ai, even after closing your laptop.

## Cost

Modal bills per-second of GPU time. An H100 is roughly $3.95/hour. A typical run (10 min train + 10 min eval + overhead) costs about $1.30. A 2-minute smoke test costs about $0.25. If you launch 10 experiments overnight, that's ~$13.
