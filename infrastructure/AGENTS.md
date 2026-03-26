# Rules for writing Parameter Golf training scripts

You are writing training scripts for the OpenAI Parameter Golf challenge. Your script will be launched on a remote 1×H100 via Modal and must satisfy the contracts below for the infrastructure to work.

## What you're optimizing

Train a language model that minimizes bits-per-byte (BPB) on FineWeb validation data. Lower is better. The baseline scores ~1.2244 BPB.

Hard constraints: 16MB total artifact (int8+zlib compressed weights + source code bytes), 10 min training, 10 min eval.

## Runtime environment

- **GPU:** 1× H100 80GB
- **Launch command:** `torchrun --standalone --nproc_per_node=1 your_script.py`
- **Packages:** `torch` (2.6, CUDA), `numpy`, `sentencepiece`, `wandb`
- **Temp files:** write to `/tmp/` only

## Environment variables your script must read

```python
DATA_PATH = os.environ.get("DATA_PATH")           # REQUIRED — dir with .bin shards
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH") # REQUIRED — .model file
RUN_ID = os.environ.get("RUN_ID")                 # REQUIRED — unique run name
MAX_WALLCLOCK_SECONDS = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600))
VAL_LOSS_EVERY = int(os.environ.get("VAL_LOSS_EVERY", 200))
TRAIN_LOG_EVERY = int(os.environ.get("TRAIN_LOG_EVERY", 50))
```

Architecture env vars (`NUM_LAYERS`, `MODEL_DIM`, `NUM_HEADS`, `NUM_KV_HEADS`, `VOCAB_SIZE`, `ITERATIONS`, etc.) are also set but you may ignore them if your model is custom.

## Data format

Training shards: `{DATA_PATH}/fineweb_train_000000.bin`, `_000001.bin`, ...
Validation shards: `{DATA_PATH}/fineweb_val_000000.bin`, ...

```python
def load_shard(path):
    header = np.fromfile(path, dtype=np.uint32, count=256 // 4)
    assert header[0] == 20240520
    return np.fromfile(path, dtype=np.uint16, offset=256, count=int(header[1]))

train_files = sorted(glob.glob(os.path.join(DATA_PATH, "fineweb_train_*.bin")))
val_files = sorted(glob.glob(os.path.join(DATA_PATH, "fineweb_val_*.bin")))
```

Tokenizer: `spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)`, vocab size 1024.

## Required stdout output

The infrastructure parses your script's stdout for W&B logging and leaderboard results. You must print these exact formats.

### During training (every TRAIN_LOG_EVERY steps)

```
step:{step} train_loss:{loss:.4f}
```

Optional fields on the same line (parsed if present): `lr:{lr}`, `tok/s:{n}`, `step_time:{ms}`, `grad_norm:{n}`

### During validation (every VAL_LOSS_EVERY steps)

```
step:{step} val_loss:{loss:.4f} val_bpb:{bpb:.4f}
```

### End of script (mandatory — without these, the run has no result)

```python
print(f"Serialized model int8+zlib: {compressed_model_bytes} bytes")
print(f"Code size: {code_bytes} bytes")
print(f"Total submission size int8+zlib: {compressed_model_bytes + code_bytes} bytes")
print(f"final_int8_zlib_roundtrip val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f}")
print(f"final_int8_zlib_roundtrip_exact val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f}")
```

The `val_bpb` on the `final_int8_zlib_roundtrip_exact` line is **the** metric. If your script doesn't print this line, the run counts as failed.

## Wallclock enforcement

You must check elapsed time in your training loop and stop when the limit is reached:

```python
t_start = time.perf_counter()
for step in range(1, max_steps + 1):
    # ... train step ...
    if time.perf_counter() - t_start > MAX_WALLCLOCK_SECONDS:
        break
```

## End-of-script sequence

After training, your script must do this in order:

1. **Run final validation** on the full val split (pre-quantization)
2. **Quantize model to int8 + compress with zlib** — copy `quantize_state_dict_int8()` and `dequantize_state_dict_int8()` from the baseline `train_gpt.py`
3. **Print model/code/total byte counts** (see format above)
4. **Reload the quantized model** (roundtrip: compress → decompress → load weights back)
5. **Run validation again on the roundtripped model**
6. **Print the `final_int8_zlib_roundtrip_exact` line** with results from step 5

This roundtrip eval is critical — the pre-quantization BPB is not the official metric.

## Computing val_bpb

`val_bpb` is bits-per-byte of the original UTF-8 text, not token-level cross-entropy. Copy the BPB calculation from the baseline `train_gpt.py` — it uses lookup tables (`base_bytes_lut`, `has_leading_space_lut`, `is_boundary_token_lut`) built from the tokenizer. Do not try to reimplement this from scratch; copy the working code.

## Measuring code size

```python
with open(__file__, "r") as f:
    code_bytes = len(f.read().encode("utf-8"))
```

## Common mistakes

- Hardcoding data paths instead of reading `DATA_PATH` env var
- Forgetting the wallclock check — script runs until Modal kills it after 30 min
- Printing `val_loss` without `val_bpb` — the parser needs both on the same line
- Evaluating the fp32 model instead of the int8 roundtripped model for the final metric
- Exceeding 16,000,000 bytes total (model + code)
- Missing the `step:` field on validation log lines — W&B needs it for the x-axis

## Launching your script

```bash
python launch.py your_script.py --name your_idea        # full 10-min run
python launch.py your_script.py --name test --smoke      # quick 2-min smoke test
```

Runs are fire-and-forget — the command returns immediately. Results sync later with `python leaderboard.py --sync`.
