"""
Modal app for Parameter Golf training runs.

Fire-and-forget: launch runs, close your laptop, sync results later.

    python launch.py my_script.py --name idea_1
    python launch.py my_script.py --name idea_2
    # close laptop, go to sleep
    python leaderboard.py --sync   # pull results from Modal Volume
    python leaderboard.py          # view standings
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal configuration
# ---------------------------------------------------------------------------

APP_NAME = "parameter-golf"
VOLUME_NAME = "parameter-golf-data"
GPU_COUNT = int(os.environ.get("PGOLF_GPU_COUNT", "1"))
GPU = f"H100:{GPU_COUNT}" if GPU_COUNT > 1 else "H100"
TIMEOUT_SECONDS = 60 * 60  # 1 hour hard cap

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.6.0",
        "numpy",
        "sentencepiece",
        "huggingface_hub",
        "datasets",
        "tqdm",
        "wandb",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .env({"HF_HOME": "/cache/hf"})
)

app = modal.App(APP_NAME, image=image)
data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

DATA_MOUNT = "/data"
RUNS_DIR = f"{DATA_MOUNT}/runs"


# ---------------------------------------------------------------------------
# One-time data setup
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_MOUNT: data_volume},
    timeout=60 * 30,
    cpu=4,
    memory=16384,
)
def setup_data(variant: str = "sp1024", train_shards: int = 80):
    """Download FineWeb challenge data into the persistent Modal volume."""
    import shutil

    marker = Path(DATA_MOUNT) / f".done_{variant}_{train_shards}"
    if marker.exists():
        print(f"Data already cached for {variant} with {train_shards} shards.")
        return

    repo_dir = Path("/tmp/parameter-golf")
    if repo_dir.exists():
        shutil.rmtree(repo_dir)

    subprocess.run(
        ["git", "clone", "--depth=1", "https://github.com/openai/parameter-golf.git", str(repo_dir)],
        check=True,
    )

    env = os.environ.copy()
    env["HF_HOME"] = "/cache/hf"

    subprocess.run(
        [sys.executable, str(repo_dir / "data" / "cached_challenge_fineweb.py"),
         "--variant", variant, "--train-shards", str(train_shards)],
        cwd=str(repo_dir), env=env, check=True,
    )

    src_datasets = repo_dir / "data" / "datasets"
    src_tokenizers = repo_dir / "data" / "tokenizers"
    dst_datasets = Path(DATA_MOUNT) / "datasets"
    dst_tokenizers = Path(DATA_MOUNT) / "tokenizers"

    if src_datasets.exists():
        shutil.copytree(src_datasets, dst_datasets, dirs_exist_ok=True)
    if src_tokenizers.exists():
        shutil.copytree(src_tokenizers, dst_tokenizers, dirs_exist_ok=True)

    marker.touch()
    print(f"Data cached successfully for {variant} with {train_shards} shards.")


# ---------------------------------------------------------------------------
# Training function (runs on GPU, saves results to Volume)
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_MOUNT: data_volume},
    gpu=GPU,
    timeout=TIMEOUT_SECONDS,
    memory=65536,
)
def train(
    train_script_code: str,
    run_id: str,
    variant: str = "sp1024",
    extra_env: dict | None = None,
    wandb_project: str = "parameter-golf",
    wandb_api_key: str = "",
    max_wallclock: int = 600,
    val_loss_every: int = 200,
    train_log_every: int = 50,
    train_batch_tokens: int = 524_288,
    iterations: int = 20_000,
    num_layers: int = 9,
    model_dim: int = 512,
    num_heads: int = 8,
    num_kv_heads: int = 4,
    mlp_mult: int = 2,
    vocab_size: int = 1024,
    seq_len: int = 1024,
    tie_embeddings: int = 1,
    gpu_count: int = 1,
):
    """Run a training job. Results saved to Volume."""
    import wandb

    # Create run output directory on the Volume
    run_dir = Path(RUNS_DIR) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write a status file so sync can see in-progress runs
    _write_status(run_dir, "running")

    work_dir = Path("/tmp/train_run")
    work_dir.mkdir(parents=True, exist_ok=True)

    script_path = work_dir / "train_gpt.py"
    script_path.write_text(train_script_code)

    # Resolve data paths
    dataset_name = f"fineweb10B_{variant}"
    data_path = f"{DATA_MOUNT}/datasets/{dataset_name}"
    tokenizer_path = f"{DATA_MOUNT}/tokenizers/fineweb_{variant.replace('sp', '')}_bpe.model"

    # Build environment
    env = os.environ.copy()
    env.update({
        "RUN_ID": run_id,
        "DATA_PATH": data_path,
        "TOKENIZER_PATH": tokenizer_path,
        "VOCAB_SIZE": str(vocab_size),
        "NUM_LAYERS": str(num_layers),
        "MODEL_DIM": str(model_dim),
        "NUM_HEADS": str(num_heads),
        "NUM_KV_HEADS": str(num_kv_heads),
        "MLP_MULT": str(mlp_mult),
        "TIE_EMBEDDINGS": str(tie_embeddings),
        "TRAIN_BATCH_TOKENS": str(train_batch_tokens),
        "TRAIN_SEQ_LEN": str(seq_len),
        "MAX_WALLCLOCK_SECONDS": str(max_wallclock),
        "VAL_LOSS_EVERY": str(val_loss_every),
        "TRAIN_LOG_EVERY": str(train_log_every),
        "ITERATIONS": str(iterations),
        "OMP_NUM_THREADS": "1",
    })
    if extra_env:
        env.update(extra_env)

    # Initialize W&B
    wandb_run = None
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
        wandb_run = wandb.init(
            project=wandb_project,
            name=run_id,
            config={
                "variant": variant, "vocab_size": vocab_size,
                "num_layers": num_layers, "model_dim": model_dim,
                "num_heads": num_heads, "num_kv_heads": num_kv_heads,
                "mlp_mult": mlp_mult, "train_batch_tokens": train_batch_tokens,
                "seq_len": seq_len, "max_wallclock": max_wallclock,
                "iterations": iterations,
            },
        )

    # Launch training
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", f"--nproc_per_node={gpu_count}",
        str(script_path),
    ]

    print(f"[parameter-golf] Starting training: {run_id}")
    print(f"[parameter-golf] Data: {data_path}")
    print(f"[parameter-golf] Config: {num_layers}L x {model_dim}D, vocab={vocab_size}, {gpu_count}xH100")
    print("=" * 80)

    t_start = time.time()
    log_lines = []

    process = subprocess.Popen(
        cmd, cwd=str(work_dir), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    for line in process.stdout:
        line = line.rstrip()
        log_lines.append(line)
        print(line, flush=True)
        if wandb_run:
            _log_to_wandb(wandb_run, line)

    process.wait()
    elapsed = time.time() - t_start

    print("=" * 80)
    print(f"[parameter-golf] Finished in {elapsed:.1f}s (exit code: {process.returncode})")

    # Parse results
    result = _parse_results(log_lines, run_id, elapsed)
    result["exit_code"] = process.returncode

    if wandb_run:
        wandb_run.summary.update(result)
        wandb_run.finish()

    # Save results + log to Volume (persists after function exits)
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    (run_dir / "train.log").write_text("\n".join(log_lines))
    _write_status(run_dir, "done")

    print(f"[parameter-golf] Results saved to Volume: {run_dir}")


# ---------------------------------------------------------------------------
# Sync: read all results from Volume
# ---------------------------------------------------------------------------
@app.function(volumes={DATA_MOUNT: data_volume})
def sync_results() -> list[dict]:
    """Read all run results from the Volume."""
    data_volume.reload()
    runs_path = Path(RUNS_DIR)
    if not runs_path.exists():
        return []

    results = []
    for run_dir in sorted(runs_path.iterdir()):
        if not run_dir.is_dir():
            continue
        result_file = run_dir / "result.json"
        status_file = run_dir / "status"
        status = status_file.read_text().strip() if status_file.exists() else "unknown"

        if result_file.exists():
            result = json.loads(result_file.read_text())
            result["status"] = status
            results.append(result)
        else:
            # Run is still in progress or failed before writing results
            results.append({
                "run_id": run_dir.name,
                "status": status,
                "val_bpb_int8": None,
            })

    return results


@app.function(volumes={DATA_MOUNT: data_volume})
def get_run_log(run_id: str) -> str:
    """Fetch the full training log for a specific run."""
    data_volume.reload()
    log_path = Path(RUNS_DIR) / run_id / "train.log"
    if log_path.exists():
        return log_path.read_text()
    return f"No log found for run: {run_id}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_status(run_dir: Path, status: str):
    (run_dir / "status").write_text(status)


def _log_to_wandb(wandb_run, line: str):
    step_match = re.search(r"step:(\d+)", line)
    train_loss_match = re.search(r"train_loss:([\d.]+)", line)

    if step_match and train_loss_match:
        step = int(step_match.group(1))
        metrics = {"train/loss": float(train_loss_match.group(1)), "step": step}
        for key, pattern in [
            ("train/lr", r"lr:([\d.eE+-]+)"),
            ("train/tokens_per_sec", r"tok/s:([\d.]+)"),
            ("train/step_ms", r"step_time:([\d.]+)"),
            ("train/grad_norm", r"grad_norm:([\d.]+)"),
        ]:
            m = re.search(pattern, line)
            if m:
                metrics[key] = float(m.group(1))
        wandb_run.log(metrics, step=step)

    val_loss_match = re.search(r"val_loss:([\d.]+)", line)
    val_bpb_match = re.search(r"val_bpb:([\d.]+)", line)
    if val_loss_match and val_bpb_match and step_match:
        step = int(step_match.group(1))
        wandb_run.log({
            "val/loss": float(val_loss_match.group(1)),
            "val/bpb": float(val_bpb_match.group(1)),
        }, step=step)

    if "final_int8_zlib_roundtrip " in line and val_loss_match and val_bpb_match:
        wandb_run.log({
            "final/val_loss_int8": float(val_loss_match.group(1)),
            "final/val_bpb_int8": float(val_bpb_match.group(1)),
        })


def _parse_results(log_lines: list[str], run_id: str, elapsed: float) -> dict:
    result = {
        "run_id": run_id,
        "elapsed_seconds": round(elapsed, 1),
        "val_loss": None, "val_bpb": None,
        "val_loss_int8": None, "val_bpb_int8": None,
        "model_bytes_int8_zlib": None, "code_bytes": None,
        "total_bytes": None, "steps_completed": None,
    }
    full_log = "\n".join(log_lines)

    m = re.search(r"final_int8_zlib_roundtrip_exact val_loss:([\d.]+) val_bpb:([\d.]+)", full_log)
    if m:
        result["val_loss_int8"] = float(m.group(1))
        result["val_bpb_int8"] = float(m.group(2))

    for line in reversed(log_lines):
        if "val_loss:" in line and "final_int8" not in line and "roundtrip" not in line:
            m_loss = re.search(r"val_loss:([\d.]+)", line)
            m_bpb = re.search(r"val_bpb:([\d.]+)", line)
            if m_loss and m_bpb:
                result["val_loss"] = float(m_loss.group(1))
                result["val_bpb"] = float(m_bpb.group(1))
            break

    for pattern, key in [
        (r"Serialized model int8\+zlib: (\d+) bytes", "model_bytes_int8_zlib"),
        (r"Code size: (\d+) bytes", "code_bytes"),
        (r"Total submission size int8\+zlib: (\d+) bytes", "total_bytes"),
    ]:
        m = re.search(pattern, full_log)
        if m:
            result[key] = int(m.group(1))

    step_matches = re.findall(r"step:(\d+)", full_log)
    if step_matches:
        result["steps_completed"] = max(int(s) for s in step_matches)

    return result


# ---------------------------------------------------------------------------
# Local entrypoint: fire-and-forget
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    train_script: str = "./train_gpt.py",
    run_id: str = "",
    variant: str = "sp1024",
    max_wallclock: int = 600,
    val_loss_every: int = 200,
    wandb_project: str = "parameter-golf",
    vocab_size: int = 1024,
    num_layers: int = 9,
    model_dim: int = 512,
    num_heads: int = 8,
    num_kv_heads: int = 4,
    iterations: int = 20000,
    skip_data_check: bool = False,
    gpu_count: int = 1,
):
    """Launch a Parameter Golf training run on Modal (fire-and-forget)."""
    from datetime import datetime

    if not run_id:
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    script_path = Path(train_script)
    if not script_path.exists():
        print(f"Error: training script not found: {script_path}")
        sys.exit(1)

    train_script_code = script_path.read_text()

    # Ensure data is cached (blocking — fast if already done)
    if not skip_data_check:
        print(f"[local] Checking data cache...")
        setup_data.remote(variant=variant)

    wandb_key = os.environ.get("WANDB_API_KEY", "")

    # Fire and forget — .spawn() returns immediately
    print(f"[local] Spawning run: {run_id}")
    print(f"[local] Script: {script_path} ({len(train_script_code)} bytes)")
    print(f"[local] Config: {num_layers}L x {model_dim}D, vocab={vocab_size}, {gpu_count}xH100, wallclock={max_wallclock}s")
    if wandb_key:
        print(f"[local] W&B logging enabled (project: {wandb_project})")

    fc = train.spawn(
        train_script_code=train_script_code,
        run_id=run_id,
        variant=variant,
        wandb_project=wandb_project,
        wandb_api_key=wandb_key,
        max_wallclock=max_wallclock,
        val_loss_every=val_loss_every,
        vocab_size=vocab_size,
        num_layers=num_layers,
        model_dim=model_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        iterations=iterations,
        gpu_count=gpu_count,
    )

    print(f"[local] Run spawned! Function call ID: {fc.object_id}")
    print(f"[local] View logs: modal.com/apps → {APP_NAME}")
    if wandb_key:
        print(f"[local] W&B: wandb.ai/{wandb_project}")
    print(f"[local] Sync results later: python leaderboard.py --sync")