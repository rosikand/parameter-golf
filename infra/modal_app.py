"""Modal app for Parameter Golf training runs."""

import os

import modal

# Import config values — these are used at module level for Modal decorators
# When running locally (via run.py/sweep.py), infra/ is on sys.path
from config import (
    DEFAULT_ENV,
    MODAL_APP_NAME,
    MODAL_VOLUME_MOUNT,
    MODAL_VOLUME_NAME,
    WANDB_ENTITY,
    WANDB_PROJECT,
)

app = modal.App(MODAL_APP_NAME)

volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)

# Mount log_parser.py into the container at /root/log_parser.py
_infra_dir = os.path.dirname(os.path.abspath(__file__))
log_parser_mount = modal.Mount.from_local_file(
    os.path.join(_infra_dir, "log_parser.py"),
    remote_path="/root/log_parser.py",
)

image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/pytorch:24.04-py3",
        add_python="3.11",
    )
    .pip_install(
        "wandb",
        "sentencepiece",
        "huggingface-hub",
        "datasets",
        "kernels",
        "tiktoken",
    )
)


@app.function(
    image=image,
    volumes={MODAL_VOLUME_MOUNT: volume},
    timeout=600,
    mounts=[log_parser_mount],
)
def ensure_data(
    data_download_script: str,
    variant: str = "sp1024",
    train_shards: int = 80,
):
    """Download dataset to persistent volume. Run once."""
    import subprocess

    # Write the download script to /tmp
    with open("/tmp/cached_challenge_fineweb.py", "w") as f:
        f.write(data_download_script)

    # Write a wrapper that patches ROOT to point at the volume
    wrapper = f"""\
import sys
sys.path.insert(0, "/tmp")
from pathlib import Path
import cached_challenge_fineweb as dl

dl.ROOT = Path("{MODAL_VOLUME_MOUNT}")
dl.DATASETS_DIR = dl.ROOT / "datasets"
dl.TOKENIZERS_DIR = dl.ROOT / "tokenizers"
dl.DATASETS_DIR.mkdir(parents=True, exist_ok=True)
dl.TOKENIZERS_DIR.mkdir(parents=True, exist_ok=True)

sys.argv = ["dl", "--variant", "{variant}", "--train-shards", str({train_shards})]
dl.main()
"""
    with open("/tmp/run_download.py", "w") as f:
        f.write(wrapper)

    print(f"Downloading dataset variant={variant} train_shards={train_shards}...")
    result = subprocess.run(
        ["python", "/tmp/run_download.py"],
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    if result.returncode != 0:
        raise RuntimeError(f"Data download failed with exit code {result.returncode}")

    volume.commit()
    print("Data download complete and committed to volume.")


# We define separate functions for each GPU count since Modal requires
# GPU spec at decoration time.

def _train_run_impl(
    script_content: str,
    env_vars: dict,
    run_name: str,
    gpus: int,
    wandb_enabled: bool,
):
    """Shared training logic for all GPU counts."""
    import subprocess
    import sys
    import time

    sys.path.insert(0, "/root")

    # Write the training script
    with open("/tmp/train_script.py", "w") as f:
        f.write(script_content)

    # Reload volume to see latest data
    volume.reload()

    # Check data exists
    data_path = env_vars.get("DATA_PATH", DEFAULT_ENV["DATA_PATH"])
    if not os.path.isdir(data_path):
        print(f"WARNING: Data directory {data_path} not found on volume.")
        print("Run 'python infra/run.py --ensure-data' first.")
        return {"run_name": run_name, "exit_code": -1, "error": "data_not_found"}

    # Build environment
    run_env = {**os.environ}
    run_env.update(DEFAULT_ENV)
    run_env.update(env_vars)
    run_env["RUN_ID"] = run_name
    run_env["PYTHONUNBUFFERED"] = "1"

    # Initialize W&B
    wandb_run = None
    if wandb_enabled and os.environ.get("WANDB_API_KEY"):
        try:
            import wandb
            wandb_run = wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                name=run_name,
                config={k: v for k, v in run_env.items()
                        if k in DEFAULT_ENV or k in env_vars},
            )
        except Exception as e:
            print(f"WARNING: W&B init failed: {e}. Continuing without W&B.")

    # Import log parser (mounted at /root/)
    from log_parser import parse_line, extract_final_metrics

    # Launch training
    cmd = [
        "torchrun", "--standalone",
        f"--nproc_per_node={gpus}",
        "/tmp/train_script.py",
    ]
    print(f"Launching: {' '.join(cmd)}")
    print(f"GPU count: {gpus}")
    print(f"Run name: {run_name}")

    all_lines = []
    start_time = time.time()

    proc = subprocess.Popen(
        cmd,
        env=run_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in proc.stdout:
        line = line.rstrip("\n")
        print(line)
        all_lines.append(line)

        if wandb_run:
            parsed = parse_line(line)
            if parsed:
                t = parsed["type"]
                if t == "train":
                    wandb_run.log({
                        "train/loss": parsed["train_loss"],
                        "train/step_time_ms": parsed["step_avg_ms"],
                    }, step=parsed["step"])
                elif t == "val":
                    wandb_run.log({
                        "val/loss": parsed["val_loss"],
                        "val/bpb": parsed["val_bpb"],
                    }, step=parsed["step"])
                elif t in ("final_int8", "final_int8_exact"):
                    wandb_run.summary["final_int8_val_bpb"] = parsed["val_bpb"]
                    wandb_run.summary["final_int8_val_loss"] = parsed["val_loss"]
                elif t == "final_ttt":
                    wandb_run.summary["final_ttt_val_bpb"] = parsed["val_bpb"]
                    wandb_run.summary["final_ttt_val_loss"] = parsed["val_loss"]
                elif t == "submission_size":
                    wandb_run.summary["submission_bytes"] = parsed["total_bytes"]

    proc.wait()
    elapsed = time.time() - start_time

    metrics = extract_final_metrics(all_lines)
    result = {
        "run_name": run_name,
        "exit_code": proc.returncode,
        "elapsed_s": round(elapsed, 1),
        **metrics,
    }

    if wandb_run:
        wandb_run.summary["exit_code"] = proc.returncode
        wandb_run.summary["elapsed_s"] = round(elapsed, 1)
        wandb_run.finish(exit_code=proc.returncode)

    return result


@app.function(
    image=image,
    gpu="H100",
    volumes={MODAL_VOLUME_MOUNT: volume},
    timeout=1200,
    secrets=[modal.Secret.from_name("wandb-secret", required=False)],
    mounts=[log_parser_mount],
)
def train_run_1gpu(
    script_content: str,
    env_vars: dict,
    run_name: str,
    wandb_enabled: bool = True,
):
    return _train_run_impl(script_content, env_vars, run_name, gpus=1, wandb_enabled=wandb_enabled)


@app.function(
    image=image,
    gpu=modal.gpu.H100(count=2),
    volumes={MODAL_VOLUME_MOUNT: volume},
    timeout=1200,
    secrets=[modal.Secret.from_name("wandb-secret", required=False)],
    mounts=[log_parser_mount],
)
def train_run_2gpu(
    script_content: str,
    env_vars: dict,
    run_name: str,
    wandb_enabled: bool = True,
):
    return _train_run_impl(script_content, env_vars, run_name, gpus=2, wandb_enabled=wandb_enabled)


@app.function(
    image=image,
    gpu=modal.gpu.H100(count=4),
    volumes={MODAL_VOLUME_MOUNT: volume},
    timeout=1200,
    secrets=[modal.Secret.from_name("wandb-secret", required=False)],
    mounts=[log_parser_mount],
)
def train_run_4gpu(
    script_content: str,
    env_vars: dict,
    run_name: str,
    wandb_enabled: bool = True,
):
    return _train_run_impl(script_content, env_vars, run_name, gpus=4, wandb_enabled=wandb_enabled)


@app.function(
    image=image,
    gpu=modal.gpu.H100(count=8),
    volumes={MODAL_VOLUME_MOUNT: volume},
    timeout=1200,
    secrets=[modal.Secret.from_name("wandb-secret", required=False)],
    mounts=[log_parser_mount],
)
def train_run_8gpu(
    script_content: str,
    env_vars: dict,
    run_name: str,
    wandb_enabled: bool = True,
):
    return _train_run_impl(script_content, env_vars, run_name, gpus=8, wandb_enabled=wandb_enabled)


# Dispatch helper
_GPU_DISPATCH = {
    1: train_run_1gpu,
    2: train_run_2gpu,
    4: train_run_4gpu,
    8: train_run_8gpu,
}


def get_train_fn(gpus: int):
    """Get the Modal function for the given GPU count."""
    fn = _GPU_DISPATCH.get(gpus)
    if fn is None:
        raise ValueError(f"Unsupported GPU count: {gpus}. Must be 1, 2, 4, or 8.")
    return fn
