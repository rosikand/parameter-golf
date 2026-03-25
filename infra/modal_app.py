"""Modal app for Parameter Golf training runs."""

import os

import modal

from config import (
    DEFAULT_ENV,
    MODAL_APP_NAME,
    MODAL_VOLUME_MOUNT,
    MODAL_VOLUME_NAME,
    WANDB_ENTITY,
    WANDB_PROJECT,
)

volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)

_infra_dir = os.path.dirname(os.path.abspath(__file__))

train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "numpy",
        "tqdm",
        "wandb",
        "sentencepiece",
        "huggingface-hub",
        "datasets",
        "kernels",
        "tiktoken",
    )
    .add_local_file(
        os.path.join(_infra_dir, "log_parser.py"),
        remote_path="/root/log_parser.py",
    )
)


def make_train_app(gpus: int):
    """Create a Modal app + function for the given GPU count.

    Each GPU count gets its own app so we only build/register what we need.
    """
    gpu_spec = "H100" if gpus == 1 else f"H100:{gpus}"
    app = modal.App(f"{MODAL_APP_NAME}-{gpus}gpu")

    @app.function(
        image=train_image,
        gpu=gpu_spec,
        volumes={MODAL_VOLUME_MOUNT: volume},
        timeout=1200,
        secrets=[modal.Secret.from_name("wandb-secret")],
    )
    def train_run(
        script_content: str,
        env_vars: dict,
        run_name: str,
        wandb_enabled: bool = True,
    ):
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

        from log_parser import parse_line, extract_final_metrics

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

    return app, train_run
