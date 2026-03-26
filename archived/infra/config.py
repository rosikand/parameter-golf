"""Shared configuration for Parameter Golf infrastructure."""

import os

# Modal
MODAL_APP_NAME = "parameter-golf"
MODAL_VOLUME_NAME = "parameter-golf-data"
MODAL_VOLUME_MOUNT = "/data"

# W&B
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "parameter-golf")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", None)

# Default env vars for every training run (paths inside Modal container)
DEFAULT_ENV = {
    "DATA_PATH": f"{MODAL_VOLUME_MOUNT}/datasets/fineweb10B_sp1024",
    "TOKENIZER_PATH": f"{MODAL_VOLUME_MOUNT}/tokenizers/fineweb_1024_bpe.model",
    "VOCAB_SIZE": "1024",
    "MAX_WALLCLOCK_SECONDS": "600",
}

# Cost estimate (USD per GPU-hour on Modal H100)
H100_COST_PER_HOUR = 3.95

# Paths
INFRA_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(INFRA_DIR)
LEADERBOARD_CSV = os.path.join(INFRA_DIR, "leaderboard.csv")
LEADERBOARD_MD = os.path.join(INFRA_DIR, "LEADERBOARD.md")
