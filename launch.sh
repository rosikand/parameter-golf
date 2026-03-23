#!/bin/bash
# launch.sh — Parameter Golf training launcher
# Usage:
#   ./launch.sh                    # baseline run, 1 GPU
#   ./launch.sh --gpus 8           # 8 GPU submission run  
#   ./launch.sh --wandb            # enable wandb logging
#   ./launch.sh --shards 80        # full dataset
#   ./launch.sh --name my_experiment --wandb --gpus 1

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
GPUS=1
RUN_NAME="baseline"
USE_WANDB=0
WANDB_PROJECT="parameter-golf"
TRAIN_SHARDS=""
DATA_PATH="./data/datasets/fineweb10B_sp1024"
TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"
VOCAB_SIZE=1024
VAL_LOSS_EVERY=200
TRAIN_LOG_EVERY=50
MAX_WALLCLOCK=600
EXTRA_ENV=""

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)       GPUS="$2"; shift 2 ;;
        --name)       RUN_NAME="$2"; shift 2 ;;
        --wandb)      USE_WANDB=1; shift ;;
        --shards)     TRAIN_SHARDS="$2"; shift 2 ;;
        --wallclock)  MAX_WALLCLOCK="$2"; shift 2 ;;
        --no-limit)   MAX_WALLCLOCK=0; shift ;;
        --val-every)  VAL_LOSS_EVERY="$2"; shift 2 ;;
        --log-every)  TRAIN_LOG_EVERY="$2"; shift 2 ;;
        --project)    WANDB_PROJECT="$2"; shift 2 ;;
        --env)        EXTRA_ENV="$EXTRA_ENV $2"; shift 2 ;;
        --help|-h)
            echo "Parameter Golf Launcher"
            echo ""
            echo "Usage: ./launch.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gpus N        Number of GPUs (default: 1)"
            echo "  --name NAME     Run name for logs/wandb (default: baseline)"
            echo "  --wandb         Enable wandb logging"
            echo "  --project NAME  Wandb project name (default: parameter-golf)"
            echo "  --shards N      Download N training shards first (skip if data exists)"
            echo "  --wallclock S   Max training seconds (default: 600)"
            echo "  --no-limit      No wallclock limit"
            echo "  --val-every N   Validate every N steps (default: 200)"
            echo "  --log-every N   Log train metrics every N steps (default: 50)"
            echo "  --env 'K=V'     Extra env var (can repeat)"
            echo ""
            echo "Examples:"
            echo "  ./launch.sh --name test1 --wandb"
            echo "  ./launch.sh --gpus 8 --name submission_v1 --wandb"
            echo "  ./launch.sh --no-limit --name long_run --shards 80"
            echo "  ./launch.sh --env 'NUM_LAYERS=12' --env 'MODEL_DIM=768'"
            exit 0
            ;;
        *)
            echo "Unknown option: $1 (try --help)"
            exit 1
            ;;
    esac
done

# ── Timestamp for unique run IDs ──────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="${RUN_NAME}_${TIMESTAMP}"

# ── Download data if requested or missing ─────────────────────────────────────
if [[ -n "$TRAIN_SHARDS" ]]; then
    echo "Downloading dataset (${TRAIN_SHARDS} shards)..."
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards "$TRAIN_SHARDS"
elif [[ ! -d "$DATA_PATH" ]]; then
    echo "Data not found, downloading (1 shard for quick start)..."
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
fi

# ── Install wandb if needed ───────────────────────────────────────────────────
if [[ "$USE_WANDB" == "1" ]]; then
    if ! python3 -c "import wandb" 2>/dev/null; then
        echo "Installing wandb..."
        pip install wandb -q
    fi
    if [[ -z "${WANDB_API_KEY:-}" ]]; then
        echo "WARNING: WANDB_API_KEY not set. Run: export WANDB_API_KEY=your_key"
        echo "         Or run: wandb login"
    fi
fi

# ── Count training shards ────────────────────────────────────────────────────
N_SHARDS=$(ls ${DATA_PATH}/fineweb_train_*.bin 2>/dev/null | wc -l)

# ── Print config ──────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  Parameter Golf — Training Launch"
echo "========================================================"
echo "  Run ID:      ${RUN_ID}"
echo "  GPUs:        ${GPUS}"
echo "  Wallclock:   ${MAX_WALLCLOCK}s"
echo "  Data shards: ${N_SHARDS}"
echo "  Val every:   ${VAL_LOSS_EVERY} steps"
echo "  Log every:   ${TRAIN_LOG_EVERY} steps"
echo "  Wandb:       $([ "$USE_WANDB" == "1" ] && echo "ON ($WANDB_PROJECT)" || echo "OFF")"
if [[ -n "$EXTRA_ENV" ]]; then
echo "  Extra env:   ${EXTRA_ENV}"
fi
echo "========================================================"
echo ""

# ── Launch ────────────────────────────────────────────────────────────────────
export RUN_ID="${RUN_ID}"
export DATA_PATH="${DATA_PATH}"
export TOKENIZER_PATH="${TOKENIZER_PATH}"
export VOCAB_SIZE="${VOCAB_SIZE}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK}"
export USE_WANDB="${USE_WANDB}"
export WANDB_PROJECT="${WANDB_PROJECT}"

# Apply extra env vars
for kv in $EXTRA_ENV; do
    export "$kv"
done

exec torchrun --standalone --nproc_per_node="${GPUS}" train_gpt.py
