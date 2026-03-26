# Quickstart

You've forked the [parameter-golf](https://github.com/openai/parameter-golf) repo and want to start running experiments.

## One-time setup (~10 min)

```bash
# From the repo root
cd parameter-golf

# Put the infra files in a subfolder
mkdir infra
# copy modal_train.py, launch.py, leaderboard.py, AGENTS.md, HOW_IT_WORKS.md into infra/

# Install tools
pip install modal wandb
modal setup              # opens browser to authenticate

# Set your W&B key (add to .bashrc/.zshrc so it persists)
export WANDB_API_KEY=<your-key>

# Cache FineWeb dataset into Modal Volume (downloads ~8GB, runs once)
cd infra
modal run modal_train.py::setup_data
```

## Verify everything works

```bash
# Smoke test the baseline (~2 min, ~$0.25)
python launch.py ../train_gpt.py --name first_smoke --smoke

# Watch it live on W&B: https://wandb.ai/<you>/parameter-golf

# After a few minutes, pull results
python leaderboard.py --sync
python leaderboard.py
```

You should see one entry with a BPB score. The number will be bad (barely trained) — that's fine, it means the pipeline works.

## Run a real baseline

```bash
# Full 10-min run (~$1.30)
python launch.py ../train_gpt.py --name baseline_full
```

This returns instantly. The run continues on Modal. Check W&B for live curves, or sync later.

## Start experimenting

```bash
# Fork the baseline
cp ../train_gpt.py ../my_idea.py
# ... edit my_idea.py ...

# Launch
python launch.py ../my_idea.py --name my_idea

# Launch a bunch and go to sleep
python launch.py ../idea_2.py --name idea_2
python launch.py ../idea_3.py --name idea_3

# Next morning
python leaderboard.py --sync
python leaderboard.py
python leaderboard.py --full                  # all columns
python leaderboard.py --log my_idea_0325_2130 # fetch full log for a run
python leaderboard.py --compare idea_2_0325_2131 idea_3_0325_2132
```

## Using an AI coding agent

Put `AGENTS.md` in the repo root (or symlink it as `CLAUDE.md`). It tells the agent exactly what stdout format to produce, what env vars to read, and what the end-of-script sequence must look like. The agent writes the training script, you launch it:

```bash
python launch.py agent_idea_7.py --name agent_idea_7
```

## Useful commands

```bash
# Launch
python launch.py script.py --name NAME              # full 10-min run
python launch.py script.py --name NAME --smoke       # 2-min smoke test
python launch.py script.py --name NAME --fast        # skip data cache check
python launch.py script.py --name NAME --num-layers 12 --model-dim 384

# Results
python leaderboard.py --sync                         # pull from Modal Volume
python leaderboard.py                                # view leaderboard
python leaderboard.py --full                         # all columns
python leaderboard.py --log RUN_ID                   # fetch training log
python leaderboard.py --compare RUN_A RUN_B          # side-by-side
python leaderboard.py --delete RUN_ID                # remove from local leaderboard

# Data setup (only needed once)
modal run modal_train.py::setup_data
```

## Files

```
infra/
  modal_train.py    # Modal app (data setup, training, result sync)
  launch.py         # CLI launcher (fire-and-forget)
  leaderboard.py    # Results viewer + sync from Volume
  AGENTS.md         # Instructions for AI coding agents
  HOW_IT_WORKS.md   # Detailed explanation of the infrastructure
```
