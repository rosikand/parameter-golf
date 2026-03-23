# Parameter Golf Scout

Daily intelligence agent that tracks the [Parameter Golf](https://github.com/openai/parameter-golf) competition — fetches PRs/issues, extracts techniques and scores using Claude, and generates actionable briefings.

## Setup

```bash
# Required: Anthropic API key (for analysis + briefing)
export ANTHROPIC_API_KEY="sk-ant-..."

# Optional but recommended: GitHub token (raises rate limit from 60 to 5000 req/hr)
export GITHUB_TOKEN="ghp_..."

# Install dependencies
pip install anthropic requests
```

## Usage

```bash
cd scout/

# Full run: fetch new data + analyze with Claude + generate briefing
python scout.py

# Fetch only (no Claude API calls — just pull latest PRs/issues/comments)
python scout.py --fetch-only

# Regenerate briefing from existing state (no fetch, one Claude call)
python scout.py --brief-only

# Fetch the code diff for specific PRs (for deep-diving into implementations)
python scout.py --diff 473 490 414
```

## Output

```
scout/
├── briefings/          # Daily markdown briefings (e.g. 2026-03-23.md)
├── state/
│   ├── leaderboard.json   # All submissions ranked by BPB score
│   ├── techniques.json    # Technique registry with effectiveness stats
│   ├── issues.json        # Analyzed issues
│   ├── history.json       # Daily snapshots for trend tracking
│   └── fetch_state.json   # Tracks last fetch time + known PR/issue numbers
└── raw/
    ├── prs/            # Cached PR data (full body + comments)
    ├── issues/         # Cached issue data (full body + comments)
    └── diffs/          # On-demand PR diffs (.diff + _files.json)
```

## Querying with Claude Code

After running the scout, you can ask Claude Code questions against the local files:

- "Read `scout/briefings/2026-03-23.md` and tell me what technique I should try next"
- "Look at `scout/state/techniques.json` and find which techniques have never been combined"
- "Read `scout/state/leaderboard.json` and show me all submissions using Value Residual"
- "Read `scout/raw/prs/473.json` and explain their TTT approach"

## How it works

- **Day 1**: Fetches all PRs + issues + comments, sends them to Claude in batches for structured extraction, builds leaderboard + technique registry, generates briefing.
- **Day 2+**: Only fetches items updated since last run. Analysis cost scales with new activity (~5-20 items/day), not total competition size.
- **Diffs**: Fetched on-demand per PR (`--diff`), not during daily runs. Use this when you want to study a specific implementation.
