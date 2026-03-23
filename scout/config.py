"""Scout configuration."""

import os

# GitHub
GITHUB_REPO = "openai/parameter-golf"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")  # optional, raises rate limit from 60 to 5000/hr
GITHUB_API = "https://api.github.com"

# Claude
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
CLAUDE_MODEL = "claude-sonnet-4-20250514"  # fast + cheap for extraction
CLAUDE_ANALYSIS_MODEL = "claude-sonnet-4-20250514"  # for daily briefing synthesis

# Paths
SCOUT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_DIR = os.path.join(SCOUT_DIR, "state")
BRIEFINGS_DIR = os.path.join(SCOUT_DIR, "briefings")
RAW_PRS_DIR = os.path.join(SCOUT_DIR, "raw", "prs")
RAW_ISSUES_DIR = os.path.join(SCOUT_DIR, "raw", "issues")

# Processing
PR_BATCH_SIZE = 10  # PRs per Claude API call (smaller batches since bodies+comments are larger now)
MAX_BODY_CHARS_FOR_ANALYSIS = 6000  # truncate at analysis time, not storage time
MAX_COMMENT_CHARS_FOR_ANALYSIS = 2000  # per-PR comment budget for analysis
