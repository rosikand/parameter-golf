# Scout — Architecture & Design

## What is Scout?

Scout is an automated intelligence agent for the Parameter Golf competition. It monitors GitHub for new submissions, extracts structured data about techniques and results using Claude, maintains a persistent knowledge base, and generates daily briefings with actionable recommendations.

The goal: give a competitor a complete, up-to-date picture of the competitive landscape — what's working, what isn't, and what to try next — without manually reading hundreds of PRs.

## System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          scout.py (orchestrator)                     │
│                                                                      │
│   --fetch-only          (full run)           --brief-only            │
│        │                    │                      │                  │
│        ▼                    ▼                      ▼                  │
│   ┌─────────┐    ┌─────────────────────┐   ┌──────────────┐         │
│   │  FETCH  │    │  FETCH → ANALYZE →  │   │   ANALYZE    │         │
│   │  only   │    │  STATE → BRIEFING   │   │  (briefing   │         │
│   └─────────┘    └─────────────────────┘   │   only)      │         │
│                                             └──────────────┘         │
└─────────────────────────────────────────────────────────────────────┘

                        DETAILED DATA FLOW
                        ==================

  ┌──────────────┐
  │  GitHub API  │
  │  (REST v3)   │
  └──────┬───────┘
         │  PRs, issues, comments
         │  (paginated, delta-based)
         ▼
  ┌──────────────┐     ┌─────────────────────┐
  │   fetch.py   │────▶│    raw/prs/*.json    │  Full body + comments
  │              │────▶│  raw/issues/*.json   │  cached per item
  │  Tracks:     │     └─────────────────────┘
  │  - last_fetch│
  │  - known IDs │     ┌─────────────────────┐
  │              │────▶│  raw/diffs/*.diff    │  On-demand only
  └──────┬───────┘     └─────────────────────┘  (--diff flag)
         │
         │  New/updated PRs + issues
         ▼
  ┌──────────────┐
  │  analyze.py  │
  │              │
  │  ┌────────────────────────────────────┐
  │  │  EXTRACTION (Claude API, batched)  │
  │  │                                    │
  │  │  Raw PR ──▶ Structured record:     │
  │  │    - BPB score, seed count         │
  │  │    - techniques used               │
  │  │    - TTT yes/no                    │
  │  │    - validation status             │
  │  │    - key findings                  │
  │  │                                    │
  │  │  Batch of 10 PRs per API call      │
  │  │  Retry + individual fallback       │
  │  │  on JSON parse failure             │
  │  └────────────────────────────────────┘
  │              │
  │              │  Structured records
  │              ▼
  │  ┌────────────────────────────────────┐
  │  │  STATE UPDATE (pure Python)        │
  │  │                                    │
  │  │  leaderboard.json ◀── merge +     │
  │  │                       sort by BPB  │
  │  │                                    │
  │  │  techniques.json  ◀── aggregate    │
  │  │                       stats per    │
  │  │                       technique    │
  │  │                                    │
  │  │  issues.json      ◀── merge       │
  │  │  history.json     ◀── daily snap  │
  │  └────────────────────────────────────┘
  │              │
  │              │  Full state
  │              ▼
  │  ┌────────────────────────────────────┐
  │  │  BRIEFING (Claude API, 1 call)     │
  │  │                                    │
  │  │  Inputs:                           │
  │  │   - Top 30 leaderboard entries     │
  │  │   - Top 30 techniques by BPB      │
  │  │   - High-relevance issues          │
  │  │   - Last 7 days of history         │
  │  │                                    │
  │  │  Outputs:                          │
  │  │   - Ranked leaderboard table       │
  │  │   - Technique effectiveness        │
  │  │   - Frontier gaps & opportunities  │
  │  │   - Risk & caveats                 │
  │  └──────────────┬─────────────────────┘
  │                  │
  └──────────────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │  briefings/         │
          │  2026-03-23.md      │  Human-readable daily report
          └─────────────────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │  Claude Code        │  User queries the briefing
          │  (interactive)      │  and state files directly
          └─────────────────────┘
```

## Key Design Decisions

### Delta-based fetching

Scout tracks `last_pr_fetch` and `last_issue_fetch` timestamps in `fetch_state.json`. On each run, it only pulls items updated since the last fetch via GitHub's `sort=updated&direction=desc` parameter. This means:

- Day 1 is expensive (all PRs + comments)
- Day 2+ is cheap (only new/modified items)
- Re-running on the same day is near-instant

### Full storage, truncated analysis

Raw PR bodies and comments are stored in full (no truncation) in `raw/`. When sent to Claude for extraction, bodies are capped at 6000 chars and comments at 2000 chars per PR. This means:

- We never lose data at the storage layer
- We can re-analyze with different truncation limits without re-fetching
- Claude gets enough context to extract structured data without blowing context limits

### Batched extraction with fallback

PRs are sent to Claude in batches of 10 for extraction. If the JSON response fails to parse:

1. Retry with doubled `max_tokens` (8192 → 16384) in case the response was truncated
2. If that fails, fall back to extracting each PR individually
3. If an individual PR still fails, skip it and log a warning

This ensures a single problematic PR (e.g., very long body with unusual formatting) doesn't cause a whole batch of 10 to be lost.

### On-demand diffs

PR code diffs are not fetched during daily runs because:

- Most PRs modify the same `train_gpt.py` file — diffs are large and redundant
- PR bodies already describe techniques, hyperparameters, and results in detail
- Fetching diffs for 486 PRs would be ~500 extra API calls

Instead, diffs are fetched on-demand via `--diff <PR numbers>` when a specific implementation needs to be studied.

### Separation of state and analysis

The state layer (`leaderboard.json`, `techniques.json`) is pure Python with no Claude dependency. The analysis layer (extraction, briefing) uses Claude. This means:

- State can be inspected, queried, and manually edited without API calls
- Briefings can be regenerated from existing state (`--brief-only`) cheaply
- The system degrades gracefully if the API is unavailable — you still have the last known state

## Cost Estimation

| Phase | Day 1 (cold start) | Day 2+ (incremental) |
|-------|--------------------|--------------------|
| GitHub API calls | ~1000 (PRs + comments) | ~10-50 |
| Claude extraction calls | ~50 batches | ~1-3 batches |
| Claude briefing calls | 1 | 1 |
| Estimated Claude cost | ~$0.50-1.00 | ~$0.05-0.10 |
| Wall clock time | ~30 min | ~2-5 min |
