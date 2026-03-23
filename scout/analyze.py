"""Analyze PRs and issues using Claude API to extract structured data and generate briefings."""

import json
import os
from datetime import datetime, timezone

import anthropic

from config import (
    CLAUDE_MODEL,
    CLAUDE_ANALYSIS_MODEL,
    PR_BATCH_SIZE,
    STATE_DIR,
    BRIEFINGS_DIR,
)

client = anthropic.Anthropic()


# ---------------------------------------------------------------------------
# Extraction: turn raw PR/issue data into structured records
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """\
You are analyzing submissions to the "Parameter Golf" competition. Participants train small language models \
(16MB limit, ≤10min on 8xH100) and are scored by validation bits-per-byte (BPB) on FineWeb — lower is better.

Given a batch of GitHub PRs, extract structured data from each. Return a JSON array where each element has:

{
  "number": <PR number>,
  "title": "<PR title>",
  "user": "<GitHub username>",
  "is_record_attempt": <true if claims record or SOTA, false if non-record/exploration>,
  "bpb_score": <float or null if not stated>,
  "bpb_is_mean": <true if score is mean of multiple seeds, false/null otherwise>,
  "num_seeds": <int or null>,
  "validated": <true/false/null — true if explicitly validated by maintainers>,
  "uses_ttt": <true/false/null — true if uses test-time training>,
  "num_layers": <int or null>,
  "techniques": ["list", "of", "technique", "names"],
  "technique_details": "brief description of notable implementation details",
  "hardware_notes": "<any hardware deviations from standard 8xH100, or null>",
  "status": "<open/closed/merged>",
  "key_findings": "1-2 sentence summary of what's interesting or novel about this submission"
}

Important:
- Normalize technique names consistently (e.g., "XSA" not "exclusive self-attention", "EMA" not "exponential moving average")
- Common techniques to look for: Int6/Int8 quantization, QAT, MLP expansion (3x, etc.), sliding window eval, \
FP16 tied embeddings, Zstd compression, SmearGate, BigramHash, TrigramHash, XSA, EMA, SWA, Partial RoPE, \
Value Residual, Gated Attention, SwiGLU, TTT (test-time training), AdamW TTT, Muon optimizer, \
GradQuant, DenseFormer, MoE, warmdown, cosine scheduling
- If a PR is clearly not a model submission (e.g., docs, infra, discussion), set is_record_attempt to false and techniques to []

Return ONLY the JSON array, no other text.
"""


def extract_pr_data(prs):
    """Send PRs to Claude in batches and extract structured records."""
    if not prs:
        return []

    all_records = []
    for i in range(0, len(prs), PR_BATCH_SIZE):
        batch = prs[i:i + PR_BATCH_SIZE]
        batch_text = "\n\n---\n\n".join(
            f"PR #{p['number']} by @{p['user']} [{p['state']}]\n"
            f"Title: {p['title']}\n"
            f"Labels: {', '.join(p['labels']) if p['labels'] else 'none'}\n"
            f"Created: {p['created_at']} | Updated: {p['updated_at']}\n"
            f"Draft: {p['draft']}\n\n"
            f"{p['body']}"
            for p in batch
        )

        print(f"  Extracting batch {i // PR_BATCH_SIZE + 1} ({len(batch)} PRs)...")
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            messages=[
                {"role": "user", "content": f"{EXTRACTION_PROMPT}\n\nHere are the PRs:\n\n{batch_text}"}
            ],
        )

        text = resp.content[0].text.strip()
        # Handle potential markdown code fences
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text[:-3].strip()

        try:
            records = json.loads(text)
            all_records.extend(records)
        except json.JSONDecodeError as e:
            print(f"  WARNING: Failed to parse Claude response for batch {i // PR_BATCH_SIZE + 1}: {e}")
            print(f"  Raw response (first 500 chars): {text[:500]}")

    return all_records


ISSUE_EXTRACTION_PROMPT = """\
You are analyzing discussion issues from the "Parameter Golf" competition (train small LMs under 16MB in ≤10min on 8xH100, \
scored by validation BPB on FineWeb — lower is better).

Given a batch of GitHub issues, extract structured data. Return a JSON array where each element has:

{
  "number": <issue number>,
  "title": "<title>",
  "user": "<GitHub username>",
  "category": "<one of: technique_discussion, competition_meta, bug_report, question, analysis, other>",
  "techniques_mentioned": ["list", "of", "techniques"],
  "key_insights": "1-3 sentence summary of any useful insights, findings, or recommendations",
  "relevance": "<high/medium/low> — high if contains novel technique ideas or important competition info"
}

Return ONLY the JSON array, no other text.
"""


def extract_issue_data(issues):
    """Send issues to Claude in batches and extract structured records."""
    if not issues:
        return []

    all_records = []
    for i in range(0, len(issues), PR_BATCH_SIZE):
        batch = issues[i:i + PR_BATCH_SIZE]
        batch_text = "\n\n---\n\n".join(
            f"Issue #{iss['number']} by @{iss['user']} [{iss['state']}]\n"
            f"Title: {iss['title']}\n"
            f"Labels: {', '.join(iss['labels']) if iss['labels'] else 'none'}\n"
            f"Created: {iss['created_at']} | Updated: {iss['updated_at']}\n\n"
            f"{iss['body']}"
            for iss in batch
        )

        print(f"  Extracting issue batch {i // PR_BATCH_SIZE + 1} ({len(batch)} issues)...")
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            messages=[
                {"role": "user", "content": f"{ISSUE_EXTRACTION_PROMPT}\n\nHere are the issues:\n\n{batch_text}"}
            ],
        )

        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text[:-3].strip()

        try:
            records = json.loads(text)
            all_records.extend(records)
        except json.JSONDecodeError as e:
            print(f"  WARNING: Failed to parse issue batch {i // PR_BATCH_SIZE + 1}: {e}")

    return all_records


# ---------------------------------------------------------------------------
# State management: merge new records into persistent leaderboard + techniques
# ---------------------------------------------------------------------------

def _load_json(filename, default=None):
    path = os.path.join(STATE_DIR, filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default if default is not None else {}


def _save_json(filename, data):
    path = os.path.join(STATE_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def update_leaderboard(pr_records):
    """Merge extracted PR records into the leaderboard."""
    leaderboard = _load_json("leaderboard.json", default=[])
    existing = {entry["number"]: i for i, entry in enumerate(leaderboard)}

    for rec in pr_records:
        if rec["number"] in existing:
            # Update existing entry
            leaderboard[existing[rec["number"]]] = rec
        else:
            leaderboard.append(rec)

    # Sort by BPB score (nulls at end)
    leaderboard.sort(key=lambda x: (x.get("bpb_score") is None, x.get("bpb_score") or 999))
    _save_json("leaderboard.json", leaderboard)
    print(f"  Leaderboard: {len(leaderboard)} entries total")
    return leaderboard


def update_techniques(pr_records):
    """Build/update technique registry from PR records."""
    techniques = _load_json("techniques.json", default={})

    for rec in pr_records:
        for tech in rec.get("techniques", []):
            if tech not in techniques:
                techniques[tech] = {
                    "name": tech,
                    "submissions": [],
                    "best_bpb": None,
                    "worst_bpb": None,
                    "avg_bpb": None,
                    "count": 0,
                }
            entry = techniques[tech]
            sub = {"pr": rec["number"], "bpb": rec.get("bpb_score"), "user": rec.get("user")}
            # Avoid duplicates
            existing_prs = {s["pr"] for s in entry["submissions"]}
            if rec["number"] not in existing_prs:
                entry["submissions"].append(sub)
            else:
                # Update existing
                entry["submissions"] = [s if s["pr"] != rec["number"] else sub for s in entry["submissions"]]

            # Recalculate stats
            scores = [s["bpb"] for s in entry["submissions"] if s["bpb"] is not None]
            entry["count"] = len(entry["submissions"])
            if scores:
                entry["best_bpb"] = min(scores)
                entry["worst_bpb"] = max(scores)
                entry["avg_bpb"] = round(sum(scores) / len(scores), 4)

    _save_json("techniques.json", techniques)
    print(f"  Techniques: {len(techniques)} tracked")
    return techniques


def update_issues_state(issue_records):
    """Store issue analysis."""
    issues = _load_json("issues.json", default=[])
    existing = {entry["number"]: i for i, entry in enumerate(issues)}

    for rec in issue_records:
        if rec["number"] in existing:
            issues[existing[rec["number"]]] = rec
        else:
            issues.append(rec)

    _save_json("issues.json", issues)
    print(f"  Issues: {len(issues)} tracked")
    return issues


def update_history(leaderboard):
    """Append a daily snapshot to history."""
    history = _load_json("history.json", default=[])
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Top 10 snapshot
    top10 = [
        {"number": e["number"], "user": e.get("user"), "bpb": e.get("bpb_score"), "techniques": e.get("techniques", [])}
        for e in leaderboard[:10]
        if e.get("bpb_score") is not None
    ]

    snapshot = {
        "date": today,
        "total_submissions": len(leaderboard),
        "submissions_with_score": len([e for e in leaderboard if e.get("bpb_score")]),
        "best_bpb": leaderboard[0].get("bpb_score") if leaderboard and leaderboard[0].get("bpb_score") else None,
        "top10": top10,
    }

    # Replace today's snapshot if re-running, otherwise append
    history = [h for h in history if h["date"] != today]
    history.append(snapshot)
    history.sort(key=lambda x: x["date"])
    _save_json("history.json", history)
    return history


# ---------------------------------------------------------------------------
# Briefing generation
# ---------------------------------------------------------------------------

BRIEFING_PROMPT = """\
You are the Parameter Golf Scout — an AI analyst tracking the Parameter Golf competition \
(train small LMs under 16MB, ≤10min on 8xH100, scored by validation BPB on FineWeb — lower is better).

Generate a daily briefing in markdown. Structure it as:

# Parameter Golf Scout — {date}

## Leaderboard (Top 15)
| Rank | PR | Author | BPB | Seeds | TTT | Key Techniques |
(include validation status if known)

## New Activity
- Summarize new/updated PRs and issues since last run
- Highlight any new records or notable results

## Technique Analysis
- Which techniques appear in top submissions
- Any technique combos that seem to help or hurt
- Emerging techniques worth watching

## Frontier Gaps & Opportunities
- Technique combos that haven't been tried yet (but components show promise individually)
- Under-explored areas of the search space
- Specific experiments worth trying next, ranked by expected impact

## Risk & Caveats
- Which top scores are unvalidated or single-seed
- Any legitimacy concerns flagged in issues/comments

Be specific, quantitative, and actionable. Reference PR numbers. This briefing will be read by a competitor \
who wants concrete guidance on what to try next.
"""


def generate_briefing(leaderboard, techniques, issues, history, new_pr_count, new_issue_count):
    """Generate the daily markdown briefing using Claude."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Build context for Claude
    top30 = json.dumps(leaderboard[:30], indent=2)
    tech_summary = json.dumps(
        {k: {"best_bpb": v["best_bpb"], "count": v["count"], "avg_bpb": v["avg_bpb"]}
         for k, v in sorted(techniques.items(), key=lambda x: x[1].get("best_bpb") or 999)[:30]},
        indent=2,
    )
    high_relevance_issues = [i for i in issues if i.get("relevance") == "high"]
    issues_text = json.dumps(high_relevance_issues[:10], indent=2) if high_relevance_issues else "None"
    history_text = json.dumps(history[-7:], indent=2) if history else "No history yet"

    context = (
        f"Date: {today}\n"
        f"New/updated PRs this run: {new_pr_count}\n"
        f"New issues this run: {new_issue_count}\n\n"
        f"## Top 30 Leaderboard Entries\n{top30}\n\n"
        f"## Top 30 Techniques by Best BPB\n{tech_summary}\n\n"
        f"## High-Relevance Issues\n{issues_text}\n\n"
        f"## Recent History (last 7 days)\n{history_text}"
    )

    print("Generating daily briefing...")
    resp = client.messages.create(
        model=CLAUDE_ANALYSIS_MODEL,
        max_tokens=4096,
        messages=[
            {"role": "user", "content": f"{BRIEFING_PROMPT}\n\nHere is today's data:\n\n{context}"}
        ],
    )

    briefing = resp.content[0].text.strip()

    # Save briefing
    path = os.path.join(BRIEFINGS_DIR, f"{today}.md")
    with open(path, "w") as f:
        f.write(briefing)
    print(f"  Briefing saved to {path}")

    return briefing


def analyze_all(fetched):
    """Main analysis entry point. Takes output from fetch.fetch_all()."""
    print("\n--- Extraction ---")
    pr_records = extract_pr_data(fetched["prs"])
    issue_records = extract_issue_data(fetched["issues"])

    print("\n--- State Update ---")
    leaderboard = update_leaderboard(pr_records)
    techniques = update_techniques(pr_records)
    issues = update_issues_state(issue_records)
    history = update_history(leaderboard)

    print("\n--- Briefing ---")
    briefing = generate_briefing(
        leaderboard,
        techniques,
        issues,
        history,
        new_pr_count=len(fetched["prs"]),
        new_issue_count=len(fetched["issues"]),
    )

    return {
        "pr_records": pr_records,
        "issue_records": issue_records,
        "leaderboard": leaderboard,
        "techniques": techniques,
        "briefing": briefing,
    }
