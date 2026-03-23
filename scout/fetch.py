"""Fetch PRs and issues from GitHub API."""

import json
import os
import time
from datetime import datetime, timezone

import requests

from config import (
    GITHUB_API,
    GITHUB_REPO,
    GITHUB_TOKEN,
    RAW_ISSUES_DIR,
    RAW_PRS_DIR,
    STATE_DIR,
    MAX_PR_BODY_CHARS,
)


def _headers():
    h = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"token {GITHUB_TOKEN}"
    return h


def _get_paginated(url, params=None, max_pages=50):
    """Fetch all pages from a paginated GitHub API endpoint."""
    params = params or {}
    params.setdefault("per_page", 100)
    results = []
    for page in range(1, max_pages + 1):
        params["page"] = page
        resp = requests.get(url, headers=_headers(), params=params)
        if resp.status_code == 403 and "rate limit" in resp.text.lower():
            reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(reset - int(time.time()), 1) + 1
            print(f"  Rate limited. Waiting {wait}s...")
            time.sleep(wait)
            resp = requests.get(url, headers=_headers(), params=params)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        results.extend(data)
        # Check if there's a next page
        if "next" not in resp.links:
            break
    return results


def _load_state():
    """Load fetch state (last run timestamp, known PR/issue numbers)."""
    path = os.path.join(STATE_DIR, "fetch_state.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"last_pr_fetch": None, "last_issue_fetch": None, "known_prs": [], "known_issues": []}


def _save_state(state):
    path = os.path.join(STATE_DIR, "fetch_state.json")
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def _slim_pr(pr):
    """Extract the fields we care about from a raw PR object."""
    body = (pr.get("body") or "")[:MAX_PR_BODY_CHARS]
    return {
        "number": pr["number"],
        "title": pr["title"],
        "state": pr["state"],
        "user": pr["user"]["login"],
        "created_at": pr["created_at"],
        "updated_at": pr["updated_at"],
        "merged_at": pr.get("merged_at") or pr.get("pull_request", {}).get("merged_at"),
        "body": body,
        "labels": [l["name"] for l in pr.get("labels", [])],
        "url": pr["html_url"],
        "draft": pr.get("draft", False),
    }


def _slim_issue(issue):
    """Extract fields from a raw issue (excluding PRs which also show as issues)."""
    body = (issue.get("body") or "")[:MAX_PR_BODY_CHARS]
    return {
        "number": issue["number"],
        "title": issue["title"],
        "state": issue["state"],
        "user": issue["user"]["login"],
        "created_at": issue["created_at"],
        "updated_at": issue["updated_at"],
        "body": body,
        "labels": [l["name"] for l in issue.get("labels", [])],
        "url": issue["html_url"],
    }


def fetch_prs(since=None):
    """Fetch PRs, optionally only those updated since a given ISO timestamp."""
    url = f"{GITHUB_API}/repos/{GITHUB_REPO}/pulls"
    params = {"state": "all", "sort": "updated", "direction": "desc"}

    print(f"Fetching PRs from {GITHUB_REPO}...")
    all_prs = _get_paginated(url, params)
    print(f"  Fetched {len(all_prs)} total PRs from API")

    # Filter to only those updated since last fetch
    if since:
        since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
        all_prs = [p for p in all_prs if
                   datetime.fromisoformat(p["updated_at"].replace("Z", "+00:00")) > since_dt]
        print(f"  {len(all_prs)} PRs updated since {since}")

    slim = [_slim_pr(p) for p in all_prs]

    # Cache raw data
    for pr in slim:
        path = os.path.join(RAW_PRS_DIR, f"{pr['number']}.json")
        with open(path, "w") as f:
            json.dump(pr, f, indent=2)

    return slim


def fetch_issues(since=None):
    """Fetch issues (not PRs), optionally only those updated since a given ISO timestamp."""
    url = f"{GITHUB_API}/repos/{GITHUB_REPO}/issues"
    params = {"state": "all", "sort": "updated", "direction": "desc"}

    print(f"Fetching issues from {GITHUB_REPO}...")
    all_issues = _get_paginated(url, params)

    # GitHub API returns PRs as issues too — filter them out
    all_issues = [i for i in all_issues if "pull_request" not in i]
    print(f"  Fetched {len(all_issues)} issues (excluding PRs)")

    if since:
        since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
        all_issues = [i for i in all_issues if
                      datetime.fromisoformat(i["updated_at"].replace("Z", "+00:00")) > since_dt]
        print(f"  {len(all_issues)} issues updated since {since}")

    slim = [_slim_issue(i) for i in all_issues]

    for issue in slim:
        path = os.path.join(RAW_ISSUES_DIR, f"{issue['number']}.json")
        with open(path, "w") as f:
            json.dump(issue, f, indent=2)

    return slim


def fetch_all():
    """Main entry point: fetch new/updated PRs and issues since last run."""
    state = _load_state()
    now = datetime.now(timezone.utc).isoformat()

    new_prs = fetch_prs(since=state["last_pr_fetch"])
    new_issues = fetch_issues(since=state["last_issue_fetch"])

    # Update state
    state["last_pr_fetch"] = now
    state["last_issue_fetch"] = now
    known_prs = set(state["known_prs"])
    known_issues = set(state["known_issues"])
    new_pr_numbers = [p["number"] for p in new_prs if p["number"] not in known_prs]
    updated_pr_numbers = [p["number"] for p in new_prs if p["number"] in known_prs]
    new_issue_numbers = [i["number"] for i in new_issues if i["number"] not in known_issues]
    known_prs.update(p["number"] for p in new_prs)
    known_issues.update(i["number"] for i in new_issues)
    state["known_prs"] = sorted(known_prs)
    state["known_issues"] = sorted(known_issues)
    _save_state(state)

    print(f"\nFetch summary:")
    print(f"  New PRs: {len(new_pr_numbers)} | Updated PRs: {len(updated_pr_numbers)}")
    print(f"  New issues: {len(new_issue_numbers)}")

    return {
        "prs": new_prs,
        "issues": new_issues,
        "new_pr_numbers": new_pr_numbers,
        "updated_pr_numbers": updated_pr_numbers,
        "new_issue_numbers": new_issue_numbers,
    }


if __name__ == "__main__":
    result = fetch_all()
    print(f"\nDone. {len(result['prs'])} PRs, {len(result['issues'])} issues processed.")
