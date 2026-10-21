"""
main.py
-------
Entrypoint for the Self-Healing Documentation agent.

When triggered by the GitHub Action (or run locally via local_test.py),
this script:
  1. Reads environment variables for GitHub context.
  2. Fetches the PR diff and current README via GitHubService.
  3. Runs the LangGraph Actor-Critic workflow (Step 3 — coming next).
  4. Opens a new PR with the updated README via GitHubService.
"""

import os
import sys

from dotenv import load_dotenv

from github_service import GitHubService

# Load .env for local development; harmless in CI where env vars are injected
load_dotenv()


def main():
    # ── 1. Read environment variables ──────────────────────────────────
    github_token = os.environ.get("GITHUB_TOKEN")
    groq_api_key = os.environ.get("GROQ_API_KEY")
    repo_name = os.environ.get("REPO_NAME")
    pr_number_str = os.environ.get("PR_NUMBER")

    # Validate required vars
    missing = []
    if not github_token:
        missing.append("GITHUB_TOKEN")
    if not groq_api_key:
        missing.append("GROQ_API_KEY")
    if not repo_name:
        missing.append("REPO_NAME")
    if not pr_number_str:
        missing.append("PR_NUMBER")

    if missing:
        print(f"[main] ERROR: Missing required environment variables: "
              f"{', '.join(missing)}")
        sys.exit(1)

    pr_number = int(pr_number_str)
    print(f"[main] Processing PR #{pr_number} in {repo_name}")

    # ── 2. Fetch GitHub data ───────────────────────────────────────────
    gh = GitHubService(token=github_token, repo_name=repo_name)

    pr_diff = gh.get_pr_diff(pr_number)
    current_readme = gh.get_readme_content(branch="main")

    if not pr_diff.strip():
        print("[main] PR diff is empty — nothing to document. Exiting.")
        sys.exit(0)

    print(f"[main] Diff length: {len(pr_diff)} chars")
    print(f"[main] Current README length: {len(current_readme)} chars")

    # ── 3. Run LangGraph Actor-Critic workflow ─────────────────────────
    from agent_graph import run_workflow

    result = run_workflow(pr_diff=pr_diff, current_readme=current_readme)
    final_readme = result["draft_readme"]
    pr_summary = result["pr_summary"]

    print(f"[main] Final README: {len(final_readme)} chars "
          f"(after {result['iteration_count']} iteration(s))")

    # ── 4. Create a docs PR with the updated README ────────────────────
    docs_pr_url = gh.create_docs_pr(
        pr_number=pr_number,
        updated_readme=final_readme,
        pr_summary=pr_summary,
    )
    print(f"[main] DONE! Docs PR: {docs_pr_url}")


if __name__ == "__main__":
    main()
