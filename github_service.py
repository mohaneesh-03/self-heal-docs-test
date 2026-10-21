"""
github_service.py
-----------------
Encapsulates all GitHub API interactions via PyGithub.

Responsibilities:
  - Fetch the combined patch/diff of a merged PR.
  - Fetch the current README.md content from the base branch.
  - Create a new branch, commit updated README, and open a PR.
"""

from github import Github, GithubException


class GitHubService:
    """Thin wrapper around PyGithub for self-healing docs operations."""

    def __init__(self, token: str, repo_name: str):
        self.github = Github(token)
        self.repo = self.github.get_repo(repo_name)

    def get_pr_diff(self, pr_number: int) -> str:
        """
        Fetch the combined patch (diff) of all files changed in the PR.

        Returns a single string with the unified diff of every file.
        """
        pull = self.repo.get_pull(pr_number)
        files = pull.get_files()

        diff_parts: list[str] = []
        for file in files:
            header = f"--- a/{file.filename}\n+++ b/{file.filename}"
            patch = file.patch if file.patch else "(binary or empty)"
            diff_parts.append(f"{header}\n{patch}")

        combined_diff = "\n\n".join(diff_parts)
        print(f"[GitHubService] Fetched diff for PR #{pr_number} "
              f"({len(diff_parts)} file(s), {len(combined_diff)} chars)")
        return combined_diff

    def get_readme_content(self, branch: str = "main") -> str:
        """
        Fetch the raw text of README.md from the given branch.

        Returns an empty string with a warning if README.md doesn't exist yet.
        """
        try:
            contents = self.repo.get_contents("README.md", ref=branch)
            readme_text = contents.decoded_content.decode("utf-8")
            print(f"[GitHubService] Fetched README.md from '{branch}' "
                  f"({len(readme_text)} chars)")
            return readme_text
        except GithubException as e:
            if e.status == 404:
                print(f"[GitHubService] No README.md found on '{branch}'. "
                      "Starting from scratch.")
                return ""
            raise

    def create_docs_pr(
        self,
        pr_number: int,
        updated_readme: str,
        pr_summary: str,
        base_branch: str = "main",
    ) -> str:
        """
        Create a new branch, commit the updated README, and open a PR.

        Args:
            pr_number:      The original PR number (used for branch naming).
            updated_readme: The new README.md content to commit.
            pr_summary:     LLM-generated summary for the PR description.
            base_branch:    The branch to target (default: main).

        Returns:
            The URL of the newly created Pull Request.
        """
        new_branch = f"docs/update-pr-{pr_number}"

        # Get the SHA of the base branch HEAD
        base_ref = self.repo.get_branch(base_branch)
        base_sha = base_ref.commit.sha

        # Create the new branch
        try:
            self.repo.create_git_ref(
                ref=f"refs/heads/{new_branch}",
                sha=base_sha,
            )
            print(f"[GitHubService] Created branch '{new_branch}'")
        except GithubException as e:
            if e.status == 422:
                # Branch already exists — update it instead
                print(f"[GitHubService] Branch '{new_branch}' already exists, "
                      "updating it.")
            else:
                raise

        # Create or update README.md on the new branch
        try:
            existing = self.repo.get_contents("README.md", ref=new_branch)
            self.repo.update_file(
                path="README.md",
                message=f"docs: update README for PR #{pr_number}",
                content=updated_readme,
                sha=existing.sha,
                branch=new_branch,
            )
        except GithubException as e:
            if e.status == 404:
                self.repo.create_file(
                    path="README.md",
                    message=f"docs: create README for PR #{pr_number}",
                    content=updated_readme,
                    branch=new_branch,
                )
            else:
                raise

        # Open a Pull Request
        pr = self.repo.create_pull(
            title=f"docs: Update README for PR #{pr_number}",
            body=pr_summary,
            head=new_branch,
            base=base_branch,
        )
        print(f"[GitHubService] Opened docs PR: {pr.html_url}")
        return pr.html_url
