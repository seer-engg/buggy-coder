from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:  # pragma: no cover - support script execution without package installation
    from .config import GitHubConfig
except ImportError:  # pragma: no cover - fallback when package context missing
    from config import GitHubConfig


class GitHubAPIError(RuntimeError):
    """Raised when the GitHub API returns an error or cannot be reached."""


@dataclass(slots=True)
class PullRequestData:
    number: int
    title: str | None
    body: str | None
    html_url: str | None
    merged: bool
    merged_at: str | None

    @classmethod
    def from_api_response(cls, payload: dict[str, Any]) -> "PullRequestData":
        # GitHub uses merged_at to indicate merged PRs. The ``merged`` field is only available
        # on some responses, so we compute a derived boolean here for convenience.
        merged_at = payload.get("merged_at")
        merged = bool(merged_at) or bool(payload.get("merged"))
        return cls(
            number=int(payload["number"]),
            title=payload.get("title"),
            body=payload.get("body"),
            html_url=payload.get("html_url"),
            merged=merged,
            merged_at=merged_at,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "title": self.title,
            "body": self.body,
            "html_url": self.html_url,
            "merged": self.merged,
            "merged_at": self.merged_at,
        }


class GitHubClient:
    def __init__(self, config: GitHubConfig, *, timeout: int = 15) -> None:
        self._token = config.token
        self._base_url = config.api_base_url.rstrip("/")
        self._timeout = timeout

    def fetch_pull_request(self, repository: str, pull_number: int) -> PullRequestData:
        path = f"/repos/{repository}/pulls/{pull_number}"
        payload = self._request("GET", path)
        return PullRequestData.from_api_response(payload)

    # Internal helpers -----------------------------------------------------------------

    def _request(self, method: str, path: str) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        request = Request(url, method=method)
        request.add_header("Accept", "application/vnd.github+json")
        request.add_header("User-Agent", "buggy-coder-sync-agent")
        if self._token:
            request.add_header("Authorization", f"Bearer {self._token}")

        try:
            with urlopen(request, timeout=self._timeout) as response:  # nosec: B310 - controlled URL
                payload = response.read()
        except HTTPError as exc:  # pragma: no cover - handled for completeness
            body = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else ""
            raise GitHubAPIError(
                f"GitHub API request failed with status {exc.code}: {body or exc.reason}"
            ) from exc
        except URLError as exc:  # pragma: no cover - handled for completeness
            raise GitHubAPIError(f"Unable to reach GitHub API: {exc.reason}") from exc

        if not payload:
            return {}

        try:
            return json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise GitHubAPIError("Received malformed JSON from GitHub API") from exc
