from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional


class ConfigurationError(RuntimeError):
    """Raised when mandatory configuration cannot be resolved."""


def _read_first_env_value(names: Iterable[str]) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


@dataclass(slots=True)
class GitHubConfig:
    token: str
    api_base_url: str = "https://api.github.com"


@dataclass(slots=True)
class AsanaConfig:
    token: str
    api_base_url: str = "https://app.asana.com/api/1.0"


@dataclass(slots=True)
class AppConfig:
    github: GitHubConfig
    asana: AsanaConfig


def load_config() -> AppConfig:
    """Load application configuration from environment variables.

    Environment Variables (first non-empty value wins where multiple names are listed):
      - GitHub token: ``GITHUB_TOKEN`` / ``GH_TOKEN``
      - GitHub API base URL (optional): ``GITHUB_API_BASE_URL``
      - Asana token: ``ASANA_TOKEN`` / ``ASANA_PERSONAL_ACCESS_TOKEN``
      - Asana API base URL (optional): ``ASANA_API_BASE_URL``
    """

    errors: list[str] = []

    github_token = _read_first_env_value(["GITHUB_TOKEN", "GH_TOKEN"])
    if not github_token:
        errors.append(
            "Missing GitHub credentials: set GITHUB_TOKEN or GH_TOKEN environment variable."
        )

    github_base_url = os.getenv("GITHUB_API_BASE_URL", "https://api.github.com").rstrip("/")
    if not github_base_url.startswith("http"):
        errors.append(
            "Invalid GitHub API base URL. Expected an absolute URL, got "
            f"{github_base_url!r}."
        )

    asana_token = _read_first_env_value(["ASANA_TOKEN", "ASANA_PERSONAL_ACCESS_TOKEN"])
    if not asana_token:
        errors.append(
            "Missing Asana credentials: set ASANA_TOKEN or ASANA_PERSONAL_ACCESS_TOKEN environment variable."
        )

    asana_base_url = os.getenv("ASANA_API_BASE_URL", "https://app.asana.com/api/1.0").rstrip("/")
    if not asana_base_url.startswith("http"):
        errors.append(
            "Invalid Asana API base URL. Expected an absolute URL, got "
            f"{asana_base_url!r}."
        )

    if errors:
        raise ConfigurationError("; ".join(errors))

    return AppConfig(
        github=GitHubConfig(token=github_token, api_base_url=github_base_url),
        asana=AsanaConfig(token=asana_token, api_base_url=asana_base_url),
    )
