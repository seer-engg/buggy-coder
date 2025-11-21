from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:  # pragma: no cover - support script execution without package installation
    from .config import AsanaConfig
except ImportError:  # pragma: no cover - fallback when package context missing
    from config import AsanaConfig


class AsanaAPIError(RuntimeError):
    """Raised when the Asana API reports an error."""


class AsanaClient:
    def __init__(self, config: AsanaConfig, *, timeout: int = 15) -> None:
        self._token = config.token
        self._base_url = config.api_base_url.rstrip("/")
        self._timeout = timeout

    def mark_task_complete(self, task_gid: str) -> dict[str, Any] | None:
        payload = {"data": {"completed": True}}
        return self._request("PUT", f"/tasks/{task_gid}", payload)

    def add_comment_to_task(self, task_gid: str, text: str) -> dict[str, Any] | None:
        payload = {"data": {"text": text}}
        return self._request("POST", f"/tasks/{task_gid}/stories", payload)

    # Internal helpers ---------------------------------------------------------------

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = Request(url, data=data, method=method)
        request.add_header("Authorization", f"Bearer {self._token}")
        request.add_header("Accept", "application/json")
        request.add_header("Content-Type", "application/json")
        request.add_header("User-Agent", "buggy-coder-sync-agent")

        try:
            with urlopen(request, timeout=self._timeout) as response:  # nosec: B310 - trusted target
                payload = response.read()
        except HTTPError as exc:  # pragma: no cover - defensive
            body_text = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else ""
            raise AsanaAPIError(
                f"Asana API request failed with status {exc.code}: {body_text or exc.reason}"
            ) from exc
        except URLError as exc:  # pragma: no cover - defensive
            raise AsanaAPIError(f"Unable to reach Asana API: {exc.reason}") from exc

        if not payload:
            return {}

        try:
            return json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise AsanaAPIError("Received malformed JSON from Asana API") from exc
