from __future__ import annotations

import re
from collections import OrderedDict
from typing import Iterable

ASANA_TASK_URL_PATTERN = re.compile(
    r"https://app\.asana\.com/0/[\w-]+/(?P<gid>\d+)(?:(?:[/?#][^\s)]+)?)",
    re.IGNORECASE,
)


def extract_asana_task_gids(text: str | None) -> list[str]:
    """Extract unique Asana task GIDs from the provided text.

    The function is resilient to trailing slashes, query parameters, and additional
    path segments that may follow the task identifier.
    """

    if not text:
        return []

    # OrderedDict preserves insertion order while removing duplicates.
    unique_gids: "OrderedDict[str, None]" = OrderedDict()
    for match in ASANA_TASK_URL_PATTERN.finditer(text):
        gid = match.group("gid")
        if gid and gid not in unique_gids:
            unique_gids[gid] = None
    return list(unique_gids.keys())


def normalise_gids(gids: Iterable[str]) -> list[str]:
    """Normalise and deduplicate a collection of GIDs, filtering out falsy values."""

    unique_gids: "OrderedDict[str, None]" = OrderedDict()
    for gid in gids:
        gid_str = str(gid).strip()
        if gid_str and gid_str.isdigit() and gid_str not in unique_gids:
            unique_gids[gid_str] = None
    return list(unique_gids.keys())
