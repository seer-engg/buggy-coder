"""Validation helpers for numerical inputs."""

from __future__ import annotations


def parse_positive_int(value: object) -> int:
    """Parse ``value`` as a strictly positive ``int``.

    Raises:
        ValueError: If ``value`` cannot be converted to an ``int`` or is not
            strictly positive. ``bool`` values are rejected because ``bool`` is a
            subclass of ``int`` in Python but semantically represents logical
            values.
    """
    if isinstance(value, bool):
        raise ValueError("value must be a positive integer")

    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("value must be a positive integer") from exc

    if number <= 0:
        raise ValueError("value must be a positive integer")

    return number
