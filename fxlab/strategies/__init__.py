"""Strategy library public API.

``build_all()`` expands ``configs/strategies.yaml`` into exactly 300 unique
Strategy instances. ``get(id)`` fetches one by id; ``families()`` lists distinct
family keys; ``ids()`` lists all strategy ids in deterministic order.
"""

from __future__ import annotations

from functools import lru_cache

from ..core import Strategy
from .registry import build_all as _build_all

__all__ = ["build_all", "get", "families", "ids"]


def build_all() -> list[Strategy]:
    """Return the full ordered list of 300 strategy instances."""
    return _build_all()


@lru_cache(maxsize=1)
def _index() -> dict[str, Strategy]:
    return {s.id: s for s in _build_all()}


def get(id: str) -> Strategy:
    """Return the strategy with the given id (raises KeyError if absent)."""
    idx = _index()
    if id not in idx:
        raise KeyError(f"unknown strategy id: {id}")
    return idx[id]


def ids() -> list[str]:
    """Return all strategy ids in deterministic build order."""
    return [s.id for s in _build_all()]


def families() -> list[str]:
    """Return distinct family keys in first-seen order."""
    out: list[str] = []
    seen: set[str] = set()
    for s in _build_all():
        if s.family not in seen:
            seen.add(s.family)
            out.append(s.family)
    return out
