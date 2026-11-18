"""Deterministic judges. Each returns (passed, score in [0, 1])."""

from __future__ import annotations

import re
from collections.abc import Callable

from .agents import AgentResult

Judge = Callable[[str, dict, AgentResult], tuple[bool, float]]


def exact(expected: str, config: dict, result: AgentResult) -> tuple[bool, float]:
    a, b = result.output, expected
    if config.get("ignore_case", True):
        a, b = a.lower(), b.lower()
    ok = a.strip() == b.strip()
    return ok, 1.0 if ok else 0.0


def contains(expected: str, config: dict, result: AgentResult) -> tuple[bool, float]:
    a, b = result.output, expected
    if config.get("ignore_case", True):
        a, b = a.lower(), b.lower()
    ok = b.strip() in a
    return ok, 1.0 if ok else 0.0


def regex(expected: str, config: dict, result: AgentResult) -> tuple[bool, float]:
    ok = re.search(expected, result.output, re.IGNORECASE | re.DOTALL) is not None
    return ok, 1.0 if ok else 0.0


def tool_called(expected: str, config: dict, result: AgentResult) -> tuple[bool, float]:
    """Passes when every tool named in `expected` (comma separated) was called."""
    wanted = [t.strip() for t in expected.split(",") if t.strip()]
    if not wanted:
        return True, 1.0
    hits = sum(1 for t in wanted if t in result.tool_calls)
    return hits == len(wanted), hits / len(wanted)


JUDGES: dict[str, Judge] = {
    "exact": exact,
    "contains": contains,
    "regex": regex,
    "tool_called": tool_called,
}


def judge(name: str, expected: str, config: dict, result: AgentResult) -> tuple[bool, float]:
    try:
        fn = JUDGES[name]
    except KeyError:
        raise ValueError(f"unknown judge: {name!r} (available: {', '.join(JUDGES)})") from None
    return fn(expected, config, result)
