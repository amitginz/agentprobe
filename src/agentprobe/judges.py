"""Judges. Deterministic ones are free and exact; the `llm` judge handles open-ended answers.

Every judge function returns ``(passed, score)``, except `llm`, which also returns its reasoning
and cost. `judge()` normalizes all of them into a `Verdict`.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from .agents import AgentResult
from .llm_judge import LLMJudge, build_default_judge


@dataclass
class Verdict:
    passed: bool
    score: float
    reasoning: str = ""
    cost_usd: float = 0.0


def exact(expected: str, config: dict, result: AgentResult, task_input: str = "") -> tuple[bool, float]:
    a, b = result.output, expected
    if config.get("ignore_case", True):
        a, b = a.lower(), b.lower()
    ok = a.strip() == b.strip()
    return ok, 1.0 if ok else 0.0


def contains(expected: str, config: dict, result: AgentResult, task_input: str = "") -> tuple[bool, float]:
    a, b = result.output, expected
    if config.get("ignore_case", True):
        a, b = a.lower(), b.lower()
    ok = b.strip() in a
    return ok, 1.0 if ok else 0.0


def regex(expected: str, config: dict, result: AgentResult, task_input: str = "") -> tuple[bool, float]:
    ok = re.search(expected, result.output, re.IGNORECASE | re.DOTALL) is not None
    return ok, 1.0 if ok else 0.0


def tool_called(expected: str, config: dict, result: AgentResult, task_input: str = "") -> tuple[bool, float]:
    """Passes when every tool named in `expected` (comma separated) was called."""
    wanted = [t.strip() for t in expected.split(",") if t.strip()]
    if not wanted:
        return True, 1.0
    hits = sum(1 for t in wanted if t in result.tool_calls)
    return hits == len(wanted), hits / len(wanted)


DETERMINISTIC: dict[str, Callable[..., tuple[bool, float]]] = {
    "exact": exact,
    "contains": contains,
    "regex": regex,
    "tool_called": tool_called,
}
JUDGE_NAMES = sorted([*DETERMINISTIC, "llm"])

_llm_judge: LLMJudge | None = None


def set_llm_judge(j: LLMJudge | None) -> None:
    """Install a specific LLM judge (tests, or a custom model/client). None resets to default."""
    global _llm_judge
    _llm_judge = j


def _get_llm_judge() -> LLMJudge:
    global _llm_judge
    if _llm_judge is None:
        _llm_judge = build_default_judge()  # raises a clear error if no API key is configured
    return _llm_judge


def judge(
    name: str, expected: str, config: dict, result: AgentResult, task_input: str = ""
) -> Verdict:
    if name == "llm":
        v = _get_llm_judge().evaluate(
            task_input=task_input,
            expected=expected,
            output=result.output,
            rubric=config.get("rubric", ""),
        )
        return Verdict(v.passed, v.score, v.reasoning, v.cost_usd)
    try:
        fn = DETERMINISTIC[name]
    except KeyError:
        raise ValueError(f"unknown judge: {name!r} (available: {', '.join(JUDGE_NAMES)})") from None
    passed, score = fn(expected, config, result, task_input)
    return Verdict(passed, score)
