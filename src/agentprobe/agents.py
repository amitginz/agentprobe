"""Agent adapters.

An agent is anything with `run(task_input) -> AgentResult`. Agents are selected by a
spec string so a run is fully described by plain text and can be reproduced later:

    mock:accuracy=0.9,seed=1,latency=25   deterministic simulated agent (demos, tests, CI)
    http:https://host/path        POSTs {"input": ...}, expects AgentResult-shaped JSON
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Protocol

import httpx


@dataclass
class AgentResult:
    output: str
    tool_calls: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class Agent(Protocol):
    def run(self, task_input: str, expected: str = "") -> AgentResult: ...


class MockAgent:
    """Simulated agent that answers correctly with a fixed probability.

    Correctness is a deterministic function of (seed, input), so reruns are identical and
    two mock agents with different accuracy form a realistic baseline/candidate pair.
    It needs `expected` to know what a correct answer looks like.
    """

    PRICE_IN = 3e-6
    PRICE_OUT = 15e-6

    def __init__(
        self, accuracy: float = 0.8, seed: int = 0, tools: str = "search", latency_ms: float = 0.0
    ) -> None:
        if not 0.0 <= accuracy <= 1.0:
            raise ValueError("accuracy must be between 0 and 1")
        self.accuracy = accuracy
        self.seed = seed
        self.tools = [t for t in tools.split("|") if t]
        self.latency_ms = latency_ms

    def _roll(self, task_input: str) -> float:
        digest = hashlib.sha256(f"{self.seed}:{task_input}".encode()).digest()
        return int.from_bytes(digest[:8], "big") / 2**64

    def run(self, task_input: str, expected: str = "") -> AgentResult:
        if self.latency_ms:  # simulated, deterministic jitter between 0.5x and 1.5x
            time.sleep(self.latency_ms * (0.5 + self._roll("latency:" + task_input)) / 1000)
        correct = self._roll(task_input) < self.accuracy
        output = expected if correct else "???"
        in_tok = max(1, len(task_input) // 4)
        out_tok = max(1, len(output) // 4)
        return AgentResult(
            output=output,
            tool_calls=list(self.tools),
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=in_tok * self.PRICE_IN + out_tok * self.PRICE_OUT,
        )


class HTTPAgent:
    """Calls a remote agent endpoint. Response JSON may contain any AgentResult field."""

    def __init__(self, url: str, timeout: float = 60.0) -> None:
        self.url = url
        self.timeout = timeout

    def run(self, task_input: str, expected: str = "") -> AgentResult:
        resp = httpx.post(self.url, json={"input": task_input}, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return AgentResult(
            output=str(data.get("output", "")),
            tool_calls=list(data.get("tool_calls", [])),
            input_tokens=int(data.get("input_tokens", 0)),
            output_tokens=int(data.get("output_tokens", 0)),
            cost_usd=float(data.get("cost_usd", 0.0)),
        )


def build_agent(spec: str) -> Agent:
    kind, _, rest = spec.partition(":")
    if kind == "mock":
        params = dict(p.split("=", 1) for p in rest.split(",") if p)
        return MockAgent(
            accuracy=float(params.get("accuracy", 0.8)),
            seed=int(params.get("seed", 0)),
            tools=params.get("tools", "search"),
            latency_ms=float(params.get("latency", 0)),
        )
    if kind == "http":
        if not rest:
            raise ValueError("http agent needs a URL: http:https://host/path")
        return HTTPAgent(rest)
    raise ValueError(f"unknown agent spec: {spec!r}")
