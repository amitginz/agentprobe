"""Statistical comparison of two runs on the same dataset.

Pass/fail outcomes are paired per task, so we use the exact McNemar test on the discordant
pairs rather than comparing two independent proportions. That is both more powerful and
correct for this design, and it needs nothing beyond the standard library.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .db import TaskResult


@dataclass
class RunStats:
    n: int
    pass_rate: float
    mean_cost_usd: float
    total_cost_usd: float
    mean_latency_ms: float
    p95_latency_ms: float
    errors: int


@dataclass
class Comparison:
    common_tasks: int
    baseline: RunStats
    candidate: RunStats
    pass_rate_delta: float
    fixed: int  # failed in baseline, passes in candidate
    broken: int  # passed in baseline, fails in candidate
    p_value: float
    significant: bool
    verdict: str  # regression | improvement | no_significant_change

    def to_dict(self) -> dict:
        return asdict(self)


def mcnemar_exact(broken: int, fixed: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts."""
    n = broken + fixed
    if n == 0:
        return 1.0
    k = min(broken, fixed)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, math.ceil(q * len(ordered)) - 1)
    return ordered[idx]


def run_stats(results: list[TaskResult]) -> RunStats:
    n = len(results)
    if n == 0:
        return RunStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    total_cost = sum(r.cost_usd for r in results)
    latencies = [r.latency_ms for r in results if not r.error]
    return RunStats(
        n=n,
        pass_rate=sum(r.passed for r in results) / n,
        mean_cost_usd=total_cost / n,
        total_cost_usd=total_cost,
        mean_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
        p95_latency_ms=percentile(latencies, 0.95),
        errors=sum(1 for r in results if r.error),
    )


def compare_runs(
    baseline: list[TaskResult], candidate: list[TaskResult], alpha: float = 0.05
) -> Comparison:
    base = {r.task_id: r for r in baseline}
    cand = {r.task_id: r for r in candidate}
    common = sorted(base.keys() & cand.keys())
    b_res, c_res = [base[t] for t in common], [cand[t] for t in common]

    broken = sum(1 for t in common if base[t].passed and not cand[t].passed)
    fixed = sum(1 for t in common if not base[t].passed and cand[t].passed)
    p = mcnemar_exact(broken, fixed)
    significant = p < alpha

    if significant and broken > fixed:
        verdict = "regression"
    elif significant and fixed > broken:
        verdict = "improvement"
    else:
        verdict = "no_significant_change"

    b_stats, c_stats = run_stats(b_res), run_stats(c_res)
    return Comparison(
        common_tasks=len(common),
        baseline=b_stats,
        candidate=c_stats,
        pass_rate_delta=c_stats.pass_rate - b_stats.pass_rate,
        fixed=fixed,
        broken=broken,
        p_value=p,
        significant=significant,
        verdict=verdict,
    )
