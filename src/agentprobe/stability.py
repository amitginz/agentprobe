"""Run-to-run stability of a non-deterministic agent.

LLM agents give different answers to the same input, so one run is one noisy sample. Repeating the
same agent on the same dataset k times shows how much of any observed change is just noise:

* **flaky tasks** pass in some repeats and fail in others (they carry the noise);
* the **standard deviation of the pass rate** across repeats is the noise floor: a candidate that
  differs from the baseline by less than about two of these is indistinguishable from re-running
  the baseline.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass

from .db import TaskResult


@dataclass
class Stability:
    repeats: int
    tasks: int
    mean_pass_rate: float
    pass_rate_sd: float
    noise_floor_pts: float  # 2 x sd, in percentage points
    always_pass: int
    always_fail: int
    flaky: int
    flaky_fraction: float
    flaky_tasks: list[dict]  # most unstable first: {"task_id", "passes", "repeats"}

    def to_dict(self) -> dict:
        return asdict(self)


def analyze(runs: list[list[TaskResult]]) -> Stability:
    """`runs` holds the results of k >= 2 repeats of the same agent on the same dataset."""
    if len(runs) < 2:
        raise ValueError("stability needs at least 2 repeats")
    per_run = [{r.task_id: r.passed for r in results} for results in runs]
    common = sorted(set.intersection(*(set(p) for p in per_run)))
    if not common:
        raise ValueError("the repeats share no tasks")

    k = len(runs)
    passes = {t: sum(p[t] for p in per_run) for t in common}
    rates = [sum(p[t] for t in common) / len(common) for p in per_run]
    flaky_ids = [t for t in common if 0 < passes[t] < k]
    # least stable = closest to a coin flip
    flaky_ids.sort(key=lambda t: (abs(passes[t] - k / 2), t))
    sd = statistics.stdev(rates)
    return Stability(
        repeats=k,
        tasks=len(common),
        mean_pass_rate=statistics.fmean(rates),
        pass_rate_sd=sd,
        noise_floor_pts=2 * sd * 100,
        always_pass=sum(1 for t in common if passes[t] == k),
        always_fail=sum(1 for t in common if passes[t] == 0),
        flaky=len(flaky_ids),
        flaky_fraction=len(flaky_ids) / len(common),
        flaky_tasks=[{"task_id": t, "passes": passes[t], "repeats": k} for t in flaky_ids],
    )
