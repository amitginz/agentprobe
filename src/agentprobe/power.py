"""Statistical power for the paired exact test: how big a change can a dataset actually detect?

A comparison of two agents on N tasks only has information in the *discordant* tasks, those one
version passes and the other fails. Let ``p_broken`` / ``p_fixed`` be the probabilities that a task
flips in each direction. Then the discordance rate is ``d = p_broken + p_fixed`` and the true
change in pass rate is ``effect = p_fixed - p_broken``.

Power is computed exactly, not by a normal approximation, because evaluation sets are small:

* the number of discordant tasks is Binomial(N, d);
* given D discordant tasks, the number that were fixed is Binomial(D, p_fixed / d);
* the exact two-sided McNemar test rejects when min(broken, fixed) <= k_crit(D).

This is the tool that explains an "inconclusive" result: with 25 tasks and a discordance rate near
8%, only enormous changes are detectable, so "not significant" says very little.
"""

from __future__ import annotations

import math
from functools import lru_cache


def _log_binom_pmf(n: int, k: int, p: float) -> float:
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return (
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        + k * math.log(p) + (n - k) * math.log1p(-p)
    )


def _binom_cdf(n: int, k: int, p: float) -> float:
    if k < 0:
        return 0.0
    return min(1.0, sum(math.exp(_log_binom_pmf(n, i, p)) for i in range(min(k, n) + 1)))


@lru_cache(maxsize=4096)
def critical_count(discordant: int, alpha: float) -> int:
    """Largest k such that min(broken, fixed) <= k rejects at `alpha` (-1 if it never can)."""
    best = -1
    for k in range(discordant // 2 + 1):
        p = min(1.0, 2 * _binom_cdf(discordant, k, 0.5))
        if p < alpha:
            best = k
        else:
            break
    return best


def power(n_tasks: int, p_broken: float, p_fixed: float, alpha: float = 0.05) -> float:
    """Probability that the exact McNemar test detects a difference (either direction)."""
    d = p_broken + p_fixed
    if n_tasks <= 0 or d <= 0.0:
        return 0.0
    if d > 1.0 or p_broken < 0 or p_fixed < 0:
        raise ValueError("p_broken and p_fixed must be non-negative and sum to at most 1")
    q = p_fixed / d
    total = 0.0
    for discordant in range(1, n_tasks + 1):
        weight = math.exp(_log_binom_pmf(n_tasks, discordant, d))
        if weight < 1e-14:
            continue
        k = critical_count(discordant, alpha)
        if k < 0:
            continue
        lower = _binom_cdf(discordant, k, q)  # few fixed: significant regression
        upper = _binom_cdf(discordant, k, 1 - q)  # few broken: significant improvement
        total += weight * min(1.0, lower + upper)
    return min(1.0, total)


def _split(discordance: float, effect: float) -> tuple[float, float]:
    """Return (p_broken, p_fixed) for a given discordance rate and net effect on pass rate."""
    return (discordance - effect) / 2, (discordance + effect) / 2


def power_for_effect(n_tasks: int, discordance: float, effect: float, alpha: float = 0.05) -> float:
    if abs(effect) > discordance:
        raise ValueError("|effect| cannot exceed the discordance rate")
    p_b, p_f = _split(discordance, abs(effect))
    return power(n_tasks, p_b, p_f, alpha)


def min_detectable_effect(
    n_tasks: int, discordance: float, alpha: float = 0.05, target_power: float = 0.8
) -> float | None:
    """Smallest pass-rate change detected with `target_power`. None if even the largest is not enough."""
    if power_for_effect(n_tasks, discordance, discordance, alpha) < target_power:
        return None
    lo, hi = 0.0, discordance
    for _ in range(25):
        mid = (lo + hi) / 2
        if power_for_effect(n_tasks, discordance, mid, alpha) >= target_power:
            hi = mid
        else:
            lo = mid
    return hi


def required_tasks(
    effect: float, discordance: float, alpha: float = 0.05, target_power: float = 0.8, limit: int = 5000
) -> int | None:
    """Approximate smallest dataset size reaching `target_power` (power is slightly saw-toothed in N)."""
    if abs(effect) > discordance or effect == 0:
        raise ValueError("need 0 < |effect| <= discordance")
    n = 10
    while power_for_effect(n, discordance, effect, alpha) < target_power:
        n *= 2
        if n > limit:
            return None
    lo, hi = n // 2, n
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if power_for_effect(mid, discordance, effect, alpha) >= target_power:
            hi = mid
        else:
            lo = mid
    return hi
