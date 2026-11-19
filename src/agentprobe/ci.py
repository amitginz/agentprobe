"""One-shot CI gate: run a baseline and a candidate agent, compare, report, and set an exit code.

Exit codes:
    0  no regression (or --no-fail)
    1  statistically significant regression
    2  infrastructure problem (a run failed or too many tasks errored), so there is no verdict.
       This is deliberately separate from 1: an outage in the agent's endpoint must not be
       reported as "the change made the agent worse", and must not be silently treated as a pass.
"""

from __future__ import annotations

import html
import json
import os
from dataclasses import dataclass, field

from sqlalchemy import select

from .compare import Comparison, compare_runs
from .db import Database, Run, Task, TaskResult
from .telemetry import tracer

MAX_ERROR_RATE = 0.2
OK_STATUSES = ("completed", "budget_exceeded")


@dataclass
class CIOutcome:
    verdict: str  # regression | improvement | no_significant_change | infrastructure_error
    comparison: Comparison | None = None
    problems: list[str] = field(default_factory=list)
    broken: list[dict] = field(default_factory=list)
    fixed: list[dict] = field(default_factory=list)
    budget_limited: bool = False
    alpha: float = 0.05

    @property
    def exit_code_failure(self) -> int:
        return {"regression": 1, "infrastructure_error": 2}.get(self.verdict, 0)


def _flips(s, baseline: Run, candidate: Run) -> tuple[list[dict], list[dict]]:
    tasks = {t.id: t for t in s.scalars(select(Task).where(Task.dataset_id == baseline.dataset_id))}
    base = {r.task_id: r for r in baseline.results}
    cand = {r.task_id: r for r in candidate.results}

    def row(tid: int) -> dict:
        return {
            "input": tasks[tid].input,
            "baseline": base[tid].output,
            "candidate": cand[tid].output,
            "reasoning": cand[tid].judge_reasoning,
        }

    common = sorted(base.keys() & cand.keys())
    broken = [row(t) for t in common if base[t].passed and not cand[t].passed]
    fixed = [row(t) for t in common if not base[t].passed and cand[t].passed]
    return broken, fixed


def _health(run: Run, results: list[TaskResult], name: str) -> list[str]:
    problems = []
    if run.status not in OK_STATUSES:
        problems.append(f"{name} run ended as '{run.status}'" + (f": {run.error}" if run.error else ""))
    if not results:
        problems.append(f"{name} run produced no results")
    else:
        errors = sum(1 for r in results if r.error)
        if errors / len(results) > MAX_ERROR_RATE:
            sample = next(r.error for r in results if r.error)
            problems.append(
                f"{name}: {errors}/{len(results)} tasks errored (limit {MAX_ERROR_RATE:.0%}); e.g. {sample[:140]}"
            )
    return problems


def run_ci(
    db: Database,
    dataset_path: str,
    baseline_agent: str,
    candidate_agent: str,
    concurrency: int = 4,
    budget_usd: float | None = None,
    alpha: float = 0.05,
) -> CIOutcome:
    from .cli import _load_dataset  # same loader as `agentprobe run`
    from .runner import execute_run

    dataset_id = _load_dataset(db, dataset_path)
    ids = {}
    # One trace for the whole gate: ci -> (baseline run, candidate run) -> tasks -> agent/judge calls.
    with tracer().start_as_current_span("agentprobe.ci", attributes={"agentprobe.alpha": alpha}):
        for label, agent in (("baseline", baseline_agent), ("candidate", candidate_agent)):
            with db.session() as s:
                run = Run(
                    dataset_id=dataset_id, agent=agent, label=label, concurrency=concurrency, budget_usd=budget_usd
                )
                s.add(run)
                s.commit()
                ids[label] = run.id
            execute_run(db, ids[label])  # sequential: the two runs must not compete for one endpoint

    with db.session() as s:
        b, c = s.get(Run, ids["baseline"]), s.get(Run, ids["candidate"])
        problems = _health(b, list(b.results), "baseline") + _health(c, list(c.results), "candidate")
        if problems:
            return CIOutcome("infrastructure_error", problems=problems, alpha=alpha)
        cmp = compare_runs(list(b.results), list(c.results), alpha)
        broken, fixed = _flips(s, b, c)
        limited = "budget_exceeded" in (b.status, c.status)
        return CIOutcome(cmp.verdict, cmp, [], broken, fixed, limited, alpha)


# ---------- reporting ----------

def _cell(text: str, limit: int = 110) -> str:
    """Make untrusted text safe for a markdown table cell (agent output can contain anything)."""
    flat = " ".join(str(text).split())
    if len(flat) > limit:
        flat = flat[: limit - 1] + "…"
    return html.escape(flat).replace("|", "\\|").replace("`", "&#96;")


def _flip_table(rows: list[dict], title: str, max_rows: int = 10) -> str:
    if not rows:
        return ""
    body = "\n".join(
        f"| {_cell(r['input'], 80)} | {_cell(r['baseline'])} | {_cell(r['candidate'])} |" for r in rows[:max_rows]
    )
    more = f"\n\n…and {len(rows) - max_rows} more." if len(rows) > max_rows else ""
    return (
        f"<details><summary>{title} ({len(rows)})</summary>\n\n"
        f"| Task | Baseline output | Candidate output |\n|---|---|---|\n{body}{more}\n\n</details>\n"
    )


def _sensitivity_line(c: Comparison) -> str:
    """Say how much this dataset can actually detect, so 'no significant change' is read correctly."""
    from .power import min_detectable_effect

    if c.common_tasks == 0:
        return ""
    observed = (c.broken + c.fixed) / c.common_tasks
    discordance = min(1.0, max(observed, 0.05))  # floor: a quiet run must not claim infinite power
    mde = min_detectable_effect(c.common_tasks, discordance)
    basis = f"{c.common_tasks} tasks, about {discordance:.0%} of them flipping between versions"
    if mde is None:
        return f"<sub>Sensitivity: with {basis}, this comparison cannot reach 80% power, so a non-significant result says little.</sub>"
    return f"<sub>Sensitivity: with {basis}, changes of {mde * 100:.0f} points or more are detected with 80% power.</sub>"


def render_markdown(o: CIOutcome, baseline_agent: str = "", candidate_agent: str = "") -> str:
    if o.verdict == "infrastructure_error":
        items = "\n".join(f"- {_cell(p, 300)}" for p in o.problems)
        return (
            "## AgentProbe: ⚠️ No verdict (infrastructure problem)\n\n"
            "The comparison could not be made, so this is **not** a statement about the change.\n\n"
            f"{items}\n"
        )
    c = o.comparison
    assert c is not None
    head = {
        "regression": "❌ Regression detected",
        "improvement": "✅ Improvement",
        "no_significant_change": "✅ No significant change",
    }[o.verdict]
    delta = c.pass_rate_delta * 100
    cost_b, cost_c = c.baseline.mean_cost_usd, c.candidate.mean_cost_usd
    cost_row = (
        f"| Mean cost / task | ${cost_b:.6f} | ${cost_c:.6f} | |\n" if (cost_b or cost_c) else ""
    )
    sig = "significant" if c.significant else "not significant"
    out = (
        f"## AgentProbe: {head}\n\n"
        f"| | Baseline | Candidate | Δ |\n|---|---|---|---|\n"
        f"| Pass rate | {c.baseline.pass_rate:.1%} | {c.candidate.pass_rate:.1%} | {delta:+.1f} pts |\n"
        f"{cost_row}"
        f"| p95 latency | {c.baseline.p95_latency_ms:.0f} ms | {c.candidate.p95_latency_ms:.0f} ms | |\n"
        f"| Errored tasks | {c.baseline.errors} | {c.candidate.errors} | |\n\n"
        f"**{c.broken} broke · {c.fixed} fixed** across {c.common_tasks} shared tasks. "
        f"Exact McNemar p = {c.p_value:.4f} ({sig} at α = {o.alpha}).\n\n"
    )
    out += _sensitivity_line(c) + "\n\n"
    if baseline_agent or candidate_agent:
        out += f"<sub>baseline `{_cell(baseline_agent, 80)}` · candidate `{_cell(candidate_agent, 80)}`</sub>\n\n"
    if o.budget_limited:
        out += "> ⚠️ A run hit its cost budget and stopped early; only tasks both runs finished are compared.\n\n"
    out += _flip_table(o.broken, "Tasks the candidate broke") + _flip_table(o.fixed, "Tasks the candidate fixed")
    return out


def write_github_outputs(o: CIOutcome, markdown: str) -> None:
    """Publish the job summary and step outputs when running under GitHub Actions."""
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(path, "a") as fh:
            fh.write(markdown + "\n")
    if path := os.environ.get("GITHUB_OUTPUT"):
        c = o.comparison
        values = {
            "verdict": o.verdict,
            "baseline-pass-rate": f"{c.baseline.pass_rate:.4f}" if c else "",
            "candidate-pass-rate": f"{c.candidate.pass_rate:.4f}" if c else "",
            "p-value": f"{c.p_value:.6f}" if c else "",
            "broken": str(c.broken) if c else "",
            "fixed": str(c.fixed) if c else "",
        }
        with open(path, "a") as fh:
            fh.writelines(f"{key}={value}\n" for key, value in values.items())


def to_json(o: CIOutcome) -> str:
    return json.dumps(
        {"verdict": o.verdict, "problems": o.problems, "comparison": o.comparison.to_dict() if o.comparison else None},
        indent=2,
    )
