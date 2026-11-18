"""Run execution: bounded concurrency, retries, and a hard cost budget.

Workers only call the agent and judge; all DB writes happen on the coordinating thread,
which keeps SQLite happy and makes result persistence a single, ordered code path.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from sqlalchemy import select

from .agents import Agent, AgentResult, build_agent
from .db import Database, Run, Task, TaskResult, utcnow
from .judges import judge


@dataclass
class TaskSpec:
    id: int
    input: str
    expected: str
    judge: str
    judge_config: dict


@dataclass
class Outcome:
    task_id: int
    passed: bool
    score: float
    result: AgentResult
    latency_ms: float
    attempts: int
    error: str


def run_task(agent: Agent, task: TaskSpec, retries: int) -> Outcome:
    """Run one task. Agent exceptions are retried; the final failure is recorded, not raised."""
    last_error = ""
    attempts = 0
    for attempt in range(retries + 1):
        attempts = attempt + 1
        start = time.perf_counter()
        try:
            result = agent.run(task.input, task.expected)
        except Exception as exc:  # noqa: BLE001 - agent code is untrusted
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        latency = (time.perf_counter() - start) * 1000
        passed, score = judge(task.judge, task.expected, task.judge_config, result)
        return Outcome(task.id, passed, score, result, latency, attempts, "")
    return Outcome(task.id, False, 0.0, AgentResult(output=""), 0.0, attempts, last_error)


def execute_run(db: Database, run_id: int, agent: Agent | None = None) -> None:
    with db.session() as s:
        run = s.get(Run, run_id)
        if run is None:
            raise LookupError(f"run {run_id} not found")
        tasks = [
            TaskSpec(t.id, t.input, t.expected, t.judge, t.judge_config)
            for t in s.scalars(select(Task).where(Task.dataset_id == run.dataset_id).order_by(Task.id))
        ]
        spec, concurrency, retries, budget = run.agent, run.concurrency, run.retries, run.budget_usd
        run.status = "running"
        s.commit()

    status, error = "completed", ""
    try:
        agent = agent or build_agent(spec)
        spent, over_budget = 0.0, False
        with db.session() as s, ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = {}
            for task in tasks:
                futures[pool.submit(run_task, agent, task, retries)] = task
            for fut in as_completed(futures):
                if over_budget:
                    continue
                out = fut.result()
                spent += out.result.cost_usd
                s.add(_to_row(run_id, out))
                if budget is not None and spent > budget:
                    over_budget = True
                    status = "budget_exceeded"
                    for pending in futures:
                        pending.cancel()
            s.commit()
    except Exception as exc:  # noqa: BLE001
        status, error = "failed", f"{type(exc).__name__}: {exc}"

    with db.session() as s:
        run = s.get(Run, run_id)
        run.status, run.error, run.finished_at = status, error, utcnow()
        s.commit()


def _to_row(run_id: int, out: Outcome) -> TaskResult:
    r = out.result
    return TaskResult(
        run_id=run_id,
        task_id=out.task_id,
        passed=out.passed,
        score=out.score,
        output=r.output,
        tool_calls=r.tool_calls,
        latency_ms=out.latency_ms,
        input_tokens=r.input_tokens,
        output_tokens=r.output_tokens,
        cost_usd=r.cost_usd,
        attempts=out.attempts,
        error=out.error,
    )
