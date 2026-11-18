"""Run execution: bounded concurrency, retries, a hard cost budget, and crash-safe resume.

Task threads only call the agent and judge; all DB writes happen on the coordinating thread,
which keeps SQLite happy and makes result persistence a single, ordered code path.

Two properties make runs safe to execute from a work queue with at-least-once delivery:

* ``claim_run`` is an atomic compare-and-set (``pending -> running``), so if the same run is
  delivered twice only one worker executes it.
* ``execute_run`` skips tasks that already have a stored result, so a run re-delivered after a
  worker crash resumes where it left off instead of starting over (and re-spending budget).
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from sqlalchemy import select, update

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


def claim_run(db: Database, run_id: int) -> bool:
    """Atomically move a run from pending to running. True only for the single winner."""
    with db.session() as s:
        res = s.execute(
            update(Run).where(Run.id == run_id, Run.status == "pending").values(status="running")
        )
        s.commit()
        return res.rowcount == 1


def release_run(db: Database, run_id: int) -> None:
    """Return a run abandoned by a dead worker to pending so it can be claimed again."""
    with db.session() as s:
        s.execute(
            update(Run).where(Run.id == run_id, Run.status == "running").values(status="pending")
        )
        s.commit()


def fail_run(db: Database, run_id: int, error: str) -> None:
    with db.session() as s:
        s.execute(
            update(Run)
            .where(Run.id == run_id, Run.status.in_(["pending", "running"]))
            .values(status="failed", error=error, finished_at=utcnow())
        )
        s.commit()


def execute_run(db: Database, run_id: int, agent: Agent | None = None) -> bool:
    """Execute a run. Returns False if another worker already owns it (or it is finished)."""
    if not claim_run(db, run_id):
        return False

    with db.session() as s:
        run = s.get(Run, run_id)
        done = {
            tid: cost
            for tid, cost in s.execute(
                select(TaskResult.task_id, TaskResult.cost_usd).where(TaskResult.run_id == run_id)
            )
        }
        tasks = [
            TaskSpec(t.id, t.input, t.expected, t.judge, t.judge_config)
            for t in s.scalars(select(Task).where(Task.dataset_id == run.dataset_id).order_by(Task.id))
            if t.id not in done
        ]
        spec, concurrency, retries, budget = run.agent, run.concurrency, run.retries, run.budget_usd

    status, error = "completed", ""
    try:
        agent = agent or build_agent(spec)
        spent = sum(done.values())  # resumed runs keep the spend of earlier attempts
        over_budget = budget is not None and spent > budget
        if over_budget:
            status, tasks = "budget_exceeded", []
        with db.session() as s, ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = {pool.submit(run_task, agent, task, retries): task for task in tasks}
            for fut in as_completed(futures):
                if over_budget:
                    continue
                out = fut.result()
                spent += out.result.cost_usd
                s.add(_to_row(run_id, out))
                s.commit()  # per-result commit: a crash loses at most the in-flight tasks
                if budget is not None and spent > budget:
                    over_budget = True
                    status = "budget_exceeded"
                    for pending in futures:
                        pending.cancel()
    except Exception as exc:  # noqa: BLE001
        status, error = "failed", f"{type(exc).__name__}: {exc}"

    with db.session() as s:
        run = s.get(Run, run_id)
        run.status, run.error, run.finished_at = status, error, utcnow()
        s.commit()
    return True


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
