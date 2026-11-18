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

from opentelemetry import context as otel_context
from opentelemetry.trace import Status, StatusCode
from sqlalchemy import select, update

from .agents import Agent, AgentResult, build_agent
from .db import Database, Run, Task, TaskResult, utcnow
from .judges import judge
from .telemetry import AGENT_LATENCY, COST, TASKS, agent_kind, mark_error, restore_context, tracer


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
    judge_reasoning: str = ""
    judge_cost_usd: float = 0.0


def run_task(agent: Agent, task: TaskSpec, retries: int, parent_ctx=None) -> Outcome:
    """Run one task inside a span. `parent_ctx` carries the run span across the thread pool."""
    with tracer().start_as_current_span(
        "agentprobe.task",
        context=parent_ctx,
        attributes={"agentprobe.task.id": task.id, "agentprobe.judge": task.judge},
    ) as span:
        out = _run_task(agent, task, retries)
        span.set_attributes(
            {
                "agentprobe.passed": out.passed,
                "agentprobe.attempts": out.attempts,
                "agentprobe.latency_ms": out.latency_ms,
                "agentprobe.cost_usd": out.result.cost_usd + out.judge_cost_usd,
            }
        )
        if out.error:
            span.set_status(Status(StatusCode.ERROR, out.error[:200]))
        attrs = {"judge": task.judge, "passed": out.passed, "errored": bool(out.error)}
        TASKS.add(1, attrs)
        COST.add(out.result.cost_usd + out.judge_cost_usd, {"judge": task.judge})
        if not out.error:
            AGENT_LATENCY.record(out.latency_ms, {"judge": task.judge})
        return out


def _run_task(agent: Agent, task: TaskSpec, retries: int) -> Outcome:
    """Agent exceptions are retried; the final failure is recorded, not raised."""
    last_error = ""
    attempts = 0
    for attempt in range(retries + 1):
        attempts = attempt + 1
        start = time.perf_counter()
        with tracer().start_as_current_span("agent.call", attributes={"agentprobe.attempt": attempts}) as aspan:
            try:
                result = agent.run(task.input, task.expected)
            except Exception as exc:  # noqa: BLE001 - agent code is untrusted
                mark_error(aspan, exc)
                last_error = f"{type(exc).__name__}: {exc}"
                continue
            aspan.set_attributes(
                {
                    "gen_ai.usage.input_tokens": result.input_tokens,
                    "gen_ai.usage.output_tokens": result.output_tokens,
                    "agentprobe.cost_usd": result.cost_usd,
                    "agentprobe.tool_calls": len(result.tool_calls),
                    "agentprobe.output_chars": len(result.output),
                }
            )
        latency = (time.perf_counter() - start) * 1000
        with tracer().start_as_current_span("judge.evaluate", attributes={"agentprobe.judge": task.judge}) as jspan:
            try:
                v = judge(task.judge, task.expected, task.judge_config, result, task.input)
            except Exception as exc:  # noqa: BLE001 - a flaky judge must not crash the run
                # The agent did answer, so keep its output, but mark the task as errored (not
                # failed on merit) so it is visible and excluded from latency statistics.
                mark_error(jspan, exc)
                err = f"judge failed: {type(exc).__name__}: {exc}"
                return Outcome(task.id, False, 0.0, result, latency, attempts, err)
            jspan.set_attributes({"agentprobe.passed": v.passed, "agentprobe.judge_cost_usd": v.cost_usd})
        return Outcome(
            task.id, v.passed, v.score, result, latency, attempts, "", v.reasoning, v.cost_usd
        )
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
            tid: cost + jcost
            for tid, cost, jcost in s.execute(
                select(TaskResult.task_id, TaskResult.cost_usd, TaskResult.judge_cost_usd).where(
                    TaskResult.run_id == run_id
                )
            )
        }
        tasks = [
            TaskSpec(t.id, t.input, t.expected, t.judge, t.judge_config)
            for t in s.scalars(select(Task).where(Task.dataset_id == run.dataset_id).order_by(Task.id))
            if t.id not in done
        ]
        spec, concurrency, retries, budget = run.agent, run.concurrency, run.retries, run.budget_usd
        trace_ctx = run.trace_context

    parent_ctx = restore_context(trace_ctx)  # None keeps whatever span is current (CLI, tests)
    with tracer().start_as_current_span(
        "agentprobe.run",
        context=parent_ctx,
        attributes={
            "agentprobe.run.id": run_id,
            "agentprobe.agent.kind": agent_kind(spec),
            "agentprobe.concurrency": concurrency,
            "agentprobe.tasks_pending": len(tasks),
            "agentprobe.tasks_resumed": len(done),
        },
    ) as span:
        run_ctx = otel_context.get_current()
        status, error = "completed", ""
        try:
            agent = agent or build_agent(spec)
            spent = sum(done.values())  # resumed runs keep the spend of earlier attempts
            over_budget = budget is not None and spent > budget
            if over_budget:
                status, tasks = "budget_exceeded", []
            with db.session() as s, ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
                futures = {pool.submit(run_task, agent, task, retries, run_ctx): task for task in tasks}
                for fut in as_completed(futures):
                    if over_budget:
                        continue
                    out = fut.result()
                    spent += out.result.cost_usd + out.judge_cost_usd
                    s.add(_to_row(run_id, out))
                    s.commit()  # per-result commit: a crash loses at most the in-flight tasks
                    if budget is not None and spent > budget:
                        over_budget = True
                        status = "budget_exceeded"
                        for pending in futures:
                            pending.cancel()
        except Exception as exc:  # noqa: BLE001
            status, error = "failed", f"{type(exc).__name__}: {exc}"
            mark_error(span, exc)

        with db.session() as s:
            run = s.get(Run, run_id)
            run.status, run.error, run.finished_at = status, error, utcnow()
            s.commit()
            rows = s.execute(
                select(TaskResult.passed, TaskResult.cost_usd, TaskResult.judge_cost_usd).where(
                    TaskResult.run_id == run_id
                )
            ).all()
        span.set_attributes(
            {
                "agentprobe.run.status": status,
                "agentprobe.tasks_total": len(rows),
                "agentprobe.pass_rate": (sum(r[0] for r in rows) / len(rows)) if rows else 0.0,
                "agentprobe.cost_usd": sum(r[1] + r[2] for r in rows),
            }
        )
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
        judge_reasoning=out.judge_reasoning,
        judge_cost_usd=out.judge_cost_usd,
    )
