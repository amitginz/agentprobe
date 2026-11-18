"""REST API."""

from __future__ import annotations

import os

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select

from . import __version__
from .agents import build_agent
from .calibration import label_sample, run_calibration
from .compare import compare_runs, run_stats
from .db import Database, Dataset, HumanLabel, Run, Task, TaskResult
from .jobqueue import RedisQueue
from .judges import JUDGE_NAMES
from .runner import execute_run, fail_run


class TaskIn(BaseModel):
    input: str
    expected: str = ""
    judge: str = "contains"
    judge_config: dict = Field(default_factory=dict)


class DatasetIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    tasks: list[TaskIn] = Field(min_length=1)


class LabelIn(BaseModel):
    task_id: int
    passed: bool


class LabelsIn(BaseModel):
    labels: list[LabelIn] = Field(min_length=1)


class RunIn(BaseModel):
    dataset_id: int
    agent: str
    label: str = ""
    concurrency: int = Field(default=4, ge=1, le=64)
    retries: int = Field(default=1, ge=0, le=5)
    budget_usd: float | None = Field(default=None, gt=0)


def create_app(
    database_url: str | None = None,
    run_inline: bool = False,
    queue: RedisQueue | None = None,
) -> FastAPI:
    """App factory.

    Execution mode: `run_inline` runs synchronously (tests); a `queue` hands runs to separate
    worker processes; with neither, runs execute in a background task of this process.
    """
    db = Database(database_url)
    app = FastAPI(title="AgentProbe", version=__version__)

    @app.get("/health")
    def health() -> dict:
        info: dict = {"status": "ok", "version": __version__, "execution": "inline"}
        if queue is not None:
            info["execution"] = "queue"
            try:
                info["queue"] = queue.stats()
            except Exception:  # noqa: BLE001
                info["status"] = "degraded"
                info["queue"] = "unreachable"
        elif not run_inline:
            info["execution"] = "in-process"
        return info

    @app.post("/datasets", status_code=201)
    def create_dataset(body: DatasetIn) -> dict:
        for t in body.tasks:
            if t.judge not in JUDGE_NAMES:
                raise HTTPException(422, f"unknown judge {t.judge!r}; available: {JUDGE_NAMES}")
        with db.session() as s:
            if s.scalar(select(Dataset).where(Dataset.name == body.name)):
                raise HTTPException(409, f"dataset {body.name!r} already exists")
            ds = Dataset(name=body.name, tasks=[Task(**t.model_dump()) for t in body.tasks])
            s.add(ds)
            s.commit()
            return {"id": ds.id, "name": ds.name, "tasks": len(ds.tasks)}

    @app.get("/datasets")
    def list_datasets() -> list[dict]:
        with db.session() as s:
            return [
                {"id": d.id, "name": d.name, "tasks": len(d.tasks)}
                for d in s.scalars(select(Dataset).order_by(Dataset.id))
            ]

    @app.post("/runs", status_code=202)
    def create_run(body: RunIn, background: BackgroundTasks) -> dict:
        try:
            build_agent(body.agent)  # validate the spec up front
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        with db.session() as s:
            if s.get(Dataset, body.dataset_id) is None:
                raise HTTPException(404, "dataset not found")
            run = Run(**body.model_dump())
            s.add(run)
            s.commit()
            run_id = run.id
        if run_inline:
            execute_run(db, run_id)
        elif queue is not None:
            try:
                queue.enqueue(run_id)
            except Exception as exc:
                fail_run(db, run_id, f"could not enqueue: {exc}")
                raise HTTPException(503, "job queue unavailable") from exc
        else:
            background.add_task(execute_run, db, run_id)
        return {"id": run_id, "status": "pending"}

    @app.get("/runs/{run_id}")
    def get_run(run_id: int) -> dict:
        with db.session() as s:
            run = _get_run(s, run_id)
            stats = run_stats(list(run.results))
            return {
                "id": run.id,
                "label": run.label,
                "agent": run.agent,
                "status": run.status,
                "error": run.error,
                "stats": stats.__dict__,
            }

    @app.get("/runs/{run_id}/results")
    def get_results(run_id: int) -> list[dict]:
        with db.session() as s:
            run = _get_run(s, run_id)
            return [
                {
                    "task_id": r.task_id,
                    "passed": r.passed,
                    "output": r.output,
                    "latency_ms": round(r.latency_ms, 2),
                    "cost_usd": r.cost_usd,
                    "attempts": r.attempts,
                    "error": r.error,
                    "judge_reasoning": r.judge_reasoning,
                }
                for r in sorted(run.results, key=lambda r: r.task_id)
            ]

    @app.get("/compare")
    def compare(baseline: int, candidate: int, alpha: float = 0.05) -> dict:
        with db.session() as s:
            b, c = _get_run(s, baseline), _get_run(s, candidate)
            if b.dataset_id != c.dataset_id:
                raise HTTPException(422, "runs were made on different datasets")
            return compare_runs(list(b.results), list(c.results), alpha).to_dict()

    @app.get("/runs/{run_id}/label-sample")
    def get_label_sample(run_id: int, n: int = 30) -> list[dict]:
        with db.session() as s:
            _get_run(s, run_id)
            return label_sample(s, run_id, max(1, min(n, 200)))

    @app.post("/runs/{run_id}/labels")
    def post_labels(run_id: int, body: LabelsIn) -> dict:
        with db.session() as s:
            _get_run(s, run_id)
            have = set(s.scalars(select(TaskResult.task_id).where(TaskResult.run_id == run_id)))
            unknown = sorted({lb.task_id for lb in body.labels} - have)
            if unknown:
                raise HTTPException(422, f"tasks without a result in this run: {unknown}")
            existing = {
                lb.task_id: lb for lb in s.scalars(select(HumanLabel).where(HumanLabel.run_id == run_id))
            }
            for lb in body.labels:  # upsert: re-labeling corrects an earlier mistake
                if lb.task_id in existing:
                    existing[lb.task_id].passed = lb.passed
                else:
                    s.add(HumanLabel(run_id=run_id, task_id=lb.task_id, passed=lb.passed))
            s.commit()
            return {"labeled": s.query(HumanLabel).filter_by(run_id=run_id).count()}

    @app.get("/runs/{run_id}/calibration")
    def get_calibration(run_id: int, min_n: int = 30) -> dict:
        with db.session() as s:
            _get_run(s, run_id)
            return run_calibration(s, run_id, min_n).to_dict()

    def _get_run(s, run_id: int) -> Run:
        run = s.get(Run, run_id)
        if run is None:
            raise HTTPException(404, f"run {run_id} not found")
        return run

    app.state.db = db
    return app


def app_factory() -> FastAPI:
    """Entry point for `uvicorn agentprobe.api:app_factory --factory`."""
    redis_url = os.environ.get("REDIS_URL")
    return create_app(queue=RedisQueue.from_url(redis_url) if redis_url else None)


__all__ = ["TaskResult", "app_factory", "create_app"]
