"""REST API."""

from __future__ import annotations

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select

from . import __version__
from .agents import build_agent
from .compare import compare_runs, run_stats
from .db import Database, Dataset, Run, Task, TaskResult
from .judges import JUDGES
from .runner import execute_run


class TaskIn(BaseModel):
    input: str
    expected: str = ""
    judge: str = "contains"
    judge_config: dict = Field(default_factory=dict)


class DatasetIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    tasks: list[TaskIn] = Field(min_length=1)


class RunIn(BaseModel):
    dataset_id: int
    agent: str
    label: str = ""
    concurrency: int = Field(default=4, ge=1, le=64)
    retries: int = Field(default=1, ge=0, le=5)
    budget_usd: float | None = Field(default=None, gt=0)


def create_app(database_url: str | None = None, run_inline: bool = False) -> FastAPI:
    """App factory. `run_inline` executes runs synchronously (used by tests)."""
    db = Database(database_url)
    app = FastAPI(title="AgentProbe", version=__version__)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.post("/datasets", status_code=201)
    def create_dataset(body: DatasetIn) -> dict:
        for t in body.tasks:
            if t.judge not in JUDGES:
                raise HTTPException(422, f"unknown judge {t.judge!r}; available: {sorted(JUDGES)}")
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

    def _get_run(s, run_id: int) -> Run:
        run = s.get(Run, run_id)
        if run is None:
            raise HTTPException(404, f"run {run_id} not found")
        return run

    app.state.db = db
    return app


def app_factory() -> FastAPI:
    """Entry point for `uvicorn agentprobe.api:app_factory --factory`."""
    return create_app()


__all__ = ["TaskResult", "app_factory", "create_app"]
