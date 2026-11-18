"""Database setup and ORM models.

SQLite is the zero-config default; set DATABASE_URL to a PostgreSQL URL
(e.g. postgresql+psycopg://user:pw@host/db) for multi-instance deployments.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    tasks: Mapped[list[Task]] = relationship(back_populates="dataset", cascade="all, delete")


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), index=True)
    input: Mapped[str] = mapped_column(Text)
    expected: Mapped[str] = mapped_column(Text, default="")
    judge: Mapped[str] = mapped_column(String(50), default="contains")
    judge_config: Mapped[dict] = mapped_column(JSON, default=dict)
    dataset: Mapped[Dataset] = relationship(back_populates="tasks")


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), index=True)
    agent: Mapped[str] = mapped_column(String(500))
    label: Mapped[str] = mapped_column(String(200), default="")
    status: Mapped[str] = mapped_column(String(30), default="pending", index=True)
    concurrency: Mapped[int] = mapped_column(Integer, default=4)
    retries: Mapped[int] = mapped_column(Integer, default=1)
    budget_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str] = mapped_column(Text, default="")
    trace_context: Mapped[str] = mapped_column(Text, default="")  # W3C carrier from whoever created the run
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    results: Mapped[list[TaskResult]] = relationship(back_populates="run", cascade="all, delete")


class TaskResult(Base):
    __tablename__ = "task_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), index=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id"), index=True)
    passed: Mapped[bool] = mapped_column(Boolean)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    output: Mapped[str] = mapped_column(Text, default="")
    tool_calls: Mapped[list] = mapped_column(JSON, default=list)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    attempts: Mapped[int] = mapped_column(Integer, default=1)
    error: Mapped[str] = mapped_column(Text, default="")
    judge_reasoning: Mapped[str] = mapped_column(Text, default="")
    judge_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    run: Mapped[Run] = relationship(back_populates="results")


class HumanLabel(Base):
    """A person's pass/fail verdict on one task of one run, used to calibrate LLM judges."""

    __tablename__ = "human_labels"
    __table_args__ = (UniqueConstraint("run_id", "task_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), index=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id"))
    passed: Mapped[bool] = mapped_column(Boolean)


class Database:
    def __init__(self, url: str | None = None) -> None:
        self.url = url or os.environ.get("DATABASE_URL", "sqlite:///agentprobe.db")
        kwargs: dict = {}
        if self.url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
        self.engine = create_engine(self.url, **kwargs)
        if self.url.startswith("sqlite"):
            event.listen(self.engine, "connect", _sqlite_pragmas)
        self._factory = sessionmaker(self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator:
        s = self._factory()
        try:
            yield s
        finally:
            s.close()


def _sqlite_pragmas(dbapi_conn, _record) -> None:
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()
