"""Initial schema (everything that existed before migrations were introduced).

Revision ID: 0001
Revises:
"""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "datasets",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(200), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "tasks",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("dataset_id", sa.Integer, sa.ForeignKey("datasets.id"), nullable=False),
        sa.Column("input", sa.Text, nullable=False),
        sa.Column("expected", sa.Text, nullable=False),
        sa.Column("judge", sa.String(50), nullable=False),
        sa.Column("judge_config", sa.JSON, nullable=False),
    )
    op.create_index("ix_tasks_dataset_id", "tasks", ["dataset_id"])
    op.create_table(
        "runs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("dataset_id", sa.Integer, sa.ForeignKey("datasets.id"), nullable=False),
        sa.Column("agent", sa.String(500), nullable=False),
        sa.Column("label", sa.String(200), nullable=False),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("concurrency", sa.Integer, nullable=False),
        sa.Column("retries", sa.Integer, nullable=False),
        sa.Column("budget_usd", sa.Float, nullable=True),
        sa.Column("error", sa.Text, nullable=False),
        sa.Column("trace_context", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_runs_dataset_id", "runs", ["dataset_id"])
    op.create_index("ix_runs_status", "runs", ["status"])
    op.create_table(
        "task_results",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("run_id", sa.Integer, sa.ForeignKey("runs.id"), nullable=False),
        sa.Column("task_id", sa.Integer, sa.ForeignKey("tasks.id"), nullable=False),
        sa.Column("passed", sa.Boolean, nullable=False),
        sa.Column("score", sa.Float, nullable=False),
        sa.Column("output", sa.Text, nullable=False),
        sa.Column("tool_calls", sa.JSON, nullable=False),
        sa.Column("latency_ms", sa.Float, nullable=False),
        sa.Column("input_tokens", sa.Integer, nullable=False),
        sa.Column("output_tokens", sa.Integer, nullable=False),
        sa.Column("cost_usd", sa.Float, nullable=False),
        sa.Column("attempts", sa.Integer, nullable=False),
        sa.Column("error", sa.Text, nullable=False),
        sa.Column("judge_reasoning", sa.Text, nullable=False),
        sa.Column("judge_cost_usd", sa.Float, nullable=False),
    )
    op.create_index("ix_task_results_run_id", "task_results", ["run_id"])
    op.create_index("ix_task_results_task_id", "task_results", ["task_id"])
    op.create_table(
        "human_labels",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("run_id", sa.Integer, sa.ForeignKey("runs.id"), nullable=False),
        sa.Column("task_id", sa.Integer, sa.ForeignKey("tasks.id"), nullable=False),
        sa.Column("passed", sa.Boolean, nullable=False),
        sa.UniqueConstraint("run_id", "task_id"),
    )
    op.create_index("ix_human_labels_run_id", "human_labels", ["run_id"])


def downgrade() -> None:
    for table in ("human_labels", "task_results", "runs", "tasks", "datasets"):
        op.drop_table(table)
