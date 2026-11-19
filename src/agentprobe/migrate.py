"""Schema migrations (Alembic), run automatically when a `Database` is opened.

* A new database is created by running every migration, so the migrations are the one source of
  truth for the schema.
* A database created before migrations existed (tables, no `alembic_version`) is adopted by
  stamping it at the baseline, but only if its schema really matches; otherwise we refuse rather
  than silently stamp a database that is missing columns.
* On PostgreSQL an advisory lock serialises concurrent starters (the API and every worker open the
  database at boot), so only one of them migrates and the rest wait, then see it is current.
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Engine, inspect, text

from .db import Base

BASELINE = "0001"
_LOCK_KEY = 0x61676E7470  # arbitrary constant shared by all agentprobe processes


class SchemaError(RuntimeError):
    pass


def _config(connection) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(Path(__file__).parent / "migrations"))
    cfg.attributes["connection"] = connection
    return cfg


def head_revision() -> str:
    cfg = Config()
    cfg.set_main_option("script_location", str(Path(__file__).parent / "migrations"))
    return ScriptDirectory.from_config(cfg).get_current_head()


def current_revision(engine: Engine) -> str | None:
    with engine.connect() as conn:
        return MigrationContext.configure(conn).get_current_revision()


def schema_diff(engine: Engine) -> list:
    """Differences between the live schema and the ORM models (empty when they agree)."""
    with engine.connect() as conn:
        return compare_metadata(MigrationContext.configure(conn, opts={"compare_type": True}), Base.metadata)


def upgrade(engine: Engine, revision: str = "head") -> None:
    with engine.connect() as conn:
        if engine.dialect.name == "postgresql":
            conn.execute(text("SELECT pg_advisory_lock(:k)"), {"k": _LOCK_KEY})
        try:
            ctx = MigrationContext.configure(conn)
            tables = set(inspect(conn).get_table_names())
            if ctx.get_current_revision() is None and "datasets" in tables:
                if schema_diff(engine):
                    raise SchemaError(
                        "this database predates migrations and does not match the current schema; "
                        "export what you need and start a fresh database"
                    )
                command.stamp(_config(conn), BASELINE)
                conn.commit()
            command.upgrade(_config(conn), revision)
            conn.commit()
        finally:
            if engine.dialect.name == "postgresql":
                conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": _LOCK_KEY})
                conn.commit()
