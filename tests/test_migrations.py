import pytest
from alembic import command
from sqlalchemy import create_engine, inspect, text

from agentprobe import migrate
from agentprobe.cli import main
from agentprobe.db import Base, Database, Dataset, Task


def url(tmp_path, name="m.db"):
    return f"sqlite:///{tmp_path / name}"


def test_new_database_is_created_by_migrations_and_matches_the_models(tmp_path):
    db = Database(url(tmp_path))
    assert migrate.current_revision(db.engine) == migrate.head_revision()
    # the hand-written baseline must not drift from the ORM models
    assert migrate.schema_diff(db.engine) == []


def test_opening_twice_is_idempotent_and_keeps_data(tmp_path):
    u = url(tmp_path)
    with Database(u).session() as s:
        s.add(Dataset(name="d", tasks=[Task(input="q", expected="a")]))
        s.commit()
    db = Database(u)
    with db.session() as s:
        assert s.query(Dataset).count() == 1
    assert migrate.current_revision(db.engine) == migrate.head_revision()


def test_database_from_before_migrations_is_adopted_not_recreated(tmp_path):
    u = url(tmp_path)
    engine = create_engine(u)
    Base.metadata.create_all(engine)  # how every database was created before Alembic
    with engine.begin() as c:
        c.execute(text("INSERT INTO datasets (name, created_at) VALUES ('old', '2026-01-01 00:00:00')"))
    engine.dispose()

    db = Database(u)
    assert migrate.current_revision(db.engine) == migrate.head_revision()
    with db.session() as s:
        assert [d.name for d in s.query(Dataset)] == ["old"]


def test_stale_pre_migration_schema_is_refused_not_stamped(tmp_path):
    u = url(tmp_path)
    engine = create_engine(u)
    Base.metadata.create_all(engine)
    with engine.begin() as c:
        c.execute(text("ALTER TABLE runs DROP COLUMN trace_context"))
    engine.dispose()
    with pytest.raises(migrate.SchemaError, match="predates migrations"):
        Database(u)


def test_downgrade_to_base_removes_everything(tmp_path):
    db = Database(url(tmp_path))
    with db.engine.connect() as conn:
        command.downgrade(migrate._config(conn), "base")
        conn.commit()
    assert set(inspect(db.engine).get_table_names()) <= {"alembic_version"}


def test_cli_db_commands(tmp_path, capsys):
    u = url(tmp_path)
    assert main(["--db", u, "db", "check"]) == 1  # nothing created yet
    assert main(["--db", u, "db", "upgrade"]) == 0
    assert main(["--db", u, "db", "check"]) == 0
    out = capsys.readouterr().out
    assert f"revision {migrate.head_revision()}" in out
