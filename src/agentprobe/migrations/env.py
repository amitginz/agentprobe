"""Alembic environment. The connection is handed in by `agentprobe.migrate`, never read from ini."""

from alembic import context

from agentprobe.db import Base

connection = context.config.attributes["connection"]
context.configure(
    connection=connection,
    target_metadata=Base.metadata,
    render_as_batch=connection.dialect.name == "sqlite",  # SQLite cannot ALTER most things in place
    compare_type=True,
)
with context.begin_transaction():
    context.run_migrations()
