"""Runs against a real Redis. Skipped unless REDIS_URL is set (CI provides one)."""

import json
import os
import threading
import uuid
from pathlib import Path

import pytest

from agentprobe.db import Database, Dataset, Run, Task
from agentprobe.jobqueue import RedisQueue
from agentprobe.worker import Worker

pytestmark = pytest.mark.skipif(not os.environ.get("REDIS_URL"), reason="REDIS_URL not set")
DATASET = json.loads((Path(__file__).parent.parent / "examples" / "dataset.json").read_text())


def test_competing_workers_run_every_job_exactly_once(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'r.db'}")
    queue = RedisQueue.from_url(os.environ["REDIS_URL"], name=f"test-{uuid.uuid4().hex}", lease_seconds=30)

    with db.session() as s:
        ds = Dataset(name="d", tasks=[Task(**t) for t in DATASET["tasks"]])
        s.add(ds)
        s.commit()
        runs = [Run(dataset_id=ds.id, agent="mock:accuracy=1.0", concurrency=2) for _ in range(6)]
        s.add_all(runs)
        s.commit()
        ids = [r.id for r in runs]
    for run_id in ids:
        queue.enqueue(run_id)

    stop = threading.Event()
    workers = [Worker(db, queue, poll_timeout=0.2) for _ in range(3)]
    threads = [threading.Thread(target=w.run_forever, args=(stop,)) for w in workers]
    [t.start() for t in threads]
    try:
        for _ in range(100):
            with db.session() as s:
                if all(s.get(Run, i).status == "completed" for i in ids):
                    break
            stop.wait(0.1)
    finally:
        stop.set()
        [t.join(timeout=10) for t in threads]

    with db.session() as s:
        for run_id in ids:
            run = s.get(Run, run_id)
            assert run.status == "completed"
            assert len(run.results) == len(DATASET["tasks"])  # no duplicated work
    assert queue.stats() == {"pending": 0, "processing": 0}
