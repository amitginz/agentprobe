import json
import threading
from pathlib import Path

import fakeredis
import pytest
from fastapi.testclient import TestClient

from agentprobe.agents import AgentResult
from agentprobe.api import create_app
from agentprobe.db import Database, Dataset, Run, Task, TaskResult
from agentprobe.jobqueue import RedisQueue
from agentprobe.runner import claim_run, execute_run
from agentprobe.worker import Worker

DATASET = json.loads((Path(__file__).parent.parent / "examples" / "dataset.json").read_text())


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def queue(clock):
    client = fakeredis.FakeRedis(decode_responses=True)
    return RedisQueue(client, lease_seconds=30, clock=clock)


@pytest.fixture
def db(tmp_path):
    return Database(f"sqlite:///{tmp_path / 'q.db'}")


def new_run(db, agent="mock:accuracy=1.0", **kw):
    with db.session() as s:
        ds = s.query(Dataset).first()
        if ds is None:
            ds = Dataset(name="d", tasks=[Task(**t) for t in DATASET["tasks"]])
            s.add(ds)
            s.commit()
        run = Run(dataset_id=ds.id, agent=agent, **kw)
        s.add(run)
        s.commit()
        return run.id


def status(db, run_id):
    with db.session() as s:
        return s.get(Run, run_id).status


# --- queue semantics ------------------------------------------------------------------

def test_fifo_and_ack(queue):
    for i in (1, 2, 3):
        queue.enqueue(i)
    assert [queue.dequeue(0.01)[0] for _ in range(3)] == [1, 2, 3]
    assert queue.stats() == {"pending": 0, "processing": 3}
    for i in (1, 2, 3):
        queue.ack(i)
    assert queue.stats() == {"pending": 0, "processing": 0}
    assert queue.dequeue(0.01) is None


def test_live_lease_is_not_reaped_but_expired_one_is(queue, clock):
    queue.enqueue(7)
    queue.dequeue(0.01)
    clock.now += 29
    assert queue.reap() == []  # still inside the lease
    queue.heartbeat(7)
    clock.now += 29
    assert queue.reap() == []  # heartbeat pushed the deadline out
    clock.now += 31
    assert queue.reap() == [7]
    assert queue.stats() == {"pending": 1, "processing": 0}
    assert queue.dequeue(0.01) == (7, 2)  # redelivery is counted


def test_missing_lease_is_adopted_not_requeued(queue, clock):
    """A job taken but not yet leased (worker mid-dequeue) must not be handed out twice."""
    queue.enqueue(5)
    queue.r.blmove(queue.pending, queue.processing, 0.01, "RIGHT", "LEFT")  # no lease written
    assert queue.reap() == []
    assert queue.stats()["processing"] == 1
    clock.now += 31
    assert queue.reap() == [5]


# --- claim and resume -----------------------------------------------------------------

def test_claim_is_exclusive(db):
    run_id = new_run(db)
    results = []
    threads = [threading.Thread(target=lambda: results.append(claim_run(db, run_id))) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results.count(True) == 1


def test_duplicate_delivery_executes_once(db):
    run_id = new_run(db)
    assert execute_run(db, run_id) is True
    assert execute_run(db, run_id) is False
    with db.session() as s:
        assert len(s.get(Run, run_id).results) == len(DATASET["tasks"])


class Crashy:
    """Fails (non-retryably) on a given task so we can simulate a run dying midway."""

    def __init__(self, die_after):
        self.calls, self.die_after = 0, die_after

    def run(self, task_input, expected=""):
        self.calls += 1
        if self.calls > self.die_after:
            raise SystemExit("worker killed")  # BaseException: not caught by run_task
        return AgentResult(output=expected, cost_usd=0.001)


def test_redelivered_run_resumes_without_redoing_finished_tasks(db):
    run_id = new_run(db, concurrency=1)
    with pytest.raises(SystemExit):
        execute_run(db, run_id, agent=Crashy(die_after=10))
    with db.session() as s:
        assert len(s.get(Run, run_id).results) == 10
    assert status(db, run_id) == "running"  # died mid-run, still marked running

    from agentprobe.runner import release_run

    release_run(db, run_id)  # what the reaper path does for an expired lease
    fresh = Crashy(die_after=10**6)
    assert execute_run(db, run_id, agent=fresh) is True
    assert fresh.calls == len(DATASET["tasks"]) - 10  # only the remaining tasks ran
    with db.session() as s:
        run = s.get(Run, run_id)
        assert run.status == "completed" and len(run.results) == len(DATASET["tasks"])
        ids = [r.task_id for r in run.results]
        assert len(ids) == len(set(ids))  # no duplicates


# --- worker ---------------------------------------------------------------------------

def test_worker_executes_enqueued_run(db, queue):
    run_id = new_run(db)
    queue.enqueue(run_id)
    worker = Worker(db, queue, poll_timeout=0.05)
    assert worker.run_once() is True
    assert worker.run_once() is False
    assert status(db, run_id) == "completed"
    assert queue.stats() == {"pending": 0, "processing": 0}


def test_dead_worker_job_is_recovered_by_another_worker(db, queue, clock):
    run_id = new_run(db)
    queue.enqueue(run_id)
    queue.dequeue(0.01)  # worker A takes the job...
    with db.session() as s:  # ...starts it, then the process dies
        s.get(Run, run_id).status = "running"
        s.commit()
    clock.now += 31

    worker_b = Worker(db, queue, poll_timeout=0.05)
    assert worker_b.run_once() is True  # reaps, releases, re-dequeues, executes
    assert status(db, run_id) == "completed"
    assert queue.stats() == {"pending": 0, "processing": 0}


def test_poison_job_is_failed_after_max_deliveries(db, queue, clock):
    run_id = new_run(db)
    queue.enqueue(run_id)
    worker = Worker(db, queue, poll_timeout=0.05, max_deliveries=2)
    for _ in range(2):  # two workers take it and die without acking
        queue.dequeue(0.01)
        clock.now += 31
        queue.reap()
    assert worker.run_once() is True  # third delivery exceeds the limit
    assert status(db, run_id) == "failed"
    with db.session() as s:
        assert "abandoned" in s.get(Run, run_id).error
    assert queue.stats() == {"pending": 0, "processing": 0}


# --- API with queue -------------------------------------------------------------------

def test_api_enqueues_and_worker_finishes(tmp_path, queue):
    url = f"sqlite:///{tmp_path / 'api.db'}"
    client = TestClient(create_app(url, queue=queue))
    ds_id = client.post("/datasets", json=DATASET).json()["id"]

    resp = client.post("/runs", json={"dataset_id": ds_id, "agent": "mock:accuracy=1.0"})
    assert resp.status_code == 202
    run_id = resp.json()["id"]
    assert client.get(f"/runs/{run_id}").json()["status"] == "pending"  # not run by the API
    health = client.get("/health").json()
    assert health["execution"] == "queue" and health["queue"]["pending"] == 1

    Worker(Database(url), queue, poll_timeout=0.05).run_once()
    assert client.get(f"/runs/{run_id}").json()["status"] == "completed"


def test_api_returns_503_and_fails_run_when_queue_is_down(tmp_path):
    class Down:
        def enqueue(self, _):
            raise ConnectionError("redis down")

    url = f"sqlite:///{tmp_path / 'api.db'}"
    client = TestClient(create_app(url, queue=Down()))
    ds_id = client.post("/datasets", json=DATASET).json()["id"]
    resp = client.post("/runs", json={"dataset_id": ds_id, "agent": "mock:accuracy=1.0"})
    assert resp.status_code == 503
    assert client.get("/runs/1").json()["status"] == "failed"


def test_budget_carries_over_resumed_run(db):
    run_id = new_run(db, concurrency=1, budget_usd=0.005)
    with pytest.raises(SystemExit):
        execute_run(db, run_id, agent=Crashy(die_after=4))  # spends 4 x 0.001
    from agentprobe.runner import release_run

    release_run(db, run_id)
    execute_run(db, run_id, agent=Crashy(die_after=10**6))
    with db.session() as s:
        run = s.get(Run, run_id)
        total = sum(r.cost_usd for r in s.query(TaskResult).filter_by(run_id=run_id))
        assert run.status == "budget_exceeded"
        assert total < 0.005 + 0.002  # stopped soon after crossing the shared budget
