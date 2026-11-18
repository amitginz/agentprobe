import json
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agentprobe.agents import AgentResult, MockAgent, build_agent
from agentprobe.api import create_app
from agentprobe.compare import compare_runs, mcnemar_exact, percentile
from agentprobe.db import Database, Dataset, Run, Task, TaskResult
from agentprobe.judges import judge
from agentprobe.runner import TaskSpec, execute_run, run_task

DATASET = json.loads((Path(__file__).parent.parent / "examples" / "dataset.json").read_text())


@pytest.fixture
def db(tmp_path):
    return Database(f"sqlite:///{tmp_path / 't.db'}")


def make_run(db, agent, **kw):
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


# --- judges ---------------------------------------------------------------------------

def test_judges():
    r = AgentResult(output="The answer is Paris.", tool_calls=["search", "calc"])
    v = judge("contains", "paris", {}, r)
    assert (v.passed, v.score) == (True, 1.0)
    assert judge("exact", "paris", {}, r).passed is False
    assert judge("regex", r"p\w+s", {}, r).passed is True
    assert judge("tool_called", "search,calc", {}, r).passed is True
    v = judge("tool_called", "search,web", {}, r)
    assert (v.passed, v.score) == (False, 0.5)
    with pytest.raises(ValueError):
        judge("nope", "", {}, r)


# --- statistics -----------------------------------------------------------------------

def test_mcnemar_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    # 0 vs 8 discordant pairs: 2 * (1/256)
    assert mcnemar_exact(0, 8) == pytest.approx(2 / 256)
    assert mcnemar_exact(8, 0) == mcnemar_exact(0, 8)


def test_percentile():
    assert percentile([], 0.95) == 0.0
    assert percentile(list(range(1, 101)), 0.95) == 95


def _res(task_id, passed):
    return TaskResult(task_id=task_id, passed=passed, latency_ms=1.0, cost_usd=0.01)


def test_compare_detects_regression_and_ignores_noise():
    base = [_res(i, True) for i in range(40)]
    bad = [_res(i, i >= 12) for i in range(40)]  # 12 tasks broke
    assert compare_runs(base, bad).verdict == "regression"
    assert compare_runs(bad, base).verdict == "improvement"
    noisy = [_res(i, i != 0) for i in range(40)]  # a single flip is not significant
    assert compare_runs(base, noisy).verdict == "no_significant_change"


# --- runner ---------------------------------------------------------------------------

class Flaky:
    def __init__(self, fail_times):
        self.calls, self.fail_times, self.lock = 0, fail_times, threading.Lock()

    def run(self, task_input, expected=""):
        with self.lock:
            self.calls += 1
            if self.calls <= self.fail_times:
                raise RuntimeError("boom")
        return AgentResult(output=expected)


def test_retry_recovers_and_final_failure_is_recorded():
    spec = TaskSpec(1, "q", "a", "contains", {})
    ok = run_task(Flaky(1), spec, retries=1)
    assert ok.passed and ok.attempts == 2
    bad = run_task(Flaky(5), spec, retries=1)
    assert not bad.passed and bad.attempts == 2 and "boom" in bad.error


def test_execute_run_persists_results(db):
    run_id = make_run(db, "mock:accuracy=1.0")
    execute_run(db, run_id)
    with db.session() as s:
        run = s.get(Run, run_id)
        assert run.status == "completed"
        assert len(run.results) == len(DATASET["tasks"])
        assert all(r.passed for r in run.results)


def test_budget_stops_run_early(db):
    run_id = make_run(db, "mock:accuracy=1.0", concurrency=1, budget_usd=1e-4)
    execute_run(db, run_id)
    with db.session() as s:
        run = s.get(Run, run_id)
        assert run.status == "budget_exceeded"
        assert 0 < len(run.results) < len(DATASET["tasks"])


def test_mock_agent_is_deterministic_and_accuracy_is_respected():
    a, b = MockAgent(accuracy=0.7, seed=3), MockAgent(accuracy=0.7, seed=3)
    inputs = [f"q{i}" for i in range(2000)]
    outs = [a.run(q, "x").output for q in inputs]
    assert outs == [b.run(q, "x").output for q in inputs]
    assert sum(o == "x" for o in outs) / len(outs) == pytest.approx(0.7, abs=0.04)


def test_build_agent_rejects_unknown_spec():
    with pytest.raises(ValueError):
        build_agent("gpt:whatever")


# --- API end to end -------------------------------------------------------------------

def test_api_end_to_end(tmp_path):
    client = TestClient(create_app(f"sqlite:///{tmp_path / 'api.db'}", run_inline=True))
    assert client.get("/health").json()["status"] == "ok"

    ds = client.post("/datasets", json=DATASET)
    assert ds.status_code == 201
    assert client.post("/datasets", json=DATASET).status_code == 409
    ds_id = ds.json()["id"]

    good = client.post("/runs", json={"dataset_id": ds_id, "agent": "mock:accuracy=1.0", "label": "v1"})
    bad = client.post("/runs", json={"dataset_id": ds_id, "agent": "mock:accuracy=0.4,seed=9", "label": "v2"})
    assert good.status_code == bad.status_code == 202

    run = client.get(f"/runs/{good.json()['id']}").json()
    assert run["status"] == "completed" and run["stats"]["pass_rate"] == 1.0

    cmp = client.get("/compare", params={"baseline": good.json()["id"], "candidate": bad.json()["id"]}).json()
    assert cmp["verdict"] == "regression"
    assert cmp["broken"] > 0 and cmp["p_value"] < 0.05

    assert client.post("/runs", json={"dataset_id": ds_id, "agent": "bogus"}).status_code == 422
    assert client.post("/runs", json={"dataset_id": 999, "agent": "mock:"}).status_code == 404
    assert client.get("/runs/999").status_code == 404
