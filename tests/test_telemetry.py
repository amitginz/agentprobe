import json
import threading
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from agentprobe.agents import AgentResult, HTTPAgent
from agentprobe.api import create_app
from agentprobe.db import Database, Dataset, Run, Task
from agentprobe.runner import execute_run
from agentprobe.telemetry import agent_kind, capture_context, restore_context

DATASET = json.loads((Path(__file__).parent.parent / "examples" / "dataset.json").read_text())


@pytest.fixture
def db(tmp_path):
    return Database(f"sqlite:///{tmp_path / 't.db'}")


def make_run(db, agent="mock:accuracy=1.0", n=6, **kw):
    with db.session() as s:
        ds = Dataset(name=f"d{id(kw)}{n}", tasks=[Task(**t) for t in DATASET["tasks"][:n]])
        s.add(ds)
        s.commit()
        run = Run(dataset_id=ds.id, agent=agent, **kw)
        s.add(run)
        s.commit()
        return run.id


def by_name(spans):
    out: dict[str, list] = {}
    for sp in spans.get_finished_spans():
        out.setdefault(sp.name, []).append(sp)
    return out


def test_span_tree_for_a_run(db, spans):
    execute_run(db, make_run(db, n=6, concurrency=3))
    named = by_name(spans)
    assert len(named["agentprobe.run"]) == 1
    assert len(named["agentprobe.task"]) == len(named["agent.call"]) == len(named["judge.evaluate"]) == 6

    run = named["agentprobe.run"][0]
    ids = {t.context.span_id for t in named["agentprobe.task"]}
    # tasks execute on pool threads, yet every one is parented to the run span in the same trace
    assert all(t.parent.span_id == run.context.span_id for t in named["agentprobe.task"])
    assert all(t.context.trace_id == run.context.trace_id for t in named["agentprobe.task"])
    assert all(c.parent.span_id in ids for c in named["agent.call"] + named["judge.evaluate"])

    attrs = run.attributes
    assert attrs["agentprobe.run.status"] == "completed" and attrs["agentprobe.tasks_total"] == 6
    assert attrs["agentprobe.pass_rate"] == 1.0 and attrs["agentprobe.cost_usd"] > 0
    call = named["agent.call"][0].attributes
    assert call["gen_ai.usage.input_tokens"] > 0 and call["agentprobe.tool_calls"] == 1


def test_no_task_content_or_agent_spec_leaks_into_spans(db, spans):
    secret = "http:http://127.0.0.1:1/agent?token=hunter2"
    with db.session() as s:
        ds = Dataset(name="p", tasks=[Task(input="my SSN is 123-45-6789", expected="x", judge="contains")])
        s.add(ds)
        s.commit()
        run = Run(dataset_id=ds.id, agent=secret, retries=0)
        s.add(run)
        s.commit()
        run_id = run.id
    execute_run(db, run_id)
    blob = json.dumps([dict(sp.attributes) | {"name": sp.name} for sp in spans.get_finished_spans()], default=str)
    assert "hunter2" not in blob and "123-45-6789" not in blob and "127.0.0.1" not in blob
    assert agent_kind(secret) == "http"


class FlakyOnce:
    def __init__(self):
        self.calls, self.lock = 0, threading.Lock()

    def run(self, task_input, expected=""):
        with self.lock:
            self.calls += 1
            first = self.calls == 1
        if first:
            raise RuntimeError("upstream 503")
        return AgentResult(output=expected)


def test_failed_attempt_is_recorded_on_its_span_and_retry_succeeds(db, spans):
    execute_run(db, make_run(db, n=1, retries=1), agent=FlakyOnce())
    calls = sorted(by_name(spans)["agent.call"], key=lambda s: s.attributes["agentprobe.attempt"])
    assert [c.status.status_code for c in calls] == [StatusCode.ERROR, StatusCode.UNSET]
    assert any(e.name == "exception" and "upstream 503" in str(e.attributes) for e in calls[0].events)
    task = by_name(spans)["agentprobe.task"][0]
    assert task.status.status_code == StatusCode.UNSET and task.attributes["agentprobe.attempts"] == 2


def test_exhausted_retries_mark_the_task_span_as_error(db, spans):
    class Dead:
        def run(self, *_):
            raise ConnectionError("refused")

    execute_run(db, make_run(db, n=1, retries=1), agent=Dead())
    task = by_name(spans)["agentprobe.task"][0]
    assert task.status.status_code == StatusCode.ERROR and "refused" in task.status.description


def test_a_run_continues_the_trace_of_whoever_created_it(db, spans):
    """API request -> (queue) -> worker: the run span must join the creator's trace."""
    tracer = trace.get_tracer("test")
    with tracer.start_as_current_span("api request") as parent:
        carrier = capture_context()
    run_id = make_run(db, n=2, trace_context=carrier)
    spans.clear()
    execute_run(db, run_id)  # outside any active span, like a worker process
    run = by_name(spans)["agentprobe.run"][0]
    assert run.context.trace_id == parent.context.trace_id
    assert run.parent.span_id == parent.context.span_id


def test_corrupt_trace_context_never_blocks_execution(db, spans):
    assert restore_context("{not json") is None and restore_context("") is None
    run_id = make_run(db, n=1, trace_context="{not json")
    execute_run(db, run_id)
    with db.session() as s:
        assert s.get(Run, run_id).status == "completed"


def test_api_stores_trace_context_and_links_worker_spans(tmp_path, spans):
    url = f"sqlite:///{tmp_path / 'a.db'}"
    client = TestClient(create_app(url, run_inline=True))
    ds_id = client.post("/datasets", json={"name": "d", "tasks": DATASET["tasks"][:3]}).json()["id"]
    run_id = client.post("/runs", json={"dataset_id": ds_id, "agent": "mock:accuracy=1.0"}).json()["id"]
    named = by_name(spans)
    create, run = named["agentprobe.create_run"][0], named["agentprobe.run"][0]
    assert run.context.trace_id == create.context.trace_id and run.parent.span_id == create.context.span_id
    with Database(url).session() as s:
        assert json.loads(s.get(Run, run_id).trace_context)["traceparent"].split("-")[1] == f"{create.context.trace_id:032x}"


def test_http_agent_propagates_traceparent(monkeypatch):
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured.update(headers)
        return httpx.Response(200, json={"output": "ok"}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    with trace.get_tracer("t").start_as_current_span("caller") as sp:
        HTTPAgent("http://agent.example/x").run("q")
    assert captured["traceparent"].split("-")[1] == f"{sp.context.trace_id:032x}"


def _totals(reader):
    """Cumulative metric totals, since counters in the shared session provider never reset."""
    data = reader.get_metrics_data()
    found = {m.name: m for rm in data.resource_metrics for sm in rm.scope_metrics for m in sm.metrics}
    tasks = found["agentprobe.tasks"].data.data_points
    return {
        "tasks": sum(p.value for p in tasks),
        "failed_tasks": sum(p.value for p in tasks if not p.attributes["passed"]),
        "cost": sum(p.value for p in found["agentprobe.cost"].data.data_points),
        "latency_count": sum(p.count for p in found["agentprobe.agent.latency"].data.data_points),
        "latency_sum": sum(p.sum for p in found["agentprobe.agent.latency"].data.data_points),
    }


def test_metrics_count_tasks_cost_and_latency(db, metric_reader):
    execute_run(db, make_run(db, n=1))  # make sure every instrument exists before the baseline read
    before = _totals(metric_reader)
    execute_run(db, make_run(db, n=5, agent="mock:accuracy=1.0,latency=2"))
    after = _totals(metric_reader)
    assert after["tasks"] - before["tasks"] == 5
    assert after["failed_tasks"] == before["failed_tasks"]  # accuracy 1.0: nothing failed
    assert after["cost"] > before["cost"]
    assert after["latency_count"] - before["latency_count"] == 5
    assert after["latency_sum"] - before["latency_sum"] >= 5  # five calls of roughly 1-3 ms each
