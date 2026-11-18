import json
import re
from pathlib import Path

from fastapi.testclient import TestClient

from agentprobe.api import create_app

DATASET = json.loads((Path(__file__).parent.parent / "examples" / "dataset.json").read_text())


def make(tmp_path):
    client = TestClient(create_app(f"sqlite:///{tmp_path / 'd.db'}", run_inline=True))
    ds_id = client.post("/datasets", json=DATASET).json()["id"]
    return client, ds_id


def test_dashboard_is_served_and_escapes_untrusted_text(tmp_path):
    client, _ = make(tmp_path)
    resp = client.get("/")
    assert resp.status_code == 200 and "text/html" in resp.headers["content-type"]
    html = resp.text
    assert "<title>AgentProbe</title>" in html
    # no external resources: the page must work offline and leak nothing to third parties
    assert "http://" not in html.replace("http://www.w3.org", "") and "https://" not in html
    # agent output and dataset text only ever reach the DOM through the escaper
    assert "const esc =" in html
    # every template interpolation that touches a user-controlled field goes through esc()
    risky = re.compile(r"\b(output|input|expected|label|agent|error|reasoning)\b")
    for expr in re.findall(r"\$\{([^}]*)\}", html):
        if risky.search(expr) and "esc(" not in expr and "?" not in expr:
            raise AssertionError(f"unescaped interpolation in dashboard: ${{{expr}}}")


def test_run_list_is_oldest_first_and_filterable(tmp_path):
    client, ds_id = make(tmp_path)
    other = client.post("/datasets", json={"name": "other", "tasks": [{"input": "q", "expected": "a"}]}).json()["id"]
    for label in ("a", "b", "c"):
        client.post("/runs", json={"dataset_id": ds_id, "agent": "mock:accuracy=1.0", "label": label})
    client.post("/runs", json={"dataset_id": other, "agent": "mock:accuracy=1.0"})

    runs = client.get("/runs", params={"dataset_id": ds_id}).json()
    assert [r["label"] for r in runs] == ["a", "b", "c"]
    assert runs[0]["tasks_total"] == len(DATASET["tasks"]) and runs[0]["stats"]["pass_rate"] == 1.0
    assert len(client.get("/runs").json()) == 4
    assert [r["label"] for r in client.get("/runs", params={"dataset_id": ds_id, "limit": 2}).json()] == ["b", "c"]


def test_results_include_task_input_and_compare_details(tmp_path):
    client, ds_id = make(tmp_path)
    a = client.post("/runs", json={"dataset_id": ds_id, "agent": "mock:accuracy=1.0"}).json()["id"]
    b = client.post("/runs", json={"dataset_id": ds_id, "agent": "mock:accuracy=0.5,seed=2"}).json()["id"]

    first = client.get(f"/runs/{a}/results").json()[0]
    assert first["input"] and "expected" in first and first["judge"] in ("contains", "tool_called", "regex")

    plain = client.get("/compare", params={"baseline": a, "candidate": b}).json()
    assert "broken_tasks" not in plain
    full = client.get("/compare", params={"baseline": a, "candidate": b, "details": True}).json()
    assert len(full["broken_tasks"]) == full["broken"] > 0
    assert full["fixed_tasks"] == []
    row = full["broken_tasks"][0]
    assert row["baseline_output"] != row["candidate_output"] and row["input"]
