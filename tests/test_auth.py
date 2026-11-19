import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agentprobe.api import create_app

DATASET = json.loads((Path(__file__).parent.parent / "examples" / "dataset.json").read_text())
ADMIN, VIEWER = "admin-secret", "viewer-secret"


def make(tmp_path, **kw):
    return TestClient(create_app(f"sqlite:///{tmp_path / 'a.db'}", run_inline=True, **kw))


def bearer(key):
    return {"Authorization": f"Bearer {key}"}


def test_open_by_default_and_warns(tmp_path, caplog, monkeypatch):
    monkeypatch.delenv("AGENTPROBE_API_KEYS", raising=False)
    monkeypatch.delenv("AGENTPROBE_READ_API_KEYS", raising=False)
    with caplog.at_level(logging.WARNING, logger="agentprobe.api"):
        client = make(tmp_path)
    assert client.get("/datasets").status_code == 200
    assert "authentication is disabled" in caplog.text
    assert client.get("/health").json()["auth"] is False


def test_require_auth_refuses_to_start_without_keys(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENTPROBE_API_KEYS", raising=False)
    monkeypatch.delenv("AGENTPROBE_READ_API_KEYS", raising=False)
    monkeypatch.setenv("AGENTPROBE_REQUIRE_AUTH", "1")
    with pytest.raises(RuntimeError, match="REQUIRE_AUTH"):
        make(tmp_path)


def test_data_routes_need_a_key_but_health_and_dashboard_do_not(tmp_path):
    client = make(tmp_path, api_keys=[ADMIN])
    assert client.get("/health").json()["auth"] is True
    assert client.get("/").status_code == 200  # static page, no data in it
    for method, path in [("get", "/datasets"), ("get", "/runs"), ("get", "/runs/1"), ("get", "/compare?baseline=1&candidate=2")]:
        r = getattr(client, method)(path)
        assert r.status_code == 401, path
        assert r.headers["www-authenticate"] == "Bearer"
    assert client.post("/datasets", json=DATASET).status_code == 401
    assert client.post("/runs", json={"dataset_id": 1, "agent": "mock:accuracy=1.0"}).status_code == 401


def test_every_data_route_is_protected(tmp_path):
    """Guards against a future route being added with @app instead of the authenticated router."""
    client = make(tmp_path, api_keys=[ADMIN])
    public = {"/health", "/", "/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
    for route in client.app.routes:
        path = getattr(route, "path", None)
        if path in public or path is None:
            continue
        concrete = path.replace("{run_id}", "1")
        for method in route.methods - {"HEAD", "OPTIONS"}:
            r = client.request(method, concrete)
            assert r.status_code == 401, f"{method} {path} is reachable without a key"


def test_valid_keys_via_bearer_or_x_api_key(tmp_path):
    client = make(tmp_path, api_keys=[ADMIN])
    assert client.get("/datasets", headers=bearer(ADMIN)).status_code == 200
    assert client.get("/datasets", headers={"X-API-Key": ADMIN}).status_code == 200
    assert client.get("/datasets", headers=bearer("wrong")).status_code == 401
    assert client.get("/datasets", headers=bearer(ADMIN + "x")).status_code == 401
    assert client.get("/datasets", headers={"Authorization": f"Basic {ADMIN}"}).status_code == 401


def test_read_only_key_can_read_but_not_write(tmp_path):
    client = make(tmp_path, api_keys=[ADMIN], read_keys=[VIEWER])
    ds = client.post("/datasets", json=DATASET, headers=bearer(ADMIN)).json()["id"]
    client.post("/runs", json={"dataset_id": ds, "agent": "mock:accuracy=1.0"}, headers=bearer(ADMIN))
    assert client.get("/runs", headers=bearer(VIEWER)).status_code == 200
    assert client.get("/runs/1/results", headers=bearer(VIEWER)).status_code == 200
    denied = client.post("/runs", json={"dataset_id": ds, "agent": "mock:accuracy=1.0"}, headers=bearer(VIEWER))
    assert denied.status_code == 403
    assert client.post("/datasets", json=DATASET, headers=bearer(VIEWER)).status_code == 403
    assert len(client.get("/runs", headers=bearer(ADMIN)).json()) == 1  # the denied run was not created


def test_keys_come_from_the_environment_and_rotate(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTPROBE_API_KEYS", " old-key , new-key ")
    monkeypatch.setenv("AGENTPROBE_READ_API_KEYS", VIEWER)
    client = make(tmp_path)
    assert client.get("/datasets", headers=bearer("old-key")).status_code == 200
    assert client.get("/datasets", headers=bearer("new-key")).status_code == 200
    assert client.get("/datasets", headers=bearer(VIEWER)).status_code == 200
    assert client.get("/datasets", headers=bearer("")).status_code == 401


def test_rejected_key_is_not_echoed(tmp_path):
    client = make(tmp_path, api_keys=[ADMIN])
    r = client.get("/datasets", headers=bearer("super-secret-guess"))
    assert "super-secret-guess" not in r.text and ADMIN not in r.text


def test_dashboard_sends_the_key_only_from_session_storage():
    html = (Path(__file__).parent.parent / "src" / "agentprobe" / "dashboard" / "index.html").read_text()
    assert "sessionStorage" in html and "localStorage" not in html
    assert "Authorization" in html
