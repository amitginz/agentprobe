"""End to end over the real OTLP/HTTP wire format, with no mocks between AgentProbe and the socket.

A tiny collector (stdlib HTTP server) receives what a separate AgentProbe process exports and
decodes it with the official protobuf definitions.
"""

import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceRequest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

DATASET = Path(__file__).parent.parent / "examples" / "dataset.json"


class Collector(HTTPServer):
    def __init__(self):
        super().__init__(("127.0.0.1", 0), self.Handler)
        self.traces, self.metrics = [], []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["content-length"]))
            if self.path == "/v1/traces":
                msg = ExportTraceServiceRequest()
                msg.ParseFromString(body)
                self.server.traces.append(msg)
            elif self.path == "/v1/metrics":
                msg = ExportMetricsServiceRequest()
                msg.ParseFromString(body)
                self.server.metrics.append(msg)
            self.send_response(200)
            self.send_header("content-type", "application/x-protobuf")
            self.end_headers()

        def log_message(self, *_):
            pass


def test_cli_ci_exports_a_connected_trace_and_metrics_over_otlp(tmp_path):
    collector = Collector()
    threading.Thread(target=collector.serve_forever, daemon=True).start()
    try:
        env = {
            "PATH": "/usr/bin:/bin",
            "OTEL_EXPORTER_OTLP_ENDPOINT": f"http://127.0.0.1:{collector.server_port}",
            "OTEL_SERVICE_NAME": "agentprobe-test",
        }
        proc = subprocess.run(
            [sys.executable, "-m", "agentprobe.cli", "--db", f"sqlite:///{tmp_path / 'w.db'}", "ci",
             "--dataset", str(DATASET), "--baseline-agent", "mock:accuracy=0.9,seed=1",
             "--candidate-agent", "mock:accuracy=0.9,seed=1", "--concurrency", "4"],
            check=False, env=env, capture_output=True, text=True, timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
    finally:
        collector.shutdown()

    spans, resources = [], set()
    for req in collector.traces:
        for rs in req.resource_spans:
            resources |= {a.value.string_value for a in rs.resource.attributes if a.key == "service.name"}
            for ss in rs.scope_spans:
                spans.extend(ss.spans)
    names = [s.name for s in spans]
    assert resources == {"agentprobe-test"}
    assert names.count("agentprobe.ci") == 1 and names.count("agentprobe.run") == 2
    assert names.count("agentprobe.task") == 80 and names.count("agent.call") == 80  # 40 tasks x 2 runs

    # the whole gate is ONE trace, and parent links survive the wire
    assert len({s.trace_id for s in spans}) == 1
    by_id = {s.span_id: s for s in spans}
    ci = next(s for s in spans if s.name == "agentprobe.ci")
    runs = [s for s in spans if s.name == "agentprobe.run"]
    assert all(r.parent_span_id == ci.span_id for r in runs)
    task = next(s for s in spans if s.name == "agentprobe.task")
    assert by_id[task.parent_span_id].name == "agentprobe.run"

    metric_names = {
        m.name
        for req in collector.metrics
        for rm in req.resource_metrics
        for sm in rm.scope_metrics
        for m in sm.metrics
    }
    assert {"agentprobe.tasks", "agentprobe.cost", "agentprobe.agent.latency"} <= metric_names


def test_telemetry_is_silent_and_harmless_when_not_configured(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "agentprobe.cli", "--db", f"sqlite:///{tmp_path / 'q.db'}", "run",
         "--dataset", str(DATASET), "--agent", "mock:accuracy=1.0"],
        check=False, env={"PATH": "/usr/bin:/bin"}, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0 and "40/40 passed" in proc.stdout
    assert "Traceback" not in proc.stderr and "opentelemetry" not in proc.stderr.lower()


def test_metrics_export_can_be_disabled_for_traces_only_backends(tmp_path):
    collector = Collector()
    threading.Thread(target=collector.serve_forever, daemon=True).start()
    try:
        env = {
            "PATH": "/usr/bin:/bin",
            "OTEL_EXPORTER_OTLP_ENDPOINT": f"http://127.0.0.1:{collector.server_port}",
            "OTEL_METRICS_EXPORTER": "none",
        }
        proc = subprocess.run(
            [sys.executable, "-m", "agentprobe.cli", "--db", f"sqlite:///{tmp_path / 'm.db'}", "run",
             "--dataset", str(DATASET), "--agent", "mock:accuracy=1.0"],
            check=False, env=env, capture_output=True, text=True, timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
    finally:
        collector.shutdown()
    assert collector.traces and not collector.metrics  # spans still flow; no metrics request was sent
