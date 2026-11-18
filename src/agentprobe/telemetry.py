"""OpenTelemetry tracing and metrics.

The API package (`opentelemetry-api`) is a core dependency because it is a no-op until an SDK is
installed and configured, so instrumented code costs nothing when telemetry is off. Exporting needs
the optional extra: ``pip install "agentprobe[otel]"``.

Enable export with the standard environment variables:

    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318   # any OTLP/HTTP collector, e.g. Jaeger
    OTEL_SERVICE_NAME=agentprobe                        # optional
    AGENTPROBE_OTEL_CONSOLE=1                           # also print spans to stdout (debugging)

What a trace looks like::

    agentprobe.create_run            (API request; worker spans link back to it)
    └── agentprobe.run               (one per execution; resumed runs get a new span)
        └── agentprobe.task          (one per task, parented across the thread pool)
            ├── agent.call           (one per attempt; failures carry the exception)
            └── judge.evaluate

Privacy: task inputs, agent outputs and judge reasoning are never put on spans (they may hold
sensitive data); only sizes, token counts, costs, verdicts and timings. The agent spec is reduced to
its kind (``http`` / ``mock``) because spec URLs can embed credentials.
"""

from __future__ import annotations

import json
import logging
import os

from opentelemetry import context as otel_context
from opentelemetry import metrics, propagate, trace
from opentelemetry.trace import Status, StatusCode

from . import __version__

log = logging.getLogger("agentprobe.telemetry")

_meter = metrics.get_meter("agentprobe", __version__)
TASKS = _meter.create_counter("agentprobe.tasks", unit="{task}", description="Tasks evaluated")
AGENT_LATENCY = _meter.create_histogram("agentprobe.agent.latency", unit="ms", description="Agent call latency")
COST = _meter.create_counter("agentprobe.cost", unit="USD", description="Agent and judge spend")


def tracer() -> trace.Tracer:
    # Resolved on each call so a provider installed after import (or by tests) is picked up.
    return trace.get_tracer("agentprobe", __version__)


def mark_error(span: trace.Span, exc: BaseException) -> None:
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, f"{type(exc).__name__}: {exc}"[:200]))


def agent_kind(spec: str) -> str:
    """`http:https://host/path?token=x` -> `http`. Never export the rest of a spec."""
    return spec.partition(":")[0] or "unknown"


def capture_context() -> str:
    """Serialize the current trace context so a run can be continued by another process."""
    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    return json.dumps(carrier) if carrier else ""


def restore_context(serialized: str | None) -> otel_context.Context | None:
    if not serialized:
        return None
    try:
        return propagate.extract(json.loads(serialized))
    except (ValueError, TypeError):
        return None  # a corrupt carrier must never stop a run from executing


_configured = False


def setup_telemetry(service_name: str = "agentprobe") -> bool:
    """Install exporters from the environment. Idempotent; returns True if telemetry is exporting."""
    global _configured
    if _configured:
        return True
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    console = os.environ.get("AGENTPROBE_OTEL_CONSOLE", "").lower() in ("1", "true", "yes")
    if not endpoint and not console:
        return False
    try:
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )
    except ImportError:
        log.warning('OpenTelemetry export requested but the SDK is missing: pip install "agentprobe[otel]"')
        return False

    resource = Resource.create(
        {"service.name": os.environ.get("OTEL_SERVICE_NAME", service_name), "service.version": __version__}
    )
    provider = TracerProvider(resource=resource)
    readers = []
    if endpoint:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        readers.append(PeriodicExportingMetricReader(OTLPMetricExporter()))
    if console:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=readers))
    _configured = True  # providers flush on interpreter exit, so short CLI runs still export
    return True
