"""Install in-memory OpenTelemetry providers once for the whole test session.

OpenTelemetry allows one global provider per process, so tests share these and clear them between
tests via the `spans` / `metric_reader` fixtures.
"""

import pytest
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

_EXPORTER = InMemorySpanExporter()
_READER = InMemoryMetricReader()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_EXPORTER))
trace.set_tracer_provider(_provider)
metrics.set_meter_provider(MeterProvider(metric_readers=[_READER]))


@pytest.fixture
def spans():
    _EXPORTER.clear()
    yield _EXPORTER
    _EXPORTER.clear()


@pytest.fixture
def metric_reader():
    _READER.get_metrics_data()  # drain anything recorded by earlier tests
    return _READER
