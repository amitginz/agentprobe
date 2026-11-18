# AgentProbe

Regression testing for LLM agents. Run an agent against a dataset, record a full trace of every
task (output, tool calls, tokens, latency, cost), and answer the question that matters before you
ship a prompt or model change: **did it get better, worse, or is that just noise?**

```text
$ agentprobe run --dataset examples/dataset.json --agent "mock:accuracy=0.95" --label v1
run 1 [completed] 39/40 passed
$ agentprobe run --dataset examples/dataset.json --agent "mock:accuracy=0.60" --label v2
run 2 [completed] 27/40 passed
$ agentprobe compare --baseline 1 --candidate 2 --fail-on-regression
REGRESSION: pass rate 97.5% -> 67.5% (broken=12, fixed=0, p=0.0005)
$ echo $?
1
```

The non-zero exit code is the point: drop the last command into CI and a PR that degrades your
agent fails the build, the same way a failing unit test would.

## Why not just compare pass rates?

Agent evals are small and noisy. "92% vs 88%" on 50 tasks is usually nothing. Because both runs
answer the *same* tasks, AgentProbe pairs the outcomes and applies an **exact McNemar test** to the
tasks that flipped (broke vs. fixed). A single flaky task does not trigger a regression; twelve
broken tasks and zero fixed ones does. The test is implemented in ~10 lines of stdlib code
([`compare.py`](src/agentprobe/compare.py)) and unit-tested against known values.

## Features

| Area | What is implemented |
|---|---|
| Execution | Thread-pool runner with bounded concurrency, per-task retries, and a **hard cost budget** that stops a run early (`budget_exceeded`) |
| Judges | `exact`, `contains`, `regex`, `tool_called` (pluggable registry) |
| Agents | Spec-string adapters: `mock:` (deterministic simulated agent) and `http:` (any remote agent that returns JSON) |
| Analysis | Paired comparison: pass-rate delta, broken/fixed counts, exact p-value, cost and p95 latency |
| API | FastAPI: datasets, runs (async, background execution), results, `/compare` |
| Storage | SQLAlchemy 2.0; SQLite by default, PostgreSQL via `DATABASE_URL` |
| Ops | Dockerfile, docker-compose with Postgres, GitHub Actions CI with an agent-quality gate |

Run the service:

```bash
pip install -e ".[dev]"
uvicorn agentprobe.api:app_factory --factory --reload   # http://localhost:8000/docs
docker compose up --build                                # API + PostgreSQL
pytest
```

Evaluate your own agent by exposing an endpoint that accepts `{"input": "..."}` and returns
`{"output": "...", "tool_calls": [...], "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}`:

```bash
agentprobe run --dataset my_tasks.json --agent "http:http://localhost:9000/agent" --budget 2.50
```

## Roadmap

Honest status: the core loop above works and is tested. Planned next:

- [ ] Redis-backed job queue + separate worker processes (today runs execute in-process)
- [ ] LLM-as-judge with calibration against human labels
- [ ] Web dashboard (pass-rate, cost and latency trends per agent version)
- [ ] OpenTelemetry traces for each agent step
- [ ] Reusable GitHub Action wrapper
- [ ] Evaluate docs-rag-mcp as the first real agent under test

See [docs/DESIGN.md](docs/DESIGN.md) for the architecture and the trade-offs behind it.
