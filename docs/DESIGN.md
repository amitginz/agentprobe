# Design notes

## Goal
Make "is this agent change safe to ship?" a fast, automated, statistically honest check.

## Architecture

```text
 CLI / CI ──┐
            ├──► FastAPI ──► Run record (pending) ──► Runner ──► Agent adapter ──► agent under test
 REST ──────┘        │                                   │
                     │                                   └──► Judge ──► TaskResult (trace)
                     └──► /compare ──► paired statistics over two runs' TaskResults
                                   storage: SQLAlchemy → SQLite | PostgreSQL
```

A run is described entirely by plain data (dataset id, agent spec string, concurrency, retries,
budget), so any run can be reproduced or re-queued from its row alone.

## Decisions and trade-offs

**Paired exact McNemar test instead of comparing two proportions.** Both runs answer identical
tasks, so outcomes are paired. Only discordant pairs carry information about the change. The exact
(binomial) form is correct for the small samples typical of agent evals, where the chi-square
approximation is not. Limitation: it only detects pass/fail shifts, not score magnitude; a
continuous-score test (paired bootstrap) is a possible extension.

**Single writer thread.** Workers only call the agent and judge. The coordinating thread persists
results. This avoids SQLite write contention and keeps persistence in one ordered code path.
Trade-off: results appear as tasks finish rather than being written by each worker, which is
fine at this scale and simple to reason about.

**Budget enforced at the coordinator.** Cost is summed as results arrive; once the budget is
exceeded, pending tasks are cancelled and the run is marked `budget_exceeded`. Tasks already in
flight can overshoot by up to `concurrency` task-costs, so the budget is a hard stop in practice but not
to the cent. A reservation scheme would make it exact at the price of complexity.

**Agent exceptions are data, not crashes.** A failing agent call is retried, then recorded as a
failed task with its error, so one bad task never loses the rest of the run.

**Agents and judges selected by name.** Spec strings (`mock:accuracy=0.9`, `http:URL`) keep runs
reproducible and let the API validate them up front (422 before anything is queued).

**Deterministic mock agent.** Correctness is a function of `(seed, input)`, so tests and the CI
demo are reproducible, and two mock agents with different accuracy give a controlled
baseline/candidate pair to prove the comparison logic detects a regression of known size.

## Known limitations
- Runs execute in the API process (background task). A crash mid-run leaves the run in `running`;
  the Redis worker design on the roadmap fixes this with leases and re-delivery.
- No authentication: intended for internal or local use.
- LLM-as-judge is not implemented yet; judges are deterministic.
- Latency numbers include only the agent call, measured with `perf_counter`.
