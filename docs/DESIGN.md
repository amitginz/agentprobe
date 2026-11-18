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

## Job queue and workers

```text
API ──enqueue──► pending ──BLMOVE (atomic)──► processing ──ack──► done
                    ▲                            │ heartbeat extends lease
                    └──── reaper requeues ◄──────┘ when the lease expires
```

The goal is that no accepted run is ever lost, and that a crashed worker costs seconds rather than
a manual restart. Runs are the unit of work (not tasks) because a run is where budget and
ordering live; concurrency inside a run is the runner's thread pool.

**At-least-once delivery, made safe by idempotency.** A job moves from `pending` to `processing`
in one atomic Redis command, so a worker dying right after taking it cannot lose it. Each worker
heartbeats its lease from a side thread; if heartbeats stop, any worker's `reap()` requeues the job.
Exactly-once delivery is not achievable across a network failure, so duplicates are made harmless
instead: `claim_run` is a compare-and-set (`pending -> running`) in the database, so of two
workers holding the same run only one executes it.

**Resume instead of restart.** Each task result is committed as it completes and `execute_run`
skips tasks that already have a result. A re-delivered run therefore continues where the dead
worker stopped, and the budget check includes the spend of earlier attempts.

**A missing lease is adopted, not requeued.** Between BLMOVE and the first lease write a job
legitimately has no lease. Requeuing on that state would hand one job to two workers, so the
reaper only sets the lease and lets it expire normally if the worker really is gone.

**Poison jobs.** Every hand-out increments a delivery counter. A run delivered more than
`max_deliveries` times (default 3), meaning it keeps killing its workers, is marked `failed`
instead of cycling forever.

**API degrades loudly.** If Redis is down at enqueue time the API returns 503 and marks the run
`failed` rather than leaving it `pending` forever; `/health` reports queue depth or `degraded`.

**Why Redis lists and not Celery/RQ.** The semantics needed here (leases, delivery counting,
run-level idempotency) are about 100 lines, fully tested, and visible. A framework would hide
exactly the part worth explaining. For larger scale I would move to Redis Streams with consumer
groups, which give pending-entry tracking and `XAUTOCLAIM` natively.

## Known limitations
- Heartbeats prove the worker *process* is alive, not that the agent call is making progress; a
  hung agent call is bounded only by the agent's own timeout (HTTP agents default to 60 s).
- Lease expiry assumes roughly synchronized clocks between workers (deadlines are wall-clock).
- No authentication: intended for internal or local use.
- LLM-as-judge is not implemented yet; judges are deterministic.
- Latency numbers include only the agent call, measured with `perf_counter`.
