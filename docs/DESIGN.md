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

## LLM-as-judge and calibration

**Why bother with a judge at all.** String rules cannot grade "explain why the sky is blue". But
a model grading a model adds a second source of error, so the design treats the judge as
something to be *measured*, not trusted.

- **Untrusted input.** The agent's answer is delimited and the system prompt says to treat it as
  data. A test asserts the attack text sits inside the fence; this reduces, not eliminates,
  injection risk, which is one more reason to calibrate.
- **Reasoning before verdict, strict JSON, temperature 0.** Auditable and reproducible. A
  malformed reply is retried once and then surfaced as an error; the parser never guesses.
- **Failure isolation.** If the judge API is down, the task is recorded with the agent's output
  and a `judge failed` error and is excluded from pass-rate and latency stats. Counting it as an
  agent failure would make a provider outage look like a regression.
- **Calibration.** Humans label a sample (judge verdict hidden to avoid anchoring; sample
  stable per run). We report agreement with a Wilson interval, precision/recall for "pass", and
  Cohen's kappa. Reliability bands (>= 0.7 reliable, 0.4-0.7 caution, below that unreliable) are the
  conventional rule of thumb, not a theorem, and fewer than 30 labels, or labels of a single
  class, yield `insufficient_data` instead of a misleading number.
- **Bias notes.** Many false passes flags a lenient judge; many false fails, a harsh one.

Not yet handled: self-preference when judge and agent share a model family (use a different
family for the judge), position/verbosity bias tests, and judge drift across model versions
(re-calibrate when `AGENTPROBE_JUDGE_MODEL` changes).

## Dashboard

A single static HTML file served by the API at `/`, with no build step, framework or CDN: it
works offline, ships inside the Python package, and charts are hand-rolled inline SVG. For an
internal tool whose job is to answer "did this change break something?", that removes a whole
toolchain (and a supply-chain surface) for little loss.

- **Untrusted content.** Agent outputs, dataset text and run labels are attacker-influenced
  (an agent can emit `<img onerror=...>`). Every dynamic value goes through one `esc()` helper,
  a test scans every template interpolation for unescaped user fields, and I verified in a real
  browser that a malicious dataset and label render as inert text.
- **The regression view is the point.** A banner states the verdict and the p-value, and the page
  lists the individual tasks that flipped with both outputs side by side, because a number tells
  you *that* something broke and the diff tells you *what*.
- **Charts flag drops, not noise.** A point turns red only if pass rate fell more than 5 points
  versus the previous run; significance testing stays in the compare view where it is computed
  on paired tasks.
- **Live updates by polling** every 2.5 s, and only while a run is pending or running.

Limitations: no pagination beyond the latest 60 runs, and the dataset list is
loaded once at page load (reload to see a newly created dataset).

## Schema migrations

- **Alembic, applied on open.** `Database()` upgrades the schema to the latest revision, so a fresh
  SQLite file, a Postgres instance and an upgraded deployment all end up identical. The baseline
  (`0001`) is written out by hand rather than generated from the models, so it stays frozen while
  the models move on; a test fails whenever the models and the migrated schema diverge, which is
  the signal to write the next migration (`alembic revision --autogenerate` output is a starting point).
- **Databases that predate migrations** are adopted by stamping them at `0001`, but only after
  checking the live schema against the models; one that is missing a column is refused with a clear
  message rather than stamped and left to fail later at runtime.
- **Concurrent starters.** The API and every worker open the database at boot. On PostgreSQL a
  session-level advisory lock makes one of them migrate while the others wait and then find nothing to do.
- `agentprobe db upgrade|current|check` for explicit control (`check` exits 1 on a stale revision or
  drift, usable as a deploy gate). SQLite migrations use batch mode so later `ALTER`s work.

## Authentication

- **Two scopes, one dependency.** `AGENTPROBE_API_KEYS` grant full access; `AGENTPROBE_READ_API_KEYS`
  allow only GET. Several comma-separated keys per scope make rotation a two-step deploy (add the
  new key, then remove the old). Keys travel as `Authorization: Bearer` or `X-API-Key`.
- **Deny by default for data.** Every data route hangs off one authenticated router, and a test
  walks the app's routes and fails if any non-public one answers without a key, so a new route
  added on the wrong object cannot ship open by accident. `/health` and the dashboard page are
  public (the page holds no data; the browser asks for a key and keeps it in `sessionStorage` only).
- **Constant-time comparison** against every configured key, and rejected keys are never echoed.
- **Open mode is explicit.** With no keys configured the API stays open for local use and logs a
  warning; `AGENTPROBE_REQUIRE_AUTH=1` makes that a startup error, for deployments.

## Telemetry

Why traces and not only logs: an evaluation is a fan-out (run, tasks, retries, agent and judge calls
across threads, processes and a queue), and the questions asked of it are causal ("which task's
retry made this run slow?", "did the worker that resumed this run share its trace?").

- **Propagation through a queue.** Thread pools and Redis lists do not carry Python contextvars, so
  context is passed explicitly: captured into a W3C carrier at `create_run`, persisted in
  `runs.trace_context` (durable, survives re-delivery), restored by `execute_run`, and handed to
  task threads as an explicit parent context. A corrupt carrier is ignored rather than failing a run.
- **Cardinality and privacy.** Only low-cardinality attributes go on metrics; sensitive content
  never goes on spans (see the module docstring and tests) Exceptions are the sneaky channel: an
  HTTP client's error text repeats the request URL, and the SDK's `record_exception` also exports a
  stack trace, so failures are recorded by hand (type plus a URL-redacted, truncated message). A real
  Jaeger trace exposed this; the earlier privacy test only inspected span attributes and now covers
  status text and event attributes too.
- **Sampling** is left to the collector or the standard `OTEL_TRACES_SAMPLER` variables.

## Power and stability

An inconclusive comparison is only useful if you know *why* it was inconclusive. Two separate
causes, two tools:

- **Too few informative tasks (power).** Power is computed exactly: discordant tasks are
  Binomial(N, d), the fixed share among them is Binomial(D, p), and the exact two-sided test
  rejects when min(broken, fixed) is at most a critical count per D. The implementation is checked
  against a simulation of the real test (agreement within 2 points), against type-I error (never
  above alpha at zero effect), and for internal consistency between required N and detectable
  effect. A non-obvious consequence worth knowing: for a fixed effect, a *higher* discordance rate
  needs *more* tasks, because disagreement is noise unless it is lopsided.
- **A noisy agent (stability).** The paired test assumes each task has one outcome per version. For
  a non-deterministic agent that is false, and the noise floor can exceed the effect being looked
  for. Repeats expose it directly (flaky tasks, SD of pass rate) instead of assuming it away.

The CI report's sensitivity line floors the observed discordance at 5%, so a suspiciously quiet
run cannot claim to detect tiny changes.

Not done: using repeats inside `compare` itself (per-task pass *probabilities* with a paired
bootstrap) would use the extra information rather than only reporting noise; that is the natural
next statistical step.

## Measured scaling

Eight workers deliver 7.84x the throughput of one (98% efficiency) when each task takes 20 ms, so the
queue protocol (BLMOVE, lease write, heartbeat thread, claim CAS) is not the bottleneck. With no agent
latency the system saturates near 4,400 tasks/s regardless of worker count, which isolates the real
limit: SQLite's single writer, serialized by per-result commits. Two consequences. First, per-result
commits (chosen for crash-safe resume) cost nothing noticeable at realistic agent latencies. Second,
if throughput on very fast agents ever mattered, batching commits or moving to PostgreSQL would be the
levers, in that order, and the resume guarantee would then need re-checking against batch size.

## Known limitations
- Heartbeats prove the worker *process* is alive, not that the agent call is making progress; a
  hung agent call is bounded only by the agent's own timeout (HTTP agents default to 60 s).
- Redis behaviour was also verified against a real Redis 8 server (integration test and benchmark), not only the fake.
- Lease expiry assumes roughly synchronized clocks between workers (deadlines are wall-clock).
- Authentication is shared API keys (full or read-only), not per-user accounts: there is no key
  management, audit log, rate limiting or per-dataset access control. Put the API behind TLS.
- The Postgres advisory lock that serialises concurrent migrations is not covered by the automated
  tests (they run on SQLite); the migration files themselves are dialect-neutral.
- The Anthropic client is tested against a scripted stub, not against the live API.
- Latency numbers include only the agent call, measured with `perf_counter`.
