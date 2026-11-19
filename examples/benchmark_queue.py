"""Measure how queue throughput scales with the number of worker processes.

    redis-server --port 6390 --save "" &            # any Redis works
    python examples/benchmark_queue.py --redis-url redis://localhost:6390/0

Each worker is a separate OS process running `agentprobe worker`. The agent is a simulated one with
a fixed per-call latency and every run uses concurrency 1, so what is measured is the *queue and
worker layer* (claiming, leases, DB writes), not the thread pool inside a run.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from agentprobe.db import Database, Dataset, Run, Task
from agentprobe.jobqueue import RedisQueue

DATASET = json.loads((Path(__file__).parent / "dataset.json").read_text())


def one_trial(workers: int, runs: int, latency_ms: int, redis_url: str) -> dict:
    tmp = tempfile.mkdtemp()
    db_url = f"sqlite:///{tmp}/bench.db"
    db = Database(db_url)
    queue_name = f"bench-{uuid.uuid4().hex}"
    with db.session() as s:
        ds = Dataset(name="bench", tasks=[Task(**t) for t in DATASET["tasks"]])
        s.add(ds)
        s.commit()
        ids = []
        for _ in range(runs):
            run = Run(dataset_id=ds.id, agent=f"mock:accuracy=0.9,latency={latency_ms}", concurrency=1)
            s.add(run)
            s.commit()
            ids.append(run.id)
    queue = RedisQueue.from_url(redis_url, name=queue_name, lease_seconds=30)

    env = {**os.environ, "REDIS_URL": redis_url}
    code = (
        "from agentprobe.db import Database;from agentprobe.jobqueue import RedisQueue;"
        "from agentprobe.worker import Worker;import os,sys;"
        f"q=RedisQueue.from_url(os.environ['REDIS_URL'], name='{queue_name}', lease_seconds=30);"
        f"Worker(Database('{db_url}'), q, poll_timeout=0.2).run_forever()"
    )
    procs = [subprocess.Popen([sys.executable, "-c", code], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
             for _ in range(workers)]
    time.sleep(3)  # let every interpreter finish importing before the clock starts

    start = time.perf_counter()
    for run_id in ids:
        queue.enqueue(run_id)
    done = 0
    while done < runs:
        time.sleep(0.1)
        with db.session() as s:
            done = sum(1 for i in ids if s.get(Run, i).status in ("completed", "failed", "budget_exceeded"))
        if time.perf_counter() - start > 300:
            raise TimeoutError("benchmark did not finish")
    elapsed = time.perf_counter() - start

    for p in procs:
        p.terminate()
    for p in procs:
        p.wait(timeout=10)
    with db.session() as s:
        statuses = [s.get(Run, i).status for i in ids]
        results = sum(len(s.get(Run, i).results) for i in ids)
    return {"elapsed": elapsed, "tasks": results, "ok": all(st == "completed" for st in statuses)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--redis-url", default="redis://localhost:6379/0")
    parser.add_argument("--workers", default="1,2,4,8")
    parser.add_argument("--runs", type=int, default=24)
    parser.add_argument("--latency-ms", type=int, default=20)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()

    print(f"{args.runs} runs x {len(DATASET['tasks'])} tasks, {args.latency_ms} ms simulated agent latency, "
          f"{args.trials} trials per setting, median reported")
    print(f"{'workers':>8} {'tasks/s':>9} {'speedup':>8} {'efficiency':>11}  trials(tasks/s)")
    base = None
    for w in [int(x) for x in args.workers.split(",")]:
        rates = []
        for _ in range(args.trials):
            r = one_trial(w, args.runs, args.latency_ms, args.redis_url)
            assert r["ok"], "a run did not complete"
            rates.append(r["tasks"] / r["elapsed"])
        med = statistics.median(rates)
        base = base or med
        print(f"{w:>8} {med:>9.1f} {med / base:>7.2f}x {med / base / w:>10.0%}  "
              + ", ".join(f"{x:.1f}" for x in rates))


if __name__ == "__main__":
    main()
