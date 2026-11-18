"""Command line interface, designed so CI can gate merges on agent quality.

    agentprobe run --dataset examples/dataset.json --agent mock:accuracy=0.9 --label v1
    agentprobe compare --baseline 1 --candidate 2 --fail-on-regression
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from sqlalchemy import select

from .compare import compare_runs
from .db import Database, Dataset, Run, Task
from .runner import execute_run


def _load_dataset(db: Database, path: str) -> int:
    with open(path) as fh:
        data = json.load(fh)
    with db.session() as s:
        ds = s.scalar(select(Dataset).where(Dataset.name == data["name"]))
        if ds is None:
            ds = Dataset(name=data["name"], tasks=[Task(**t) for t in data["tasks"]])
            s.add(ds)
            s.commit()
        return ds.id


def cmd_run(args: argparse.Namespace) -> int:
    db = Database(args.db)
    dataset_id = _load_dataset(db, args.dataset)
    with db.session() as s:
        run = Run(
            dataset_id=dataset_id,
            agent=args.agent,
            label=args.label,
            concurrency=args.concurrency,
            budget_usd=args.budget,
        )
        s.add(run)
        s.commit()
        run_id = run.id
    execute_run(db, run_id)
    with db.session() as s:
        run = s.get(Run, run_id)
        passed = sum(r.passed for r in run.results)
        print(f"run {run_id} [{run.status}] {passed}/{len(run.results)} passed ({args.agent})")
        return 0 if run.status in ("completed", "budget_exceeded") else 1


def cmd_compare(args: argparse.Namespace) -> int:
    db = Database(args.db)
    with db.session() as s:
        b, c = s.get(Run, args.baseline), s.get(Run, args.candidate)
        if b is None or c is None:
            print("run not found", file=sys.stderr)
            return 2
        cmp = compare_runs(list(b.results), list(c.results), args.alpha)
    print(json.dumps(cmp.to_dict(), indent=2))
    print(
        f"\n{cmp.verdict.upper()}: pass rate {cmp.baseline.pass_rate:.1%} -> "
        f"{cmp.candidate.pass_rate:.1%} (broken={cmp.broken}, fixed={cmp.fixed}, p={cmp.p_value:.4f})"
    )
    return 1 if args.fail_on_regression and cmp.verdict == "regression" else 0


def cmd_worker(args: argparse.Namespace) -> int:
    import logging
    import signal
    import threading

    from .jobqueue import RedisQueue
    from .worker import Worker

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())  # finish the current run, then exit
    queue = RedisQueue.from_url(args.redis_url, lease_seconds=args.lease)
    Worker(Database(args.db), queue).run_forever(stop)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentprobe")
    parser.add_argument("--db", default=None, help="database URL (default: sqlite:///agentprobe.db)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run an agent against a dataset file")
    p_run.add_argument("--dataset", required=True)
    p_run.add_argument("--agent", required=True)
    p_run.add_argument("--label", default="")
    p_run.add_argument("--concurrency", type=int, default=4)
    p_run.add_argument("--budget", type=float, default=None, help="max spend in USD")
    p_run.set_defaults(func=cmd_run)

    p_cmp = sub.add_parser("compare", help="compare two runs")
    p_cmp.add_argument("--baseline", type=int, required=True)
    p_cmp.add_argument("--candidate", type=int, required=True)
    p_cmp.add_argument("--alpha", type=float, default=0.05)
    p_cmp.add_argument("--fail-on-regression", action="store_true")
    p_cmp.set_defaults(func=cmd_compare)

    p_wk = sub.add_parser("worker", help="run a queue worker that executes enqueued runs")
    p_wk.add_argument("--redis-url", default=os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    p_wk.add_argument("--lease", type=float, default=60.0, help="seconds before a silent worker's job is requeued")
    p_wk.set_defaults(func=cmd_worker)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
