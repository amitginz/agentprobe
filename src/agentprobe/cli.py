"""Command line interface, designed so CI can gate merges on agent quality.

    agentprobe run --dataset examples/dataset.json --agent mock:accuracy=0.9 --label v1
    agentprobe compare --baseline 1 --candidate 2 --fail-on-regression
    agentprobe ci --dataset d.json --baseline-agent A --candidate-agent B   # both steps, for CI
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


def cmd_calibrate(args: argparse.Namespace) -> int:
    from .calibration import run_calibration
    from .db import HumanLabel

    db = Database(args.db)
    with db.session() as s:
        if s.get(Run, args.run) is None:
            print("run not found", file=sys.stderr)
            return 2
        if args.labels:
            with open(args.labels) as fh:
                labels = json.load(fh)  # [{"task_id": 1, "passed": true}, ...]
            existing = {
                lb.task_id: lb for lb in s.scalars(select(HumanLabel).where(HumanLabel.run_id == args.run))
            }
            for item in labels:
                if item["task_id"] in existing:
                    existing[item["task_id"]].passed = bool(item["passed"])
                else:
                    s.add(HumanLabel(run_id=args.run, task_id=item["task_id"], passed=bool(item["passed"])))
            s.commit()
        cal = run_calibration(s, args.run, args.min_n)
    print(json.dumps(cal.to_dict(), indent=2))
    print(f"\n{cal.reliability.upper()}: kappa={cal.kappa:.2f}, agreement={cal.accuracy:.1%} on {cal.n} labels. {cal.note}")
    return 1 if args.require_reliable and cal.reliability != "reliable" else 0


def cmd_ci(args: argparse.Namespace) -> int:
    from .ci import render_markdown, run_ci, to_json, write_github_outputs

    outcome = run_ci(
        Database(args.db),
        args.dataset,
        args.baseline_agent,
        args.candidate_agent,
        concurrency=args.concurrency,
        budget_usd=args.budget,
        alpha=args.alpha,
    )
    markdown = render_markdown(outcome, args.baseline_agent, args.candidate_agent)
    print(to_json(outcome))
    write_github_outputs(outcome, markdown)
    label = outcome.verdict.replace("_", " ").upper()
    print(f"\n{label}", file=sys.stderr)
    if outcome.verdict == "infrastructure_error":
        return 2  # no verdict: never swallowed by --no-fail, because nothing was actually checked
    return 0 if args.no_fail else outcome.exit_code_failure


def cmd_power(args: argparse.Namespace) -> int:
    from .power import min_detectable_effect, power_for_effect, required_tasks

    d, alpha, target = args.discordance, args.alpha, args.target_power
    if args.effect is not None and args.tasks is not None:
        p = power_for_effect(args.tasks, d, args.effect, alpha)
        print(f"{args.tasks} tasks, {d:.0%} of tasks flipping: a {args.effect * 100:.1f}-point change is detected {p:.0%} of the time.")
    elif args.effect is not None:
        n = required_tasks(args.effect, d, alpha, target)
        msg = f"about {n} tasks" if n else "more than 5000 tasks"
        print(f"To detect a {args.effect * 100:.1f}-point change with {target:.0%} power (alpha {alpha}, {d:.0%} discordance) you need {msg}.")
    elif args.tasks is not None:
        mde = min_detectable_effect(args.tasks, d, alpha, target)
        if mde is None:
            print(f"{args.tasks} tasks cannot reach {target:.0%} power at {d:.0%} discordance, whatever the effect.")
        else:
            print(f"{args.tasks} tasks, {d:.0%} of tasks flipping: changes of {mde * 100:.1f} points or more are detected with {target:.0%} power.")
    else:
        print("give --tasks, --effect, or both", file=sys.stderr)
        return 2
    return 0


def cmd_stability(args: argparse.Namespace) -> int:
    from .stability import analyze

    db = Database(args.db)
    dataset_id = _load_dataset(db, args.dataset)
    ids = []
    for i in range(args.repeats):
        with db.session() as s:
            run = Run(dataset_id=dataset_id, agent=args.agent, label=f"repeat {i + 1}", concurrency=args.concurrency)
            s.add(run)
            s.commit()
            ids.append(run.id)
        execute_run(db, ids[-1])
    with db.session() as s:
        runs = [list(s.get(Run, i).results) for i in ids]
        report = analyze(runs)
    print(json.dumps(report.to_dict(), indent=2))
    print(
        f"\n{report.flaky}/{report.tasks} tasks flaky ({report.flaky_fraction:.0%}); pass rate "
        f"{report.mean_pass_rate:.1%} +/- {report.pass_rate_sd * 100:.1f} pts. Changes under "
        f"~{report.noise_floor_pts:.1f} points are indistinguishable from re-running the same agent."
    )
    return 1 if args.max_flaky is not None and report.flaky_fraction > args.max_flaky else 0


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

    p_cal = sub.add_parser("calibrate", help="measure an LLM judge's agreement with human labels")
    p_cal.add_argument("--run", type=int, required=True)
    p_cal.add_argument("--labels", help="JSON file of [{task_id, passed}] to import first")
    p_cal.add_argument("--min-n", type=int, default=30)
    p_cal.add_argument("--require-reliable", action="store_true", help="exit 1 unless the judge is reliable")
    p_cal.set_defaults(func=cmd_calibrate)

    p_ci = sub.add_parser("ci", help="run baseline + candidate agents and gate on the comparison")
    p_ci.add_argument("--dataset", required=True)
    p_ci.add_argument("--baseline-agent", required=True)
    p_ci.add_argument("--candidate-agent", required=True)
    p_ci.add_argument("--concurrency", type=int, default=4)
    p_ci.add_argument("--budget", type=float, default=None, help="max spend in USD per run")
    p_ci.add_argument("--alpha", type=float, default=0.05)
    p_ci.add_argument("--no-fail", action="store_true", help="report a regression but exit 0")
    p_ci.set_defaults(func=cmd_ci)

    p_pw = sub.add_parser("power", help="how large a change can a dataset of N tasks detect?")
    p_pw.add_argument("--tasks", type=int, help="dataset size")
    p_pw.add_argument("--effect", type=float, help="true change in pass rate, e.g. 0.08 for 8 points")
    p_pw.add_argument("--discordance", type=float, default=0.15, help="share of tasks whose outcome flips between versions")
    p_pw.add_argument("--alpha", type=float, default=0.05)
    p_pw.add_argument("--target-power", type=float, default=0.8)
    p_pw.set_defaults(func=cmd_power)

    p_st = sub.add_parser("stability", help="repeat an agent on a dataset and measure run-to-run noise")
    p_st.add_argument("--dataset", required=True)
    p_st.add_argument("--agent", required=True)
    p_st.add_argument("--repeats", type=int, default=5)
    p_st.add_argument("--concurrency", type=int, default=4)
    p_st.add_argument("--max-flaky", type=float, default=None, help="exit 1 if more than this fraction of tasks is flaky")
    p_st.set_defaults(func=cmd_stability)

    p_wk = sub.add_parser("worker", help="run a queue worker that executes enqueued runs")
    p_wk.add_argument("--redis-url", default=os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    p_wk.add_argument("--lease", type=float, default=60.0, help="seconds before a silent worker's job is requeued")
    p_wk.set_defaults(func=cmd_worker)

    args = parser.parse_args(argv)
    from .telemetry import setup_telemetry

    setup_telemetry("agentprobe-worker" if args.command == "worker" else "agentprobe")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
