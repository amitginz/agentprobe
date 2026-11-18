"""Calibrating a judge against human labels.

An LLM judge is a measuring instrument; before trusting it to gate a merge you measure how often
it agrees with people. We report raw accuracy (with a Wilson interval, because label sets are
small), precision/recall/specificity, and Cohen's kappa, which corrects agreement for chance.
Kappa matters because with 90% of answers passing, a judge that always says "pass" is 90%
"accurate" and completely useless.

Convention: "pass" is the positive class. Human labels are the ground truth.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import HumanLabel, Task, TaskResult


@dataclass
class Calibration:
    n: int
    tp: int  # judge pass, human pass
    fp: int  # judge pass, human fail  (judge too lenient)
    tn: int
    fn: int  # judge fail, human pass  (judge too harsh)
    accuracy: float
    accuracy_ci: tuple[float, float]
    precision: float | None
    recall: float | None
    specificity: float | None
    kappa: float
    reliability: str  # reliable | use_with_caution | unreliable | insufficient_data
    note: str

    def to_dict(self) -> dict:
        return asdict(self)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def cohens_kappa(tp: int, fp: int, tn: int, fn: int) -> float:
    n = tp + fp + tn + fn
    if n == 0:
        return 0.0
    observed = (tp + tn) / n
    expected = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    if expected == 1.0:  # both raters constant and identical: agreement is perfect but untestable
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def _ratio(a: int, b: int) -> float | None:
    return a / b if b else None


def calibrate(pairs: list[tuple[bool, bool]], min_n: int = 30) -> Calibration:
    """`pairs` is a list of (human_passed, judge_passed)."""
    tp = sum(1 for h, j in pairs if h and j)
    fp = sum(1 for h, j in pairs if not h and j)
    tn = sum(1 for h, j in pairs if not h and not j)
    fn = sum(1 for h, j in pairs if h and not j)
    n = len(pairs)
    kappa = cohens_kappa(tp, fp, tn, fn)
    accuracy = (tp + tn) / n if n else 0.0

    humans_pass = tp + fn
    if n < min_n:
        reliability, note = "insufficient_data", f"only {n} labels; collect at least {min_n}"
    elif humans_pass in (0, n):
        reliability = "insufficient_data"
        note = "all human labels are the same class, so kappa cannot separate judge from chance"
    elif kappa >= 0.7:
        reliability, note = "reliable", "substantial agreement with humans"
    elif kappa >= 0.4:
        reliability, note = "use_with_caution", "moderate agreement; review disagreements, refine the rubric"
    else:
        reliability, note = "unreliable", "weak agreement; do not gate on this judge"
    if reliability != "insufficient_data" and fp > fn * 2 and fp >= 3:
        note += "; leans lenient (many false passes)"
    elif reliability != "insufficient_data" and fn > fp * 2 and fn >= 3:
        note += "; leans harsh (many false fails)"

    return Calibration(
        n=n, tp=tp, fp=fp, tn=tn, fn=fn,
        accuracy=accuracy,
        accuracy_ci=wilson_interval(tp + tn, n),
        precision=_ratio(tp, tp + fp),
        recall=_ratio(tp, tp + fn),
        specificity=_ratio(tn, tn + fp),
        kappa=kappa,
        reliability=reliability,
        note=note,
    )


def run_calibration(s: Session, run_id: int, min_n: int = 30) -> Calibration:
    """Judge-vs-human agreement for one run, over its labeled, LLM-judged, non-errored tasks."""
    rows = s.execute(
        select(HumanLabel.passed, TaskResult.passed)
        .join(TaskResult, (TaskResult.run_id == HumanLabel.run_id) & (TaskResult.task_id == HumanLabel.task_id))
        .join(Task, Task.id == HumanLabel.task_id)
        .where(HumanLabel.run_id == run_id, Task.judge == "llm", TaskResult.error == "")
    ).all()
    return calibrate([(h, j) for h, j in rows], min_n)


def label_sample(s: Session, run_id: int, n: int) -> list[dict]:
    """Unlabeled LLM-judged answers for a person to grade.

    The judge's verdict is deliberately withheld so it cannot anchor the human. The shuffle is
    seeded by the run id, so repeated calls return a stable sample.
    """
    labeled = set(s.scalars(select(HumanLabel.task_id).where(HumanLabel.run_id == run_id)))
    rows = s.execute(
        select(Task.id, Task.input, Task.expected, TaskResult.output)
        .join(TaskResult, TaskResult.task_id == Task.id)
        .where(TaskResult.run_id == run_id, Task.judge == "llm", TaskResult.error == "")
        .order_by(Task.id)
    ).all()
    pool = [
        {"task_id": tid, "input": inp, "expected": exp, "output": out}
        for tid, inp, exp, out in rows
        if tid not in labeled
    ]
    random.Random(run_id).shuffle(pool)
    return pool[:n]
