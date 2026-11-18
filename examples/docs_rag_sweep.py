"""Grid-search the docs-rag relevance settings on the TRAINING datasets only.

    python examples/docs_rag_sweep.py            # needs docs_rag_agent.py running on :9000

The held-out set (docs_rag_holdout.json) is deliberately not touched here: it is evaluated once,
for the single configuration this sweep selects, so its score is not inflated by tuning.
"""

import itertools
import json
from pathlib import Path

from agentprobe.db import Database, Dataset, Run, Task
from agentprobe.runner import execute_run

BASE = "http:http://localhost:9000/agent"
TRAIN = ["docs_rag_easy.json", "docs_rag_hard.json"]
DISTANCES = [0.75, 0.95]
RERANK_MINS = [None, -10, -8, -6, -5, -4, -3, -2, 0]

db = Database()
here = Path(__file__).parent
dataset_ids = []
with db.session() as s:
    for name in TRAIN:
        data = json.loads((here / name).read_text())
        ds = s.query(Dataset).filter_by(name=data["name"]).first()
        if ds is None:
            ds = Dataset(name=data["name"], tasks=[Task(**t) for t in data["tasks"]])
            s.add(ds)
            s.commit()
        dataset_ids.append(ds.id)

rows = []
for dist, rmin in itertools.product(DISTANCES, RERANK_MINS):
    query = f"k=3&max_distance={dist}" + (f"&rerank_min={rmin}" if rmin is not None else "")
    passed = total = ins_ok = ins_n = oos_ok = oos_n = 0
    for ds_id in dataset_ids:
        with db.session() as s:
            run = Run(dataset_id=ds_id, agent=f"{BASE}?{query}", label=f"sweep {query}", concurrency=1)
            s.add(run)
            s.commit()
            run_id = run.id
        execute_run(db, run_id)
        with db.session() as s:
            for r in s.get(Run, run_id).results:
                judge = s.get(Task, r.task_id).judge
                total += 1
                passed += r.passed
                if judge == "regex":
                    ins_n += 1
                    ins_ok += r.passed
                else:
                    oos_n += 1
                    oos_ok += r.passed
    rows.append((passed / total, dist, rmin, ins_ok, ins_n, oos_ok, oos_n))

print(f"{'distance':>8} {'rerank_min':>10} {'in-scope':>9} {'refused':>8} {'pass':>7}")
for acc, dist, rmin, io, i_n, oo, o_n in sorted(rows, key=lambda r: -r[0]):
    print(f"{dist:>8} {rmin!s:>10} {io:>4}/{i_n:<4} {oo:>3}/{o_n:<4} {acc:>7.1%}")
