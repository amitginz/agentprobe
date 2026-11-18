"""Fill a database with a realistic history of runs so the dashboard has something to show.

    python examples/seed_demo.py                       # writes ./agentprobe.db
    uvicorn agentprobe.api:app_factory --factory       # open http://localhost:8000
"""

import json
import sys
from pathlib import Path

from agentprobe.db import Database, Dataset, Run, Task
from agentprobe.runner import execute_run

HISTORY = [  # (label, accuracy, seed, latency ms): improvement, a bad model swap, then a fix
    ("baseline prompt", 0.85, 1, 38),
    ("added few-shot examples", 0.90, 2, 42),
    ("tool descriptions v2", 0.92, 3, 40),
    ("switch to cheaper model", 0.62, 4, 14),
    ("rollback + tighter rubric", 0.90, 5, 41),
    ("retrieval reranking", 0.95, 6, 55),
]

url = sys.argv[1] if len(sys.argv) > 1 else None
db = Database(url)
data = json.loads((Path(__file__).parent / "dataset.json").read_text())
with db.session() as s:
    ds = Dataset(name=data["name"], tasks=[Task(**t) for t in data["tasks"]])
    s.add(ds)
    s.commit()
    run_ids = []
    for label, acc, seed, lat in HISTORY:
        run = Run(dataset_id=ds.id, agent=f"mock:accuracy={acc},seed={seed},latency={lat}", label=label, concurrency=4)
        s.add(run)
        s.commit()
        run_ids.append(run.id)
for rid in run_ids:
    execute_run(db, rid)
print(f"seeded {len(run_ids)} runs into {db.url}")
