"""Turn docs-rag-mcp's hand-written eval sets into AgentProbe datasets.

    python examples/build_docs_rag_datasets.py [~/Downloads/docs-rag-mcp]

For each question we check what the agent actually *did*, with deterministic judges:
  in-scope      -> the expected source file appears in the ranked retrieval list
  out-of-scope  -> the agent refused ("I don't know") instead of answering from thin air
"""

import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "~/Downloads/docs-rag-mcp").expanduser()
out_dir = Path(__file__).parent

for name, filename in [("docs-rag-easy", "eval_set.json"), ("docs-rag-hard", "eval_hard.json")]:
    data = json.loads((root / "eval" / filename).read_text())
    tasks = [
        {
            "input": item["q"],
            "expected": r"Retrieved[^\n]*" + re.escape(item["source"]),
            "judge": "regex",
        }
        for item in data["in_scope"]
    ] + [
        {"input": q, "expected": "I don't know", "judge": "contains"} for q in data["out_of_scope"]
    ]
    target = out_dir / (name.replace("-", "_") + ".json")
    target.write_text(json.dumps({"name": name, "tasks": tasks}, indent=1) + "\n")
    print(f"{target.name}: {len(data['in_scope'])} in-scope + {len(data['out_of_scope'])} out-of-scope")
