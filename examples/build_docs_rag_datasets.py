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

for name, filename in [
    ("docs-rag-easy", "eval_set.json"),
    ("docs-rag-hard", "eval_hard.json"),
    ("docs-rag-holdout", "eval_holdout.json"),
    ("docs-rag-holdout2", "eval_holdout2.json"),
]:
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


# One combined, LLM-judged dataset for calibrating the `llm` judge against human labels.
IN_SCOPE_RUBRIC = (
    "PASS if the answer text contains information that directly answers the question (an excerpt "
    "from documentation is fine). FAIL if it is off-topic, only loosely related, or says it does not know."
)
OUT_RUBRIC = (
    "PASS only if the answer says it does not know or that the documents do not cover the question. "
    "FAIL if it presents documentation text as if it answered the question."
)
llm_tasks, seen = [], set()
for filename in ("eval_set.json", "eval_hard.json", "eval_holdout.json", "eval_holdout2.json"):
    data = json.loads((root / "eval" / filename).read_text())
    for item in data["in_scope"]:
        if item["q"] not in seen:
            seen.add(item["q"])
            llm_tasks.append({
                "input": item["q"],
                "expected": f"Answerable from {item['source']}.",
                "judge": "llm",
                "judge_config": {"rubric": IN_SCOPE_RUBRIC},
            })
    for q in data["out_of_scope"]:
        if q not in seen:
            seen.add(q)
            llm_tasks.append({
                "input": q,
                "expected": "Not covered by the documents; the correct response is to say it does not know.",
                "judge": "llm",
                "judge_config": {"rubric": OUT_RUBRIC},
            })
(out_dir / "docs_rag_llm.json").write_text(json.dumps({"name": "docs-rag-llm", "tasks": llm_tasks}, indent=1) + "\n")
print(f"docs_rag_llm.json: {len(llm_tasks)} tasks for LLM-judge calibration")


# Both held-out sets together (97 questions): the pooled, higher-power evaluation.
pooled = []
for name in ("docs_rag_holdout.json", "docs_rag_holdout2.json"):
    pooled += json.loads((out_dir / name).read_text())["tasks"]
(out_dir / "docs_rag_holdout_all.json").write_text(json.dumps({"name": "docs-rag-holdout-all", "tasks": pooled}, indent=1) + "\n")
print(f"docs_rag_holdout_all.json: {len(pooled)} tasks (both held-out sets)")
