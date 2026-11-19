"""The docs-rag datasets must stay consistent with the output format of examples/docs_rag_agent.py."""

import json
from pathlib import Path

import pytest

from agentprobe.agents import AgentResult
from agentprobe.judges import judge

EXAMPLES = Path(__file__).parent.parent / "examples"
REFUSAL = "I don't know: the indexed documents do not cover this question."


def agent_output(answer: str, ranked: list[str]) -> str:
    """Mirror of handle_question() in docs_rag_agent.py."""
    return f"{answer}\nRetrieved (ranked): {', '.join(ranked) or '(none)'}\nGrounded: {bool(ranked)}"


@pytest.mark.parametrize("name", ["docs_rag_easy", "docs_rag_hard"])
def test_dataset_is_well_formed_and_judges_behave(name):
    data = json.loads((EXAMPLES / f"{name}.json").read_text())
    in_scope = [t for t in data["tasks"] if t["judge"] == "regex"]
    out_scope = [t for t in data["tasks"] if t["judge"] == "contains"]
    assert in_scope and out_scope and len(in_scope) + len(out_scope) == len(data["tasks"])

    for t in in_scope:
        source = t["expected"].split("Retrieved[^\\n]*")[1].replace("\\", "")
        hit = AgentResult(output=agent_output("some passage", ["other.md", source]))
        miss = AgentResult(output=agent_output("some passage", ["other.md"]))
        assert judge("regex", t["expected"], {}, hit).passed
        assert not judge("regex", t["expected"], {}, miss).passed
        # the source name must not be matched from the answer text, only from the ranked line
        sneaky = AgentResult(output=agent_output(f"see {source}", ["other.md"]))
        assert not judge("regex", t["expected"], {}, sneaky).passed

    for t in out_scope:
        assert judge("contains", t["expected"], {}, AgentResult(output=agent_output(REFUSAL, []))).passed
        assert not judge("contains", t["expected"], {}, AgentResult(output=agent_output("an answer", ["a.md"]))).passed


def test_llm_calibration_dataset_is_well_formed():
    data = json.loads((EXAMPLES / "docs_rag_llm.json").read_text())
    tasks = data["tasks"]
    assert len(tasks) >= 40  # enough to reach the 30-label minimum for calibration
    assert all(t["judge"] == "llm" and t["judge_config"]["rubric"] for t in tasks)
    assert len({t["input"] for t in tasks}) == len(tasks)  # no duplicates across the source sets
    assert sum("say it does not know" in t["expected"] for t in tasks) >= 10  # both classes are present
