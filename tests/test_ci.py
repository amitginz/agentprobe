from pathlib import Path

import pytest
import yaml

from agentprobe.ci import render_markdown, run_ci
from agentprobe.cli import main
from agentprobe.db import Database

DATASET = str(Path(__file__).parent.parent / "examples" / "dataset.json")
ROOT = Path(__file__).parent.parent
GOOD, BAD, SAME = "mock:accuracy=0.95,seed=1", "mock:accuracy=0.5,seed=1", "mock:accuracy=0.95,seed=1"


@pytest.fixture
def gh(tmp_path, monkeypatch):
    summary, output = tmp_path / "summary.md", tmp_path / "output.txt"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    return summary, output


def ci(tmp_path, baseline, candidate, *extra):
    return main(["--db", f"sqlite:///{tmp_path / 'ci.db'}", "ci", "--dataset", DATASET,
                 "--baseline-agent", baseline, "--candidate-agent", candidate, *extra])


def parse_outputs(path):
    return dict(line.split("=", 1) for line in path.read_text().splitlines())


def test_regression_fails_and_reports(tmp_path, gh, capsys):
    summary, output = gh
    assert ci(tmp_path, GOOD, BAD) == 1
    out = parse_outputs(output)
    assert out["verdict"] == "regression" and int(out["broken"]) > 0 and float(out["p-value"]) < 0.05
    assert float(out["candidate-pass-rate"]) < float(out["baseline-pass-rate"])
    md = summary.read_text()
    assert "Regression detected" in md and "Tasks the candidate broke" in md and "McNemar" in md


def test_no_fail_flag_reports_but_exits_zero(tmp_path, gh):
    assert ci(tmp_path, GOOD, BAD, "--no-fail") == 0
    assert parse_outputs(gh[1])["verdict"] == "regression"


def test_identical_agents_pass(tmp_path, gh):
    assert ci(tmp_path, GOOD, SAME) == 0
    out = parse_outputs(gh[1])
    assert out["verdict"] == "no_significant_change" and out["broken"] == "0" and out["fixed"] == "0"


def test_improvement_passes(tmp_path, gh):
    assert ci(tmp_path, BAD, GOOD) == 0
    assert parse_outputs(gh[1])["verdict"] == "improvement"


def test_dead_endpoint_is_an_infrastructure_error_not_a_verdict(tmp_path, gh):
    dead = "http:http://127.0.0.1:1/agent"
    assert ci(tmp_path, GOOD, dead) == 2
    assert parse_outputs(gh[1])["verdict"] == "infrastructure_error"
    md = gh[0].read_text()
    assert "No verdict" in md and "not** a statement about the change" in md


def test_no_fail_does_not_hide_infrastructure_errors(tmp_path, gh):
    assert ci(tmp_path, "http:http://127.0.0.1:1/agent", GOOD, "--no-fail") == 2


def test_works_without_github_env(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    assert ci(tmp_path, GOOD, SAME) == 0


def test_report_escapes_untrusted_agent_output(tmp_path):
    outcome = run_ci(Database(f"sqlite:///{tmp_path / 'x.db'}"), DATASET, GOOD, BAD)
    outcome.broken[0].update(candidate="<script>alert(1)</script> | `x`\nnewline", baseline="ok")
    md = render_markdown(outcome)
    assert "<script>" not in md and "&lt;script&gt;" in md
    row = next(line for line in md.splitlines() if "&lt;script&gt;" in line)
    assert row.count("|") - row.count("\\|") == 4  # a pipe in the output cannot add table columns


def test_action_metadata_is_valid_and_complete():
    action = yaml.safe_load((ROOT / "action.yml").read_text())
    assert action["runs"]["using"] == "composite"
    inputs = action["inputs"]
    assert inputs["dataset"]["required"] and inputs["baseline-agent"]["required"] and inputs["candidate-agent"]["required"]
    step = next(s for s in action["runs"]["steps"] if s.get("id") == "gate")
    # every input must reach the script through env, never interpolated into the shell text
    assert "${{" not in step["run"]
    assert {"AP_DATASET", "AP_BASELINE", "AP_CANDIDATE"} <= set(step["env"])
    # each declared output must read from the gate step
    for name, spec in action["outputs"].items():
        assert f"steps.gate.outputs.{name}" in spec["value"]
