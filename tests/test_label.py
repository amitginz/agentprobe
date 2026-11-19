import json
from pathlib import Path

import pytest

from agentprobe import judges
from agentprobe.cli import main
from agentprobe.db import Database, HumanLabel, Run
from agentprobe.llm_judge import Completion, LLMJudge

DATASET = str(Path(__file__).parent.parent / "examples" / "dataset_llm.json")


class Always:
    def complete(self, system, user):
        return Completion(json.dumps({"reasoning": "r", "verdict": "pass", "score": 1}), 10, 5)


@pytest.fixture
def run_id(tmp_path):
    judges.set_llm_judge(LLMJudge(Always()))
    url = f"sqlite:///{tmp_path / 'l.db'}"
    assert main(["--db", url, "run", "--dataset", DATASET, "--agent", "mock:accuracy=1.0"]) == 0
    yield url
    judges.set_llm_judge(None)


def feed(monkeypatch, answers):
    it = iter(answers)
    monkeypatch.setattr("builtins.input", lambda _="": next(it))


def labels(url):
    with Database(url).session() as s:
        return {lb.task_id: lb.passed for lb in s.query(HumanLabel)}


def test_labeling_saves_each_grade_and_hides_the_judge_verdict(run_id, monkeypatch, capsys):
    feed(monkeypatch, ["p", "f", "s"])
    assert main(["--db", run_id, "label", "--run", "1", "--n", "3"]) == 0
    out = capsys.readouterr().out
    assert "QUESTION" in out and "ANSWER" in out and "RUBRIC" in out
    assert "verdict" not in out.lower() and "judge" not in out.lower().replace("llm judge", "")
    got = labels(run_id)
    assert sorted(got.values()) == [False, True]  # p and f stored, s skipped
    assert "graded 2 this session" in out


def test_quit_and_end_of_input_keep_what_was_graded(run_id, monkeypatch):
    feed(monkeypatch, ["p", "p", "q"])
    main(["--db", run_id, "label", "--run", "1", "--n", "3"])
    assert len(labels(run_id)) == 2

    def eof(_=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    main(["--db", run_id, "label", "--run", "1", "--n", "3"])  # Ctrl-D right away: no crash, no loss
    assert len(labels(run_id)) == 2


def test_invalid_keys_are_reprompted_and_graded_items_are_not_offered_again(run_id, monkeypatch, capsys):
    feed(monkeypatch, ["x", "?", "p", "q"])
    main(["--db", run_id, "label", "--run", "1", "--n", "5"])
    assert capsys.readouterr().out.count("type p, f, s or q") == 2
    first = set(labels(run_id))
    feed(monkeypatch, ["p", "q"])
    main(["--db", run_id, "label", "--run", "1", "--n", "5"])
    assert len(labels(run_id)) == 2 and first < set(labels(run_id))  # a different task this time


def test_unknown_run_and_nothing_to_label(tmp_path, capsys):
    url = f"sqlite:///{tmp_path / 'e.db'}"
    assert main(["--db", url, "label", "--run", "9"]) == 2
    Database(url)
    with Database(url).session() as s:
        from agentprobe.db import Dataset, Task

        ds = Dataset(name="d", tasks=[Task(input="q", expected="a", judge="contains")])
        s.add(ds)
        s.commit()
        s.add(Run(dataset_id=ds.id, agent="mock:accuracy=1.0"))
        s.commit()
    assert main(["--db", url, "label", "--run", "1"]) == 0
    assert "nothing left to label" in capsys.readouterr().out
