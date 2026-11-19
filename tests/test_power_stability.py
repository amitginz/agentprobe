import random
import subprocess
import sys

import pytest

from agentprobe.agents import MockAgent, build_agent
from agentprobe.compare import mcnemar_exact
from agentprobe.db import TaskResult
from agentprobe.power import (
    critical_count,
    min_detectable_effect,
    power,
    power_for_effect,
    required_tasks,
)
from agentprobe.stability import analyze

# --- power ----------------------------------------------------------------------------

def test_critical_count_agrees_with_the_test_itself():
    for d in range(1, 60):
        k = critical_count(d, 0.05)
        if k >= 0:
            assert mcnemar_exact(k, d - k) < 0.05  # min(b, c) <= k rejects...
        assert mcnemar_exact(k + 1, d - k - 1) >= 0.05  # ...and k + 1 does not


def test_power_matches_a_monte_carlo_simulation_of_the_actual_test():
    n, p_b, p_f, alpha = 60, 0.04, 0.16, 0.05
    rng, hits, trials = random.Random(7), 0, 6000
    for _ in range(trials):
        b = c = 0
        for _ in range(n):
            u = rng.random()
            if u < p_b:
                b += 1
            elif u < p_b + p_f:
                c += 1
        hits += mcnemar_exact(b, c) < alpha
    assert power(n, p_b, p_f, alpha) == pytest.approx(hits / trials, abs=0.02)


def test_type_one_error_is_at_most_alpha_when_there_is_no_effect():
    for n in (10, 25, 80, 200):
        assert power(n, 0.1, 0.1, 0.05) <= 0.05 + 1e-9


def test_power_grows_with_size_and_effect():
    assert power_for_effect(50, 0.2, 0.1) < power_for_effect(200, 0.2, 0.1)
    assert power_for_effect(100, 0.2, 0.05) < power_for_effect(100, 0.2, 0.15)
    assert power_for_effect(100, 0.2, 0.1) == pytest.approx(power_for_effect(100, 0.2, -0.1))  # two-sided


def test_tiny_datasets_cannot_reach_significance_at_all():
    # Even if every flip goes the same way, 5 flipped tasks give exact two-sided p = 2/32 = 0.0625,
    # and 5 is what 20% of 25 tasks yields on average, so power stays far below 80%.
    assert power_for_effect(25, 0.2, 0.2) < 0.5
    assert min_detectable_effect(25, 0.2) is None
    assert critical_count(5, 0.05) == -1  # five discordant tasks can never be significant


def test_min_detectable_effect_and_required_tasks_are_consistent():
    mde = min_detectable_effect(200, 0.15)
    assert 0.05 < mde < 0.11
    assert power_for_effect(200, 0.15, mde + 0.005) >= 0.8
    n = required_tasks(0.08, 0.15)
    assert power_for_effect(n, 0.15, 0.08) >= 0.8
    assert min_detectable_effect(n // 2, 0.15) is None or min_detectable_effect(n // 2, 0.15) > 0.08
    with pytest.raises(ValueError):
        required_tasks(0.5, 0.1)


def test_cli_power_commands(capsys):
    from agentprobe.cli import main

    assert main(["power", "--tasks", "25", "--discordance", "0.08"]) == 0
    assert "cannot reach" in capsys.readouterr().out
    assert main(["power", "--tasks", "200", "--discordance", "0.15"]) == 0
    assert "detected with 80% power" in capsys.readouterr().out
    assert main(["power", "--effect", "0.08"]) == 0
    assert "need about" in capsys.readouterr().out
    assert main(["power"]) == 2


# --- flaky mock agent -------------------------------------------------------------------

def test_deterministic_mock_stays_deterministic_and_jitter_does_not():
    plain = MockAgent(accuracy=0.7, seed=1)
    assert [plain.run(f"q{i}", "x").output for i in range(200)] == [plain.run(f"q{i}", "x").output for i in range(200)]
    noisy = MockAgent(accuracy=0.7, seed=1, jitter=0.5, rng_seed=3)
    first = [noisy.run(f"q{i}", "x").output for i in range(300)]
    second = [noisy.run(f"q{i}", "x").output for i in range(300)]
    assert first != second
    assert build_agent("mock:accuracy=0.7,jitter=0.4,rng=9").jitter == 0.4
    with pytest.raises(ValueError):
        MockAgent(jitter=1.5)


# --- stability --------------------------------------------------------------------------

def result(task_id, passed):
    return TaskResult(task_id=task_id, passed=passed)


def test_identical_repeats_have_no_noise():
    runs = [[result(i, i % 2 == 0) for i in range(10)] for _ in range(4)]
    s = analyze(runs)
    assert s.flaky == 0 and s.pass_rate_sd == 0.0 and s.noise_floor_pts == 0.0
    assert s.always_pass == 5 and s.always_fail == 5


def test_flaky_tasks_are_found_and_ordered_by_instability():
    k = 4
    patterns = {0: [1, 1, 1, 1], 1: [0, 0, 0, 0], 2: [1, 0, 1, 0], 3: [1, 1, 1, 0]}  # task -> passes per repeat
    runs = [[result(t, bool(p[r])) for t, p in patterns.items()] for r in range(k)]
    s = analyze(runs)
    assert s.flaky == 2 and s.always_pass == 1 and s.always_fail == 1
    assert [t["task_id"] for t in s.flaky_tasks] == [2, 3]  # the coin flip before the mostly-passing task
    assert s.flaky_tasks[0]["passes"] == 2 and s.pass_rate_sd > 0


def test_stability_needs_two_repeats_and_shared_tasks():
    with pytest.raises(ValueError):
        analyze([[result(1, True)]])
    with pytest.raises(ValueError):
        analyze([[result(1, True)], [result(2, True)]])


def test_cli_stability_end_to_end(tmp_path):
    base = [sys.executable, "-m", "agentprobe.cli", "--db", f"sqlite:///{tmp_path / 's.db'}", "stability",
            "--dataset", "examples/dataset.json", "--repeats", "6"]
    steady = subprocess.run([*base, "--agent", "mock:accuracy=0.8,seed=2"], capture_output=True, text=True, check=False)
    assert steady.returncode == 0 and "0/40 tasks flaky" in steady.stdout
    shaky = subprocess.run([*base, "--agent", "mock:accuracy=0.8,seed=2,jitter=0.6", "--max-flaky", "0.05"],
                           capture_output=True, text=True, check=False)
    assert shaky.returncode == 1  # more than 5% of tasks are flaky
    assert "indistinguishable from re-running" in shaky.stdout
