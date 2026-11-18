from collections import Counter
from datetime import datetime

import pytest

from crossverify.live_suite import build_live_evaluation_suite
from scripts.build_fresh_holdout import build
from scripts import run_fresh_holdout as fresh


def test_frozen_problem_families_and_integer_keys():
    bank = fresh.read_json(fresh.BANK)
    assert build() == bank
    assert fresh.stable_hash(bank) == fresh.BANK_HASH
    assert len({t["id"] for t in bank}) == len(bank) == 24
    assert set(Counter(t["family"] for t in bank).values()) == {2}
    assert set(Counter(t["domain"] for t in bank).values()) == {8}
    old_prompts = {t.prompt for t in build_live_evaluation_suite()}
    assert not old_prompts.intersection(t["prompt"] for t in bank)
    assert not fresh.is_right([True], [1])
    assert not fresh.is_right([1.0], [1])
    assert fresh.verdict({"reason": "no verdict"}) == "INVALID"


@pytest.mark.parametrize("higher", [False, True])
def test_retained_calls_replay_without_provider_or_writes(monkeypatch, higher):
    directory = fresh.OUT / "higher_effort" if higher else fresh.OUT
    snapshots = []
    monkeypatch.setattr(fresh, "save", lambda path, value: snapshots.append((path, value)))
    result = fresh.run(collect=False, high_effort=higher)
    assert result == fresh.read_json(directory / "paired_results.json")
    assert snapshots == [(directory / "paired_results.json", result)]
    protocol = fresh.read_json(directory / "protocol.json")
    frozen = datetime.fromisoformat(protocol["frozen_utc"])
    for trace in directory.glob("*-*.json"):
        record = fresh.read_json(trace)
        assert datetime.fromisoformat(record["collected_utc"]) >= frozen
        assert record["tool_events"] == 0
    n = result["counts"]
    assert n["initial_correct"] + n["fixed"] - n["broken"] == n["final_correct"]
    assert sum(d["n"] for d in result["by_domain"].values()) == 24
