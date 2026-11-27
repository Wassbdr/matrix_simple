from pathlib import Path

import pytest

from rca_pipeline.bench import group_split, score
from rca_pipeline.bench_data import Case
from rca_pipeline.bench_reference import selected


def test_group_holdout_keeps_repetitions_together():
    cases = [
        Case(Path(f"{service}_{fault}_{repeat}"), "re1", "ob", service, fault)
        for service in ("one", "two")
        for fault in ("cpu", "mem", "disk", "delay", "loss")
        for repeat in range(5)
    ]
    train, test = group_split(cases, 42)
    assert len(train) == 40 and len(test) == 10
    assert {c.group for c in train}.isdisjoint(c.group for c in test)
    assert set(train + test) == set(cases)


def test_rcaeval_avg5_and_failures():
    result = score([
        {"answer": "a", "ranks": ["a"], "status": "ok"},
        {"answer": "a", "ranks": ["b", "c", "a"], "status": "ok"},
        {"answer": "a", "ranks": ["a"], "status": "timeout"},
    ])
    assert result == pytest.approx({"ac1": 1 / 3, "ac3": 2 / 3, "avg5": 8 / 15})


def test_slow_methods_share_train_ticket_cohort():
    cases = [Case(Path(str(i)), "re2", "tt", "cart", "cpu") for i in range(15)]
    assert selected(cases, "circa", 4, 42) == selected(cases, "rcd", 4, 42)
    assert len(selected(cases, "circa", 4, 42)) == 4
    assert selected(cases, "baro", 4, 42) == cases
