import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rca_pipeline import bench_reference
from rca_pipeline.bench import group_split, score, summarize
from rca_pipeline.bench_data import Case
from rca_pipeline.bench_reference import selected
from rca_pipeline.bench_sta import METRICS, STEPS, prepare


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


def test_circa_fallback_is_a_cached_timeout(tmp_path, monkeypatch):
    def run(*args, **kwargs):
        print("circa failed (TimeoutError: case exceeded); returning the unranked column order")
        return {"ranks": ["cart_cpu"]}

    monkeypatch.setitem(sys.modules, "RCAEval", SimpleNamespace(e2e=SimpleNamespace(circa=run)))
    monkeypatch.setattr(bench_reference, "read_window", lambda *_: (None, 0))
    monkeypatch.setattr(bench_reference, "candidates", lambda _: ("cart",))
    case = Case(tmp_path, "re1", "tt", "cart", "cpu")
    job = (case, "circa", 300, 90, 42, tmp_path / "prediction.json")
    result = bench_reference.compute(job)
    assert result["status"] == "timeout"
    assert result["ranks"] == []
    monkeypatch.setattr(bench_reference, "read_window", lambda *_: pytest.fail("cached prediction expected"))
    assert bench_reference.compute(job) == result


def test_report_marks_fixed_train_ticket_sample_and_pending_references():
    metadata = {"system": "tt", "suite": "re1", "seed": 42, "answer": "cart", "latency_ms": 2}
    run = {
        "variant": "relative", "command": "bench", "cpu": "CPU", "seeds": [42],
        "rows": 300, "tt_limit": 12,
        "cases": [
            {**metadata, "method": "sta", "cohorts": ["all", "fixed TT sample"],
             "status": "ok", "ranks": ["cart"]},
            {**metadata, "method": "circa", "cohorts": ["fixed TT sample"],
             "status": "pending", "ranks": []},
        ],
    }
    report = summarize([run], run)
    assert "seed 42" in report
    assert "| STA | all | 1.000" in report
    assert "| CIRCA | fixed TT sample | pending" in report


def test_sta_tail_preserves_short_downward_deviation(tmp_path):
    before = np.resize([-1, 0, 1], 36)
    after = np.zeros(STEPS * 5)
    after[1] = -8
    frame = pd.DataFrame({"time": np.arange(len(before) + len(after)),
                          "cart_cpu": np.concatenate([before, after])})
    case = Case(tmp_path, "re1", "tt", "cart", "cpu")
    sample = prepare(case, frame, len(before), "tails")
    assert sample.names == ("cart",)
    assert sample.x.shape == (1, 2 * len(METRICS), STEPS)
    assert sample.x[0, len(METRICS), 0] < sample.x[0, 0, 0] < 0
    assert sample.x[0, len(METRICS), 1] == 0
