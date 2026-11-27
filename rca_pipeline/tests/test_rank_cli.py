import json
import sys

import pandas as pd
import pytest

from rca_pipeline import rank
from rca_pipeline.cli import main
from rca_pipeline.data import DEMO_CASE, DEMO_FILES, demo_case


@pytest.fixture()
def metrics(tmp_path):
    case = tmp_path / "my_incident"
    case.mkdir()
    frame = pd.DataFrame({"time": range(20), "cart_cpu": [1] * 10 + [7] * 10, "order_cpu": [1] * 20})
    (case / "metrics.json").write_text(json.dumps({
        name: list(zip(frame["time"], frame[name])) for name in ("cart_cpu", "order_cpu")
    }))
    (case / "inject_time.txt").write_text("10\n")
    csv = tmp_path / "export.csv"
    frame.to_csv(csv, index=False)
    return case, csv, frame


def test_input_formats_use_the_same_model(metrics):
    case, csv, frame = metrics
    result = rank(case, top_k=2)
    assert result == rank(case / "metrics.json")
    assert result == rank(csv, inject_time=10)
    assert result == rank(frame, inject_time=10)
    assert {row.service for row in result} == {"cart", "order"}
    assert [row.rank for row in result] == [1, 2]
    assert result[0].score >= result[1].score


def test_cli_json_and_actionable_error(metrics, capsys, monkeypatch):
    case, csv, _ = metrics
    monkeypatch.setattr(sys, "argv", ["rca", "rank", str(case), "--top-k", "1", "--json"])
    main()
    candidates = json.loads(capsys.readouterr().out)
    assert len(candidates) == 1
    assert candidates[0]["service"] == rank(case, top_k=1)[0].service
    monkeypatch.setattr(sys, "argv", ["rca", "rank", str(csv)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "Set inject_time" in capsys.readouterr().err


@pytest.mark.parametrize("command,option", [("data", "--suites"), ("bench", "--split")])
def test_cli_subcommand_help(command, option, capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["rca", command, "--help"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    assert option in capsys.readouterr().out


def test_demo_case_rejects_unverified_cache(tmp_path):
    cache = tmp_path / "demo" / DEMO_CASE
    cache.mkdir(parents=True)
    for name in DEMO_FILES:
        (cache / name).write_text("invalid")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        demo_case(tmp_path)
