import json

from rca_pipeline.bench_data import candidates, cases_in, read_window


def test_flat_case_uses_injection_and_services_from_telemetry(tmp_path):
    case = tmp_path / "re2ss_carts_cpu_1"
    case.mkdir()
    (case / "inject_time.txt").write_text("12\n")
    (case / "metrics.json").write_text(json.dumps({
        "carts_cpu": [[13, 5], [10, 1], [11, 2], [12, 4], [14, 7]],
        "carts-db_mem": [[10, 3], [14, 6]],
        "orders_cpu": [[10, 0], [14, 0]],
        "orders_latency-50": [[10, 5]],
    }))
    (item,) = cases_in(tmp_path)
    frame, cut = read_window(item, 2)
    assert item.group == ("carts", "cpu")
    assert item.system == "ss"
    assert cut == 12
    assert frame.time.to_list() == [10, 11, 12, 13]
    assert frame["carts_cpu"].to_list() == [1, 2, 4, 5]
    assert frame["carts-db_mem"].to_list() == [3, 3, 3, 3]
    assert "orders_latency-50" not in frame
    assert candidates(frame) == ("carts", "orders")
