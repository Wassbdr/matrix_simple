"""Isolated RCAEval runner; write one restartable cache entry per case and method."""

import argparse
import contextlib
import hashlib
import io
import json
import os
import random
import signal
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from rca_pipeline.bench_data import Case, candidates, cases_in, read_window, service_name

METHODS = ("baro", "nsigma", "e_diagnosis", "circa", "rcd")


def cache_file(
    cache: Path, case: Case, method: str, rows: int, timeout: int, seed: int,
) -> Path:
    path = case.path / "metrics.json"
    if not path.is_file():
        path = case.path / "metrics.parquet"
    inputs = (path, case.path / "inject_time.txt")
    stamp = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in inputs]
    key = hashlib.sha256(repr((stamp, rows, timeout, seed, "RCAEval:3c3e25f", 1)).encode())
    return cache / method / f"{case.id}-{key.hexdigest()[:16]}.json"


def _timeout(*_args: object) -> None:
    raise TimeoutError("reference exceeded the per-case timeout")


def compute(job: tuple[Case, str, int, int, int, Path]) -> dict:
    case, method, rows, timeout, seed, path = job
    if path.is_file():
        return json.loads(path.read_text())
    from RCAEval import e2e

    frame, cut = read_window(case, rows)
    allowed = set(candidates(frame))
    random.seed(seed)
    np.random.seed(seed)
    signal.signal(signal.SIGALRM, _timeout)
    log = io.StringIO()
    start = time.perf_counter()
    result: dict = {"ranks": [], "status": "ok"}
    try:
        signal.alarm(timeout)
        with contextlib.redirect_stderr(log), contextlib.redirect_stdout(log):
            run = getattr(e2e, method)
            options = {"seed": seed} if method == "rcd" else {}
            raw = run(frame, cut, dataset=f"{case.suite}-{case.system}", **options)["ranks"]
        if "returning the unranked column order" in log.getvalue():
            result.update(status="failed", error=log.getvalue().strip()[-400:])
            if "TimeoutError" in result["error"]:
                result["status"] = "timeout"
        else:
            names = (service_name(name) for name in raw)
            result["ranks"] = list(dict.fromkeys(name for name in names if name in allowed))
    except Exception as exc:
        result.update(status="timeout" if isinstance(exc, TimeoutError) else "failed", error=str(exc))
    finally:
        signal.alarm(0)
    result["latency_ms"] = round(1e3 * (time.perf_counter() - start), 4)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(result) + "\n")
    temp.replace(path)
    return result


def selected(
    cases: list[Case], method: str, tt_limit: int, seed: int,
) -> list[Case]:
    if method not in {"circa", "rcd"} or not tt_limit:
        return cases
    out = [case for case in cases if case.system != "tt"]
    for suite in sorted({case.suite for case in cases}):
        pool = sorted((case for case in cases if case.system == "tt" and case.suite == suite),
                      key=lambda case: case.id)
        out.extend(random.Random(seed).sample(pool, min(tt_limit, len(pool))))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, required=True)
    parser.add_argument("--rows", type=int, default=300)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--tt-limit", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--suites", nargs="+", choices=("re1", "re2"), default=["re1", "re2"])
    args = parser.parse_args()
    if min(args.rows, args.timeout, args.workers) <= 0 or args.tt_limit < 0:
        parser.error("Rows, timeout and workers must be positive; TT limit cannot be negative.")
    cases = [case for case in cases_in(args.data_root) if case.suite in args.suites]
    jobs = []
    for method in args.methods:
        for case in selected(cases, method, args.tt_limit, args.seed):
            path = cache_file(args.cache, case, method, args.rows, args.timeout, args.seed)
            if not path.is_file():
                jobs.append((case, method, args.rows, args.timeout, args.seed, path))
    print(f"{', '.join(args.methods)}: {len(jobs)} uncached cases", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        counts: dict[str, int] = {}
        for i, result in enumerate(pool.map(compute, jobs), start=1):
            status = result["status"]
            counts[status] = counts.get(status, 0) + 1
            if i % 25 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} {counts}", flush=True)


if __name__ == "__main__":
    main()
