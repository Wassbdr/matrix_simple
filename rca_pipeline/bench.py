"""Compare STA and RCAEval references on public service-level ground truth."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import fmean, median, pstdev

import torch

from rca_pipeline.bench_data import Case, cases_in, read_window
from rca_pipeline.bench_reference import METHODS, cache_file, selected
from rca_pipeline.bench_sta import VARIANTS, fit, prepare, rank

SYSTEMS = {"ob": "Online Boutique", "ss": "Sock Shop", "tt": "Train Ticket"}
DISPLAY = {
    "sta": "STA", "baro": "BARO", "nsigma": "Nsigma", "e_diagnosis": "epsilon-Diagnosis",
    "circa": "CIRCA", "rcd": "RCD",
}
METRICS = ("ac1", "ac3", "avg5")
ITERATION_BASE = {"relative": "baseline", "onset": "relative", "tails": "relative", "prior": "relative"}


def group_split(cases: list[Case], seed: int) -> tuple[list[Case], list[Case]]:
    by_service: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        by_service[case.service].add(case.fault)
    rng = random.Random(seed)
    test_groups = {
        (service, fault)
        for service, faults in sorted(by_service.items())
        for fault in rng.sample(sorted(faults), len(faults) // 5 or int(len(faults) > 1))
    }
    return (
        [case for case in cases if case.group not in test_groups],
        [case for case in cases if case.group in test_groups],
    )


def score(results: list[dict]) -> dict[str, float]:
    ranks = [
        result["ranks"].index(result["answer"]) + 1
        if result["status"] == "ok" and result["answer"] in result["ranks"] else float("inf")
        for result in results
    ]
    hit = [fmean(rank <= k for rank in ranks) for k in range(1, 6)]
    return {"ac1": hit[0], "ac3": hit[2], "avg5": fmean(hit)}


def references(args: argparse.Namespace, suites: list[str]) -> None:
    groups = [
        (args.reference_python, METHODS[:-1], min(5, args.workers)),
        (args.rcd_python, METHODS[-1:], max(1, args.workers - 5)),
    ]
    jobs = []
    for executable, _, _ in groups:
        if not Path(executable).is_file():
            raise ValueError(f"Reference Python not found: {executable}. See docs/rcaeval_data.md.")
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    try:
        for executable, methods, workers in groups:
            jobs.append(subprocess.Popen([
                executable, "-m", "rca_pipeline.bench_reference",
                "--data-root", str(args.data_root), "--cache", str(args.cache),
                "--rows", str(args.rows), "--timeout", str(args.timeout),
                "--tt-limit", str(args.tt_limit), "--workers", str(workers),
                "--suites", *suites, "--methods", *methods,
            ], env=env))
        if any(job.wait() for job in jobs):
            raise ValueError("A reference worker exited unsuccessfully; inspect its output and retry.")
    finally:
        for job in jobs:
            if job.poll() is None:
                job.terminate()
                job.wait()


def rows_for_report(records: list[dict], method: str, cohort: str) -> tuple:
    matching = [r for r in records if r["method"] == method and cohort in r["cohorts"]]
    if not matching:
        return ("n/a",) * 5
    failures = sum(r["status"] not in ("ok", "pending") for r in matching)
    pending = sum(r["status"] == "pending" for r in matching)
    coverage = f"{failures}/{len(matching)}" + (f"; {pending} pending" if pending else "")
    if pending:
        return ("pending", "pending", "pending", "pending", coverage)
    by_seed: dict[int, list[dict]] = defaultdict(list)
    for item in matching:
        by_seed[item["seed"]].append(item)
    values = [score(rs) for rs in by_seed.values()]
    metrics = [
        f"{fmean(v[key] for v in values):.3f} ± {pstdev(v[key] for v in values):.3f}"
        for key in METRICS
    ]
    timing = median(r["latency_ms"] for r in matching)
    return (*metrics, f"{timing:.2f}", coverage)


def train_ticket_errors(run: dict, re1_faults: set[str]) -> list[str]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in run["cases"]:
        if (row["method"], row["suite"], row["system"]) == ("sta", "re2", "tt"):
            grouped[row["fault"]].append(row)
    if not grouped:
        return []
    lines = [
        f"### `{run['variant']}`", "",
        "Counts include each case once per training seed.", "",
        "| Fault | Attempts | Top-1 hits | Miss, root outside top-k | Miss, root inside top-k |",
        "|---|---:|---:|---:|---:|",
    ]
    for fault, rows in sorted(grouped.items()):
        misses = [row for row in rows if not row["ranks"] or row["ranks"][0] != row["answer"]]
        inside = sum(row["selector_hit"] for row in misses)
        lines.append(f"| {fault} | {len(rows)} | {len(rows) - len(misses)} | {len(misses) - inside} | {inside} |")
    misses = sum(row["ranks"][:1] != [row["answer"]] for rows in grouped.values() for row in rows)
    unseen_faults = grouped.keys() - re1_faults
    if re1_faults and unseen_faults:
        unseen = [row for fault in unseen_faults for row in grouped[fault]]
        unseen_misses = sum(row["ranks"][:1] != [row["answer"]] for row in unseen)
        names = ", ".join(sorted(unseen_faults))
        lines += ["", f"Fault types absent from the RE1 cohort ({names}) account for {unseen_misses} of {misses} misses."]
    return [*lines, ""]


def summarize(outputs: list[dict], latest: dict) -> str:
    lines = [
        "# Public RCAEval benchmark", "",
        latest["command"], "",
        f"STA variant in the comparison tables: `{latest['variant']}`.", "",
        f"CPU: {latest['cpu']}; one thread per ranker. Seeds: {latest['seeds']}; "
        f"{latest['rows']} samples on each side of injection at most, metrics only. "
        "Latency is median wall time from a loaded window to a ranking, including STA feature preparation.", "",
        "RE1 uses service/fault groups; RE2 is a separate transfer set. "
        "AC@k is a top-k hit rate; Avg@5 averages AC@1 through AC@5. "
        "Failures count as misses. Repeated predictions of the same case across seeds are not independent.",
        "",
        "MicroRCA: unavailable on the common metrics input. The upstream implementation requires "
        "call-edge latency and host topology; its example also lacks the expected source latency CSV.",
        "",
    ]
    if latest["tt_limit"]:
        lines.extend([
            f"CIRCA and RCD use up to {latest['tt_limit']} Train Ticket cases per suite "
            "sampled with seed 42 because the full runs are too slow. RE1 further intersects "
            "that sample with each grouped test split. Compare them only with the other "
            "methods in the fixed TT sample rows.",
            "",
        ])
    for suite in ("re1", "re2"):
        for system, title in SYSTEMS.items():
            records = [r for r in latest["cases"] if r["system"] == system and r["suite"] == suite]
            if not records:
                continue
            lines += [
                f"## {suite.upper()} / {title}", "",
                "| Method | Cohort | AC@1 | AC@3 | Avg@5 | Median ms | Failed / scored attempts |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
            for method, cohort in [(m, "all") for m in DISPLAY] + [
                (m, "fixed TT sample") for m in DISPLAY
            ]:
                cells = rows_for_report(records, method, cohort)
                if cells[0] == "n/a":
                    continue
                lines.append(f"| {DISPLAY[method]} | {cohort} | " + " | ".join(cells) + " |")
            lines.append("")
    baseline = next((run for run in outputs if run["variant"] == "baseline"), None)
    if baseline is not None:
        blog = {(r["case"], r["seed"]): r for r in baseline["cases"]
                if (r["method"], r["suite"], r["system"]) == ("baro", "re1", "ob")}
        missed: dict[str, list[bool]] = defaultdict(list)
        for row in baseline["cases"]:
            if (row["method"], row["suite"], row["system"]) != ("sta", "re1", "ob"):
                continue
            baro = blog[row["case"], row["seed"]]
            answer = [row["answer"]]
            if baro["status"] == "ok" and baro["ranks"][:1] == answer and row["ranks"][:1] != answer:
                missed[row["fault"]].append(row["selector_hit"])
        if missed:
            lines += [
                "## Online Boutique RE1 baseline errors", "",
                "| Fault | BARO hit at 1, STA missed | Root in STA top-k |",
                "|---|---:|---:|",
            ]
            for fault, hits in sorted(missed.items()):
                lines.append(f"| {fault} | {len(hits)} | {sum(hits)} |")
            lines.append("")
    re1_faults = set(latest.get("re1_tt_faults", []))
    incumbent = next((run for run in outputs if run["variant"] == "relative"), latest)
    breakdown = train_ticket_errors(incumbent, re1_faults)
    if incumbent is not latest:
        breakdown += train_ticket_errors(latest, re1_faults)
    if breakdown:
        lines += ["## RE2 Train Ticket errors", "", *breakdown]
        metrics = latest.get("tt_metric_families", {})
        if "re2" in metrics:
            new_metrics = ", ".join(sorted(set(metrics["re2"]) - set(metrics.get("re1", []))))
            if new_metrics:
                lines += [
                    f"Train Ticket metric families present only in RE2: {new_metrics}. "
                    "The fault breakdown does not attribute a miss to a metric family. "
                    "The prior scores all available families together, so its gains do not isolate either family.",
                    "",
                ]
    by_variant = {out["variant"]: out for out in outputs}
    for i, out in enumerate(outputs):
        previous = by_variant.get(ITERATION_BASE.get(out["variant"], ""))
        lines += [
            f"## Iteration {i}: {VARIANTS[out['variant']]}", "",
            f"Delta against `{previous['variant']}`." if previous else "No earlier design measured.", "",
            "| Suite / system | AC@1 | Delta | Avg@5 | Delta |",
            "|---|---:|---:|---:|---:|",
        ]
        for suite in ("re1", "re2"):
            for system, title in SYSTEMS.items():
                def sta_metrics(run: dict) -> tuple[float, float] | None:
                    by_seed: dict[int, list[dict]] = defaultdict(list)
                    for item in run["cases"]:
                        if (item["method"], item["suite"], item["system"]) == ("sta", suite, system):
                            by_seed[item["seed"]].append(item)
                    values = [score(rs) for rs in by_seed.values()]
                    if not values:
                        return None
                    return fmean(v["ac1"] for v in values), fmean(v["avg5"] for v in values)

                current = sta_metrics(out)
                if current is None:
                    continue
                before = sta_metrics(previous) if previous else None
                a, b = current
                da, db = (f"{a - before[0]:+.3f}", f"{b - before[1]:+.3f}") if before else ("n/a", "n/a")
                lines.append(f"| {suite.upper()} / {title} | {a:.3f} | {da} | {b:.3f} | {db} |")
        lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    suites = ["re1"] if args.split == "dev" else ["re1", "re2"]
    cases = [case for case in cases_in(args.data_root) if case.suite in suites]
    if not args.skip_references:
        references(args, suites)
    torch.set_num_threads(1)
    records: list[dict] = []
    tt_metric_families: dict[str, set[str]] = defaultdict(set)
    sampled_ids = {
        case.id for case in selected(cases, "circa", args.tt_limit, 42)
        if args.tt_limit and case.system == "tt"
    }
    for system in SYSTEMS:
        development = [case for case in cases if (case.system, case.suite) == (system, "re1")]
        holdout = [case for case in cases if (case.system, case.suite) == (system, "re2")]
        if not development:
            continue
        prepared = {}
        for case in development + holdout:
            frame, cutoff = read_window(case, args.rows)
            if system == "tt":
                tt_metric_families[case.suite].update(col.split("_", 1)[1] for col in frame.columns if "_" in col)
            prepared[case.id] = prepare(case, frame, cutoff, args.variant)
        for seed in args.seeds:
            train, dev_test = group_split(development, seed)
            fitted = fit([prepared[c.id] for c in train], seed, args.epochs, args.variant)
            print(f"{system} seed={seed}: train={len(train)}, RE1 test={len(dev_test)}, RE2={len(holdout)}",
                  flush=True)
            rank(fitted, prepared[train[0].id])
            for case in dev_test + holdout:
                extra = ["fixed TT sample"] if case.id in sampled_ids else []
                meta = {"case": case.id, "suite": case.suite, "system": case.system,
                        "answer": case.service, "fault": case.fault, "seed": seed}
                records.append({**meta, "method": "sta", "cohorts": ["all", *extra],
                                **rank(fitted, prepared[case.id])})
                for method in METHODS:
                    restrict = case.system == "tt" and args.tt_limit and method in ("circa", "rcd")
                    if restrict and not extra:
                        continue
                    path = cache_file(args.cache, case, method, args.rows, args.timeout, 42)
                    value = json.loads(path.read_text()) if path.is_file() else {
                        "status": "pending", "ranks": [], "latency_ms": 0.0,
                    }
                    records.append({**meta, "method": method, "cohorts": extra if restrict else ["all", *extra],
                                    **value})
    if not records:
        raise ValueError("No scored cases. Supply the complete RE1 release for training and development.")
    args.output.mkdir(parents=True, exist_ok=True)
    cpu = platform.processor()
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            cpu = line.split(":", 1)[1].strip()
            break
    path = args.output / f"{args.variant}.json"
    seeded = ",".join(map(str, args.seeds))
    output = {
        "cpu": cpu, "seeds": args.seeds, "rows": args.rows, "epochs": args.epochs,
        "tt_limit": args.tt_limit, "timeout": args.timeout, "variant": args.variant,
        "re1_tt_faults": sorted({c.fault for c in cases if (c.suite, c.system) == ("re1", "tt")}),
        "tt_metric_families": {suite: sorted(families) for suite, families in tt_metric_families.items()},
        "command": f"`make bench BENCH_ARGS='--variant {args.variant} --split {args.split} "
                   f"--seeds {seeded} --epochs {args.epochs} --rows {args.rows} "
                   f"--tt-limit {args.tt_limit} --timeout {args.timeout}'`",
        "cases": records,
    }
    path.write_text(json.dumps(output, indent=2) + "\n")
    outputs = [
        json.loads(p.read_text()) for variant in VARIANTS
        if (p := args.output / f"{variant}.json").is_file()
    ]
    matching = [
        prior for prior in outputs
        if all(prior[key] == output[key] for key in ("seeds", "rows", "epochs", "tt_limit", "timeout"))
    ]
    (args.output / "results.md").write_text(summarize(matching, output))
    print(f"Wrote {path} and {args.output / 'results.md'}")


def main() -> None:
    root = Path.home() / "data" / "rcaeval"
    default_data = root / "figshare" / "data"
    if not default_data.exists() and (root / "hf").exists():
        default_data = root / "hf"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_data)
    parser.add_argument("--cache", type=Path, default=root / "reference-cache")
    parser.add_argument("--output", type=Path, default=Path("benchmarks"))
    parser.add_argument("--reference-python", default=str(Path.home() / ".venv-rcaeval/bin/python"))
    parser.add_argument("--rcd-python", default=str(Path.home() / ".venv-rcaeval-rcd/bin/python"))
    parser.add_argument("--seeds", type=lambda value: [int(seed) for seed in value.split(",")],
                        default=[7, 42, 123])
    parser.add_argument("--rows", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--tt-limit", type=int, default=12)
    parser.add_argument("--split", choices=("dev", "all"), default="all")
    parser.add_argument("--variant", choices=VARIANTS, default="prior")
    parser.add_argument("--skip-references", action="store_true", help="Use cached results only.")
    args = parser.parse_args()
    if min(args.rows, args.epochs, args.workers, args.timeout) <= 0 or args.tt_limit < 0:
        parser.error("Rows, epochs, workers and timeout must be positive; TT limit cannot be negative.")
    try:
        run(args)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
