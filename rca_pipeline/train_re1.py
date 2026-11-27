"""Build the packaged STA model using only the pinned RE1 release."""

import argparse
from pathlib import Path

import torch

from rca_pipeline.bench_data import cases_in, read_window
from rca_pipeline.bench_sta import fit, prepare

ROWS, EPOCHS, SEED = 300, 36, 42


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cases = [case for case in cases_in(args.data_root) if case.suite == "re1"]
    if len(cases) != 375:
        parser.error("Supply the complete 375-case pinned RE1 release; run rca data --suites re1.")
    samples = [prepare(case, *read_window(case, ROWS), "relative") for case in cases]
    model, _, mean, scale = fit(samples, SEED, EPOCHS, "relative")
    torch.save(
        {"state": model.state_dict(), "mean": mean, "scale": scale, "hidden_dim": model.hidden_dim,
         "k": model.k, "rows": ROWS, "version": "re1-pooled-seed42-v1"},
        args.output,
    )
    print(f"Wrote {args.output} (RE1, {len(cases)} cases, seed {SEED}).")


if __name__ == "__main__":
    main()
