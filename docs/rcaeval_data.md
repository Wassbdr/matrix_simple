# Public RCAEval data

`make data` downloads the authors' [RCAEval v2 Figshare release](https://doi.org/10.6084/m9.figshare.31048672.v2), published under the MIT licence. Cite Pham, Ha and Zhang, [RCAEval: A Benchmark for Root Cause Analysis of Microservice Systems with Telemetry Data](https://arxiv.org/abs/2412.17015), WWW 2025. The original [Zenodo deposit](https://doi.org/10.5281/zenodo.14590730) lists CC BY 4.0 in its DataCite metadata; this downloader does not use that deposit.

The archive takes about 4.5 GB to download and 38 GB to extract. The downloader checks the expected SHA-256 and Figshare MD5 before extracting. It writes to `~/data/rcaeval/figshare/data` by default; use `make data RCA_DATA=/path/outside/the/clone` to choose a different location.

If Figshare is unreachable, install `huggingface-hub` and `pyarrow` in the benchmark environment and repeat the command. The fallback fetches the authors' [MIT-licensed Parquet conversion](https://huggingface.co/datasets/phamquiluan/RCAEval) at a pinned revision and verifies the case index. To choose this smaller download directly, run:

```sh
python scripts/download_rcaeval.py --source hf --suites re1 re2
```

Keep the data outside Git. Each directory name encodes the system, injected service, fault and repetition; use it to evaluate, not as a model input. The benchmark reads `metrics.json` (or `metrics.parquet`) and `inject_time.txt`. It uses service metrics only, even on cases with logs or traces.

## Reference environments

Install the reference methods separately from STA. These commands assume the upstream clone is a sibling of this checkout:

```sh
git clone https://github.com/phamquiluan/RCAEval ../RCAEval
git -C ../RCAEval checkout 3c3e25f3f50a511beb777117b63ead4acb8fdf5a
python3.12 -m venv ~/.venv-rcaeval
~/.venv-rcaeval/bin/pip install -e '../RCAEval[core]'
python3.10 -m venv ~/.venv-rcaeval-rcd
~/.venv-rcaeval-rcd/bin/pip install -e ../RCAEval numpy==1.23.5 pandas==2.0.1 \
    scipy==1.10.1 scikit-learn==1.2.2 networkx==2.5 causal-learn==0.1.2.3 \
    matplotlib==3.7.2 tqdm requests
for file in utils/PCUtils/SkeletonDiscovery.py graph/GraphClass.py utils/Fas.py search/ConstraintBased/FCI.py; do
    ln -fs "$(realpath ../RCAEval/lib/causallearn)/$file" \
        "$HOME/.venv-rcaeval-rcd/lib/python3.10/site-packages/causallearn/$file"
done
```

Run `make bench` in the STA environment. Outputs are `benchmarks/results.md` and `benchmarks/prior.json`; reference predictions are cached outside the checkout. The default adds an all-metric change prior to the fitted median and IQR design. We tested the fixed prior on RE1 development with seeds 7, 42 and 123. No system lost mean AC@1 or Avg@5, so we froze it before scoring RE2 with the same seeds. To reproduce the experiments in order:

```sh
for variant in baseline relative onset tails prior; do
    make bench BENCH_ARGS="--split dev --variant $variant"
done
make bench BENCH_ARGS="--variant relative --split all"
make bench
```

Slow CIRCA and RCD runs use the same seed-42 Train Ticket sample. Compare all methods on the fixed TT sample rows; the faster methods also report on the full cohort. Timeouts count as misses, not column-order predictions.
