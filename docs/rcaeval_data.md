# Public RCAEval data

`make data` downloads the authors' [RCAEval v2 Figshare release](https://doi.org/10.6084/m9.figshare.31048672.v2), published under the MIT licence. Cite Pham, Ha and Zhang, [RCAEval: A Benchmark for Root Cause Analysis of Microservice Systems with Telemetry Data](https://arxiv.org/abs/2412.17015), WWW 2025. The original [Zenodo deposit](https://doi.org/10.5281/zenodo.14590730) lists CC BY 4.0 in its DataCite metadata; this downloader does not use that deposit.

The archive takes about 4.5 GB to download and 38 GB to extract. The downloader checks the expected SHA-256 and Figshare MD5 before extracting. It writes to `~/data/rcaeval/figshare/data` by default; use `make data RCA_DATA=/path/outside/the/clone` to choose a different location.

If Figshare is unreachable, install `huggingface-hub` and `pyarrow` in the benchmark environment and repeat the command. The fallback fetches the authors' [MIT-licensed Parquet conversion](https://huggingface.co/datasets/phamquiluan/RCAEval) at a pinned revision and verifies the case index. To choose this smaller download directly, run:

```sh
python scripts/download_rcaeval.py --source hf --suites re1 re2
```

Keep the data outside Git. Each directory name encodes the system, injected service, fault and repetition; use it to evaluate, not as a model input. The benchmark reads `metrics.json` (or `metrics.parquet`) and `inject_time.txt`. It uses service metrics only, even on cases with logs or traces.
