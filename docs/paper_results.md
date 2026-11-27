# Frozen paper results

These are historical, not results of the [current public benchmark](../benchmarks/results.md). The required input snapshots for the old runs are not public as run: the exact RCAEval split and preprocessing inputs were not published with two of the original result JSONs, and the ObservIT and registry runs used private Devoteam cluster telemetry. Public RCAEval is available, but its current benchmark uses a different protocol. The synthetic comparison below is also frozen: newer runs do not reproduce all of its original cells.

## RCAEval metrics only

Five seeds, mean and standard deviation from the historical paper table.

| Dataset | STA Avg@1 | STA MRR | GATv2 Avg@1 | GATv2 MRR |
|---|---:|---:|---:|---:|
| Train Ticket | 96.0% ± 3.7% | 98.0% ± 1.9% | 95.0% ± 4.5% | 97.5% ± 2.2% |
| Sock Shop | 85.0% ± 8.9% | 90.8% ± 5.8% | 81.0% ± 8.0% | 87.6% ± 5.8% |
| Online Boutique | 65.0% ± 9.5% | 77.9% ± 5.3% | 68.0% ± 5.1% | 78.8% ± 4.2% |

Under the same historical split and candidate set, BARO Avg@1 was **0.80 / 0.84 / 0.86** on Train Ticket, Sock Shop, and Online Boutique. See the [historical comparison notes](COMPARABILITY_MATRIX.md).

## RCAEval metrics and traces

| Dataset | STA-MM Avg@1 | STA-MM MRR | GATv2-MM Avg@1 | GATv2-MM MRR |
|---|---:|---:|---:|---:|
| Train Ticket | 91.0% ± 2.0% | 94.3% ± 1.0% | 88.0% ± 5.1% | 92.4% ± 3.4% |
| Online Boutique | 65.0% ± 15.5% | 79.4% ± 8.3% | 79.0% ± 8.6% | 87.6% ± 4.5% |

Sock Shop had no traces in the input release and was skipped.

## Synthetic and private-cluster results

- Synthetic Barabási-Albert graphs at N=100: STA led GATv2 by **2.3 percentage points Avg@1** across ten seeds, down from 6.2 points in the five-seed run.
- ObservIT Sock Shop: STA metrics-only Avg@1 was 79.6% ± 4.3%, against 76.2% ± 3.4% for GATv2. STA-MM with Loki logs scored 77.2% ± 2.1%.
- The historical registry model was recorded at **Avg@1 0.786** and MRR 0.871. It is not the RE1 checkpoint shipped with the CLI.

See the [historical artifact inventory](../results_transmission/historical/) for provenance.

## Legacy synthetic regression check

The values 0.30 / 0.183 / 0.067 at N=20 / 50 / 100 in `rca_pipeline/benchmark.py` are historical regression references, not measured results of the current run. The alert compares new top-one accuracy against the reference minus the larger of half a percentage point and two historical standard deviations. The JSON field `reference_top1` remains for readers of the old artifact format. The README uses `make bench` for public accuracy and `make scaling` for size versus latency because those commands answer different questions.
