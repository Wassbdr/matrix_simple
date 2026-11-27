# Public RCAEval benchmark

`make bench BENCH_ARGS='--variant prior --split all --seeds 7,42,123 --epochs 36 --rows 300 --tt-limit 12 --timeout 90'`

STA variant in the comparison tables: `prior`.

CPU: Intel(R) Xeon(R) Processor; one thread per ranker. Seeds: [7, 42, 123]; 300 samples on each side of injection at most, metrics only. Latency is median wall time from a loaded window to a ranking, including STA feature preparation.

RE1 uses service/fault groups; RE2 is a separate transfer set. AC@k is a top-k hit rate; Avg@5 averages AC@1 through AC@5. Failures count as misses. Repeated predictions of the same case across seeds are not independent.

MicroRCA: unavailable on the common metrics input. The upstream implementation requires call-edge latency and host topology; its example also lacks the expected source latency CSV.

CIRCA and RCD use up to 12 Train Ticket cases per suite sampled with seed 42 because the full runs are too slow. RE1 further intersects that sample with each grouped test split. Compare them only with the other methods in the fixed TT sample rows.

## RE1 / Online Boutique

| Method | Cohort | AC@1 | AC@3 | Avg@5 | Median ms | Failed / scored attempts |
|---|---|---:|---:|---:|---:|---:|
| STA | all | 0.947 ± 0.019 | 0.987 ± 0.019 | 0.979 ± 0.019 | 13.91 | 0/75 |
| BARO | all | 0.667 ± 0.100 | 0.933 ± 0.068 | 0.893 ± 0.051 | 44.66 | 0/75 |
| Nsigma | all | 0.640 ± 0.086 | 0.947 ± 0.050 | 0.885 ± 0.038 | 35.72 | 0/75 |
| epsilon-Diagnosis | all | 0.067 ± 0.019 | 0.200 ± 0.065 | 0.157 ± 0.049 | 1769.90 | 0/75 |
| CIRCA | all | 0.693 ± 0.082 | 0.920 ± 0.086 | 0.869 ± 0.067 | 4059.48 | 0/75 |
| RCD | all | 0.613 ± 0.075 | 0.653 ± 0.050 | 0.669 ± 0.053 | 8254.86 | 0/75 |

## RE1 / Sock Shop

| Method | Cohort | AC@1 | AC@3 | Avg@5 | Median ms | Failed / scored attempts |
|---|---|---:|---:|---:|---:|---:|
| STA | all | 0.960 ± 0.057 | 0.987 ± 0.019 | 0.984 ± 0.023 | 14.54 | 0/75 |
| BARO | all | 0.840 ± 0.057 | 0.987 ± 0.019 | 0.941 ± 0.032 | 44.62 | 0/75 |
| Nsigma | all | 0.840 ± 0.033 | 0.987 ± 0.019 | 0.947 ± 0.010 | 36.25 | 0/75 |
| epsilon-Diagnosis | all | 0.133 ± 0.019 | 0.600 ± 0.033 | 0.469 ± 0.036 | 1712.93 | 0/75 |
| CIRCA | all | 0.707 ± 0.068 | 0.960 ± 0.000 | 0.909 ± 0.026 | 3640.13 | 0/75 |
| RCD | all | 0.387 ± 0.068 | 0.560 ± 0.033 | 0.541 ± 0.042 | 4113.21 | 0/75 |

## RE1 / Train Ticket

| Method | Cohort | AC@1 | AC@3 | Avg@5 | Median ms | Failed / scored attempts |
|---|---|---:|---:|---:|---:|---:|
| STA | all | 0.840 ± 0.065 | 0.987 ± 0.019 | 0.952 ± 0.023 | 49.91 | 0/75 |
| BARO | all | 0.573 ± 0.068 | 0.933 ± 0.050 | 0.859 ± 0.033 | 166.53 | 0/75 |
| Nsigma | all | 0.507 ± 0.019 | 0.880 ± 0.033 | 0.811 ± 0.030 | 206.41 | 0/75 |
| epsilon-Diagnosis | all | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 6312.43 | 0/75 |
| STA | fixed TT sample | 0.750 ± 0.204 | 1.000 ± 0.000 | 0.933 ± 0.062 | 49.54 | 0/11 |
| BARO | fixed TT sample | 0.500 ± 0.354 | 0.833 ± 0.118 | 0.833 ± 0.118 | 162.38 | 0/11 |
| Nsigma | fixed TT sample | 0.472 ± 0.171 | 0.833 ± 0.118 | 0.778 ± 0.117 | 167.33 | 0/11 |
| epsilon-Diagnosis | fixed TT sample | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 6258.40 | 0/11 |
| CIRCA | fixed TT sample | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 90057.93 | 11/11 |
| RCD | fixed TT sample | 0.000 ± 0.000 | 0.278 ± 0.208 | 0.183 ± 0.143 | 13802.52 | 0/11 |

## RE2 / Online Boutique

| Method | Cohort | AC@1 | AC@3 | Avg@5 | Median ms | Failed / scored attempts |
|---|---|---:|---:|---:|---:|---:|
| STA | all | 0.870 ± 0.005 | 0.970 ± 0.010 | 0.952 ± 0.004 | 17.92 | 0/270 |
| BARO | all | 0.133 ± 0.000 | 0.878 ± 0.000 | 0.731 ± 0.000 | 45.09 | 0/270 |
| Nsigma | all | 0.800 ± 0.000 | 0.933 ± 0.000 | 0.911 ± 0.000 | 41.92 | 0/270 |
| epsilon-Diagnosis | all | 0.011 ± 0.000 | 0.144 ± 0.000 | 0.107 ± 0.000 | 1478.01 | 0/270 |
| CIRCA | all | 0.667 ± 0.000 | 0.878 ± 0.000 | 0.849 ± 0.000 | 7401.12 | 6/270 |
| RCD | all | 0.600 ± 0.000 | 0.633 ± 0.000 | 0.647 ± 0.000 | 16632.85 | 9/270 |

## RE2 / Sock Shop

| Method | Cohort | AC@1 | AC@3 | Avg@5 | Median ms | Failed / scored attempts |
|---|---|---:|---:|---:|---:|---:|
| STA | all | 0.870 ± 0.005 | 0.985 ± 0.005 | 0.959 ± 0.006 | 19.72 | 0/270 |
| BARO | all | 0.133 ± 0.000 | 0.956 ± 0.000 | 0.780 ± 0.000 | 51.82 | 0/270 |
| Nsigma | all | 0.844 ± 0.000 | 0.978 ± 0.000 | 0.951 ± 0.000 | 58.56 | 0/270 |
| epsilon-Diagnosis | all | 0.178 ± 0.000 | 0.456 ± 0.000 | 0.364 ± 0.000 | 1630.90 | 0/270 |
| CIRCA | all | 0.778 ± 0.000 | 0.933 ± 0.000 | 0.909 ± 0.000 | 9644.25 | 6/270 |
| RCD | all | 0.389 ± 0.000 | 0.556 ± 0.000 | 0.542 ± 0.000 | 11340.29 | 0/270 |

## RE2 / Train Ticket

| Method | Cohort | AC@1 | AC@3 | Avg@5 | Median ms | Failed / scored attempts |
|---|---|---:|---:|---:|---:|---:|
| STA | all | 0.815 ± 0.005 | 0.900 ± 0.018 | 0.888 ± 0.012 | 86.80 | 0/270 |
| BARO | all | 0.722 ± 0.000 | 0.844 ± 0.000 | 0.820 ± 0.000 | 266.77 | 0/270 |
| Nsigma | all | 0.789 ± 0.000 | 0.856 ± 0.000 | 0.842 ± 0.000 | 246.38 | 0/270 |
| epsilon-Diagnosis | all | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 9017.80 | 0/270 |
| STA | fixed TT sample | 0.667 ± 0.000 | 0.722 ± 0.039 | 0.722 ± 0.016 | 87.67 | 0/36 |
| BARO | fixed TT sample | 0.583 ± 0.000 | 0.750 ± 0.000 | 0.700 ± 0.000 | 295.66 | 0/36 |
| Nsigma | fixed TT sample | 0.667 ± 0.000 | 0.667 ± 0.000 | 0.683 ± 0.000 | 235.62 | 0/36 |
| epsilon-Diagnosis | fixed TT sample | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 9323.91 | 0/36 |
| CIRCA | fixed TT sample | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 90031.07 | 36/36 |
| RCD | fixed TT sample | 0.083 ± 0.000 | 0.083 ± 0.000 | 0.083 ± 0.000 | 90000.13 | 21/36 |

## Online Boutique RE1 baseline errors

| Fault | BARO hit at 1, STA missed | Root in STA top-k |
|---|---:|---:|
| delay | 8 | 3 |
| disk | 4 | 4 |
| loss | 3 | 0 |

## RE2 Train Ticket errors

### `relative`

Counts include each case once per training seed.

| Fault | Attempts | Top-1 hits | Miss, root outside top-k | Miss, root inside top-k |
|---|---:|---:|---:|---:|
| cpu | 45 | 45 | 0 | 0 |
| delay | 45 | 27 | 3 | 15 |
| disk | 45 | 28 | 3 | 14 |
| loss | 45 | 17 | 8 | 20 |
| mem | 45 | 28 | 1 | 16 |
| socket | 45 | 43 | 0 | 2 |

Fault types absent from the RE1 cohort (socket) account for 2 of 82 misses.

### `prior`

Counts include each case once per training seed.

| Fault | Attempts | Top-1 hits | Miss, root outside top-k | Miss, root inside top-k |
|---|---:|---:|---:|---:|
| cpu | 45 | 45 | 0 | 0 |
| delay | 45 | 30 | 3 | 12 |
| disk | 45 | 45 | 0 | 0 |
| loss | 45 | 20 | 8 | 17 |
| mem | 45 | 38 | 0 | 7 |
| socket | 45 | 42 | 0 | 3 |

Fault types absent from the RE1 cohort (socket) account for 3 of 50 misses.

Train Ticket metric families present only in RE2: diskio, socket. The fault breakdown does not attribute a miss to a metric family. The prior scores all available families together, so its gains do not isolate either family.

## Iteration 0: Post-injection telemetry with train-only scaling

No earlier design measured.

| Suite / system | AC@1 | Delta | Avg@5 | Delta |
|---|---:|---:|---:|---:|
| RE1 / Online Boutique | 0.600 | n/a | 0.717 | n/a |
| RE1 / Sock Shop | 0.893 | n/a | 0.923 | n/a |
| RE1 / Train Ticket | 0.827 | n/a | 0.891 | n/a |

## Iteration 1: Pre-injection median and IQR scaling per metric and case

Delta against `baseline`.

| Suite / system | AC@1 | Delta | Avg@5 | Delta |
|---|---:|---:|---:|---:|
| RE1 / Online Boutique | 0.947 | +0.347 | 0.979 | +0.261 |
| RE1 / Sock Shop | 0.947 | +0.053 | 0.981 | +0.059 |
| RE1 / Train Ticket | 0.840 | +0.013 | 0.952 | +0.061 |
| RE2 / Online Boutique | 0.881 | n/a | 0.950 | n/a |
| RE2 / Sock Shop | 0.885 | n/a | 0.962 | n/a |
| RE2 / Train Ticket | 0.696 | n/a | 0.812 | n/a |

## Iteration 2: Median and IQR scaling plus onset and persistence channels

Delta against `relative`.

| Suite / system | AC@1 | Delta | Avg@5 | Delta |
|---|---:|---:|---:|---:|
| RE1 / Online Boutique | 0.867 | -0.080 | 0.920 | -0.059 |
| RE1 / Sock Shop | 0.933 | -0.013 | 0.987 | +0.005 |
| RE1 / Train Ticket | 0.840 | +0.000 | 0.941 | -0.011 |

## Iteration 3: Median and IQR scaling plus signed within-bin tail deviations

Delta against `relative`.

| Suite / system | AC@1 | Delta | Avg@5 | Delta |
|---|---:|---:|---:|---:|
| RE1 / Online Boutique | 0.880 | -0.067 | 0.952 | -0.027 |
| RE1 / Sock Shop | 0.973 | +0.027 | 0.992 | +0.011 |
| RE1 / Train Ticket | 0.827 | -0.013 | 0.928 | -0.024 |

## Iteration 4: Add an all-metric change prior to the existing STA ranking

Delta against `relative`.

| Suite / system | AC@1 | Delta | Avg@5 | Delta |
|---|---:|---:|---:|---:|
| RE1 / Online Boutique | 0.947 | +0.000 | 0.979 | +0.000 |
| RE1 / Sock Shop | 0.960 | +0.013 | 0.984 | +0.003 |
| RE1 / Train Ticket | 0.840 | +0.000 | 0.952 | +0.000 |
| RE2 / Online Boutique | 0.870 | -0.011 | 0.952 | +0.002 |
| RE2 / Sock Shop | 0.870 | -0.015 | 0.959 | -0.004 |
| RE2 / Train Ticket | 0.815 | +0.119 | 0.888 | +0.076 |
