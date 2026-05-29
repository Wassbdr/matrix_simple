# ObservIT cluster — per-scenario breakdown

Source: `results_transmission/paper_benchmark_observit{,_mm}.json`.
Five chaos scenarios from `scripts/run_sockshop_campaign.sh`, five seeds each (42, 123, 7, 1337, 2025).
Values are Avg@1 mean +/- std across 5 seeds.

## Scenario x Model matrix

| Scenario | Failure Type | Root Cause | STA | GATv2 | Late-STGCN+GIB |
|---|---|---|---:|---:|---:|
| `stress-cpu-catalogue` | **hard** | `catalogue` | 0.966 +/- 0.026 | 0.959 +/- 0.023 | 0.951 +/- 0.035 |
| `stress-mem-carts` | **hard** | `carts` | 0.987 +/- 0.014 | 0.987 +/- 0.014 | 0.984 +/- 0.012 |
| `network-delay-orders-to-orders-db` | **cascade** | `orders-db` | 0.043 +/- 0.095 | 0.459 +/- 0.445 | 0.003 +/- 0.007 |
| `network-loss-catalogue-to-catalogue-db` | **gray** | `catalogue-db` | 0.987 +/- 0.017 | 0.508 +/- 0.404 | 0.997 +/- 0.007 |
| `pod-kill-user` | **hard** | `user` | 0.951 +/- 0.000 | 0.951 +/- 0.000 | 0.741 +/- 0.360 |

## Aggregated by failure type

| Failure type | Scenarios | STA mean Avg@1 | GATv2 mean Avg@1 | Late-STGCN+GIB mean Avg@1 |
|---|---|---:|---:|---:|
| **hard** | stress-cpu-catalogue, stress-mem-carts, pod-kill-user | 0.968 | 0.965 | 0.892 |
| **cascade** | network-delay-orders-to-orders-db | 0.043 | 0.459 | 0.003 |
| **gray** | network-loss-catalogue-to-catalogue-db | 0.987 | 0.508 | 0.997 |

## Observations

- **Gray failure (`catalogue-db` / network-loss)**: STA Avg@1 = 0.987, GATv2 Avg@1 = 0.508. This is the hardest scenario for STA's hard top-k selector when the signal sits in a sub-threshold metric.
- **Cascade failure (`orders-db` / network-delay)**: STA Avg@1 = 0.043, GATv2 Avg@1 = 0.459. Multi-hop propagation; GATv2's soft attention can spread weight along the cascade chain.
- **Hard failures** (CPU stress, memory stress, pod-kill): STA mean = 0.968, GATv2 mean = 0.965. Both models handle isolated, high-signal anomalies well.

