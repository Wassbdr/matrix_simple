# Architecture

## One ranking path

`rca rank`, `rca_pipeline.rank`, and the default FastAPI `/predict` route use the same `Ranker`. Telemetry determines the candidates; incident directory names never do. We split at the injection time rather than use the final rows of the export, since how much time passes after an incident changes what a later sample tells the ranker.

## Learned ranking and change prior

For each service, we scale post-injection metrics by their pre-injection median and interquartile range. The light temporal encoder scores all services; a deep scorer handles the selected subset only. The public track supplies no dependency graph, so these models use an empty edge list. The deep block still models each selected service's temporal features. Older topology-specific models and bundle serving remain separate from this path.

The prior uses the strongest high-tail change per service across the available metric families, including the ones the learned encoder does not consume. It nudges the final scores without changing the selector or demanding a retrain when a metric family first appears. [The iteration log](../benchmarks/results.md) shows which systems gained and which regressed.

## Reproduction and evaluation

The packaged checkpoint pools RE1 so unnamed, variable-size exports can use a single model. The benchmark trains separate models per system with grouped development splits and a held-out release, then compares services rather than metric column names. Reference methods run in isolated environments with a per-case cache so their dependency pins do not affect the product.

The deep phase has a fixed candidate cap, but feature preparation and the selector still see every service. Full-graph sparse models can be faster at small sizes. Use `make scaling` for the synthetic latency and accuracy protocol; `make bench` is the metrics-only public incident comparison.
