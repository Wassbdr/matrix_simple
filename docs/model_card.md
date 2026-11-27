# STA model card

**Intended use:** suggest service-level root causes to an on-call engineer after an incident starts. Treat the ranking as a diagnostic cue, not a trigger for automated remediation.

**Packaged model:** `re1-pooled-seed42-v1` is trained on all public RCAEval RE1 cases with seed 42, for 36 epochs on one CPU thread. The pool covers Online Boutique, Sock Shop, and Train Ticket. Cases without a metric for the injected service cannot supervise training. No RE2 or RE3 case is used. `make bundle` rebuilds the checkpoint from the pinned release described in [data setup](rcaeval_data.md). Weights and normalization constants are packaged; raw telemetry is not.

**Evaluation:** [the benchmark](../benchmarks/results.md) compares separate per-system models, not the packaged pooled checkpoint. It keeps RE1 repetitions together in service/fault groups, fits with seeds 7, 42, and 123, and evaluates transfer on RE2 with no tuning on it. The ranking design was selected on RE1 development. Use the benchmark to compare designs and do not assign those scores to the pooled checkpoint. The showcase shows one held-out case, not an accuracy estimate.

**Inputs:** a known incident start, numeric service metrics on both sides of it, and consistent time units. The learned channels cover CPU, memory, disk I/O, socket, workload, latency, and errors; a training-free change score also uses other present metric families. See the [export tutorial](tutorial.md).

**Limits:** missing services cannot be ranked. Loss faults remain the largest group of RE2 Train Ticket misses, and STA only ties Nsigma at top one on the fixed Train Ticket subset. There is no RE3 result, no trace or log input in the product ranker, and no calibration of scores into probabilities. Timing was measured on CPU. Check ranks against your actual incidents before relying on them.
