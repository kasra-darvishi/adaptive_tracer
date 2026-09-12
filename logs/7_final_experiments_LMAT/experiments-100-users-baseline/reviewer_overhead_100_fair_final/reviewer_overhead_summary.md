# Reviewer-Facing Overhead Results

Reference load: **100 virtual users**
Maximum-throughput criterion: **error rate <= 3.0%**

## P95/P99 Latency At Reference Load

| Condition | Throughput (req/s) | P50 (ms) | P95 (ms) | P99 (ms) | Error rate | P95 overhead vs baseline | P95 overhead vs LTTng only |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 89.0 ± 0.2 | 47.6 | 54.1 | 78.3 | 9.69% | 0.0% | -3.3% |
| lttng_only | 88.4 ± 1.1 | 48.4 | 55.9 | 82.9 | 9.90% | 3.4% | 0.0% |
| lmat_async | 88.7 ± 0.3 | 47.1 | 53.4 | 73.6 | 9.93% | -1.3% | -4.5% |

## Maximum Throughput

| Condition | Users at max throughput | Max throughput (req/s) | P95 at max throughput (ms) | P99 at max throughput (ms) | Error rate |
|---|---:|---:|---:|---:|---:|

## Files

- Per-user summary CSV: `/home/sehgaluv17/experiments/reviewer_overhead_100_fair_final/reviewer_overhead_by_users.csv`
- Maximum-throughput CSV: `/home/sehgaluv17/experiments/reviewer_overhead_100_fair_final/reviewer_overhead_max_throughput.csv`

## Interpretation Note

- `lttng_only` is the direct tracing-overhead condition.
- `lmat_async` is a co-located replay-based proxy for tracing-plus-inference CPU overhead, so its fairest incremental comparison is against `lttng_only`, not against a colder baseline run.
