# Reviewer-Facing Overhead Results

Reference load: **200 virtual users**
Maximum-throughput criterion: **error rate <= 3.0%**

## P95/P99 Latency At Reference Load

| Condition | Throughput (req/s) | P50 (ms) | P95 (ms) | P99 (ms) | Error rate | P95 overhead vs baseline | P95 overhead vs LTTng only |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 178.8 ± 4.5 | 66.6 | 313.7 | 718.7 | 2.05% | 0.0% | 57.2% |
| lttng_only | 183.3 ± 0.8 | 56.3 | 199.5 | 645.1 | 2.03% | -36.4% | 0.0% |
| lmat_async | 180.9 ± 3.8 | 64.9 | 225.5 | 496.2 | 2.16% | -28.1% | 13.0% |

## Maximum Throughput

| Condition | Users at max throughput | Max throughput (req/s) | P95 at max throughput (ms) | P99 at max throughput (ms) | Error rate |
|---|---:|---:|---:|---:|---:|
| baseline | 200 | 178.8 | 313.7 | 718.7 | 2.05% |
| lttng_only | 200 | 183.3 | 199.5 | 645.1 | 2.03% |
| lmat_async | 200 | 180.9 | 225.5 | 496.2 | 2.16% |

## Files

- Per-user summary CSV: `/home/sehgaluv17/experiments/reviewer_overhead_200_fair/reviewer_overhead_by_users.csv`
- Maximum-throughput CSV: `/home/sehgaluv17/experiments/reviewer_overhead_200_fair/reviewer_overhead_max_throughput.csv`

## Interpretation Note

- `lttng_only` is the direct tracing-overhead condition.
- `lmat_async` is a co-located replay-based proxy for tracing-plus-inference CPU overhead, so its fairest incremental comparison is against `lttng_only`, not against a colder baseline run.
