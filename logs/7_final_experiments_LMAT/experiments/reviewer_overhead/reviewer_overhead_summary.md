# Reviewer-Facing Overhead Results

Reference load: **200 virtual users**
Maximum-throughput criterion: **error rate <= 3.0%**

## P95/P99 Latency At Reference Load

| Condition | Throughput (req/s) | P50 (ms) | P95 (ms) | P99 (ms) | Error rate | P95 overhead vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 172.7 ± 11.3 | 96.0 | 412.2 | 744.6 | 2.04% | 0.0% |
| lttng_only | 164.8 ± 12.4 | 120.4 | 557.1 | 922.5 | 1.98% | 35.1% |
| lmat_async | 181.1 ± 3.5 | 60.4 | 239.1 | 584.7 | 2.21% | -42.0% |

## Maximum Throughput

| Condition | Users at max throughput | Max throughput (req/s) | P95 at max throughput (ms) | P99 at max throughput (ms) | Error rate |
|---|---:|---:|---:|---:|---:|
| baseline | 200 | 172.7 | 412.2 | 744.6 | 2.04% |
| lttng_only | 200 | 164.8 | 557.1 | 922.5 | 1.98% |
| lmat_async | 200 | 181.1 | 239.1 | 584.7 | 2.21% |

## Files

- Per-user summary CSV: `/home/sehgaluv17/experiments/reviewer_overhead/reviewer_overhead_by_users.csv`
- Maximum-throughput CSV: `/home/sehgaluv17/experiments/reviewer_overhead/reviewer_overhead_max_throughput.csv`
