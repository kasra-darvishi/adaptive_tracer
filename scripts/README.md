# Training sweep

SLURM job scripts for the sweep reported in the papers: five seeds per architecture,
each job on 4× V100 with mixed precision (`--amp`), and gradient checkpointing
(`--chk`) for the Transformer's 2048-event sequences.

| Script | Backbone | Seeds |
|---|---|---|
| `lstm-{1..5}.sh` | 2-layer LSTM, 256 hidden | 1–5 |
| `transformer-{1..5}.sh` | 2-layer Transformer, 672 hidden, 4 heads, SwiGLU | 1–5 |

Both train the multi-task model (`--train_event_model --train_latency_model`). Drop
either flag to reproduce the corresponding single-task baseline.

Set `DATA_ROOT` to the unpacked Zenodo trace data and fill in `--account` with your own
allocation, then submit from the repository root:

```bash
DATA_ROOT=/scratch/$USER/trace_data sbatch scripts/lstm-1.sh
```

`PROJECT_ROOT` (default: the submission directory) and `VENV` (default: `~/venv`) can be
overridden the same way.

These scripts pass `--n_categories 6`, i.e. **5** duration categories. The best Apache
operating point reported in the journal paper is 7 categories — `--n_categories 8`, on
data prepared with the same value. See the note in the top-level README.
