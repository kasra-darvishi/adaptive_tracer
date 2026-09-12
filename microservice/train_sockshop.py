#!/usr/bin/env python3
"""
train_sockshop.py — LMAT training on SockShop NPZ dataset
==========================================================

H100-optimized training script for the SockShop anomaly detection
experiment.  Reads preprocessed NPZ shards, trains an LSTM or Transformer
model with next-syscall prediction + latency categorisation, then runs OOD
evaluation (AUROC/AUPR) on all anomaly splits.

Key features:
  - BF16 mixed-precision (H100 native)
  - torch.compile (reduce-overhead mode)
  - Gradient accumulation for large effective batch sizes
  - pin_memory + non_blocking H2D transfers
  - WandB + CSV logging
  - Cosine LR schedule with linear warm-up
  - OOD evaluation: per-anomaly-type AUROC, AUPR, and Apache-style F1
    (threshold tuned on valid_id + valid_ood_* when present)

Usage (single H100):
  python -u microservice/train_sockshop.py \\
      --preprocessed_dir /scratch/.../preprocessed \\
      --model transformer \\
      --n_head 8 --n_hidden 1024 --n_layer 6 \\
      --dim_sys 64 --dim_entry 8 --dim_ret 8 \\
      --dim_proc 8 --dim_pid 16 --dim_tid 16 \\
      --dim_order 16 --dim_time 16 \\
      --batch 512 --accum_steps 4 --n_epochs 20 \\
      --lr 3e-4 --warmup_steps 2000 \\
      --train_event_model --train_latency_model \\
      --amp --compile \\
      --wandb_project sockshop_lmat \\
      --log_dir logs/sockshop_exp1
"""

import os
import sys
import json
import math
import time
import random
import argparse
import csv
import pickle
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch._inductor.config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

# Ensure project root on sys.path
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from models import LSTM, Transformer
from microservice.NpzDataset import SockshopNpzDataset, sockshop_collate_fn

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    import sklearn.metrics  # noqa: F401 — detect sklearn for OOD metrics

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

if HAS_SKLEARN:
    from microservice.ood_metrics import (
        balanced_binary_scores_labels,
        metrics_at_threshold,
        rank_metrics,
        tune_threshold_max_f1,
    )


###############################################################################
# Arguments
###############################################################################

def get_args():
    p = argparse.ArgumentParser(description="Train LMAT on SockShop NPZ dataset")

    # Data
    p.add_argument("--preprocessed_dir", required=True,
                   help="Path to preprocessed/ directory containing split subdirs")
    p.add_argument("--n_categories", type=int, default=6)
    p.add_argument("--max_seq_len",  type=int, default=512)
    p.add_argument("--max_samples",  type=int, default=None)
    p.add_argument("--lat_score_weight", type=float, default=0.5,
                   help="Legacy option from the old weighted-fusion path. "
                        "Paper-aligned scoring now uses MAD-normalized "
                        "event+duration losses when both heads are trained.")
    p.add_argument(
        "--ood_score",
        choices=["combined", "event", "latency"],
        default="combined",
        help="Which head(s) drive OOD anomaly scores. For paper-aligned runs, "
             "single-task models use their own head loss and multi-task models "
             "use MAD-normalized event+duration loss summation.",
    )
    p.add_argument(
        "--ood_threshold_grid",
        type=int,
        default=100,
        help="Number of threshold steps for validation F1 tuning (Apache n-gram style).",
    )

    # Model
    p.add_argument("--model", choices=["lstm", "transformer"], default="transformer")
    p.add_argument("--n_head",   type=int,   default=8)
    p.add_argument("--n_hidden", type=int,   default=1024)
    p.add_argument("--n_layer",  type=int,   default=6)
    p.add_argument("--dropout",  type=float, default=0.1)
    p.add_argument("--activation", choices=["relu", "gelu", "swiglu"], default="gelu")
    p.add_argument("--tfixup", action="store_true")
    p.add_argument("--dim_sys",    type=int, default=64)
    p.add_argument("--dim_entry",  type=int, default=8)
    p.add_argument("--dim_ret",    type=int, default=8)
    p.add_argument("--dim_proc",   type=int, default=8)
    p.add_argument("--dim_pid",    type=int, default=16)
    p.add_argument("--dim_tid",    type=int, default=16)
    p.add_argument("--dim_order",  type=int, default=16)
    p.add_argument("--dim_time",   type=int, default=16)
    p.add_argument("--dim_f_mean", type=int, default=0)
    p.add_argument("--train_event_model",    action="store_true")
    p.add_argument("--train_latency_model",  action="store_true")
    p.add_argument("--ordinal_latency",      action="store_true")
    p.add_argument(
        "--multitask_lambda",
        type=float,
        default=0.5,
        help="Lambda in the paper's multitask loss: L = lambda*L_event + "
             "(1-lambda)*L_duration. Ignored when only one head is trained.",
    )
    p.add_argument("--label_smoothing", type=float, default=0.0)

    # Training
    p.add_argument("--batch",        type=int,   default=256)
    p.add_argument("--accum_steps",  type=int,   default=4,
                   help="Gradient accumulation steps (eff_batch = batch * accum_steps)")
    p.add_argument("--n_epochs",     type=int,   default=20)
    p.add_argument("--early_stopping_patience", type=int, default=0,
                   help="Stop after this many consecutive validations without "
                        "improvement in validation loss. 0 disables early stopping.")
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int,   default=2000)
    p.add_argument("--clip",         type=float, default=1.0)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--amp",     action="store_true", help="BF16 mixed-precision (H100)")
    p.add_argument("--compile", action="store_true", help="torch.compile the model")
    p.add_argument("--chk",     action="store_true", help="Gradient checkpointing")
    p.add_argument("--seed",    type=int, default=42)

    # Logging
    p.add_argument("--log_dir",        type=str, default="logs/sockshop")
    p.add_argument("--save_every",     type=int, default=5000)
    p.add_argument("--eval_every",     type=int, default=2000)
    p.add_argument("--wandb_project",  type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--load_model",     type=str, default=None)

    # GPU — when launched via torchrun, LOCAL_RANK overrides --gpu
    p.add_argument("--gpu", type=int, default=0,
                   help="GPU index (single-GPU mode). Overridden by torchrun LOCAL_RANK.")

    args = p.parse_args()
    if not (args.train_event_model or args.train_latency_model):
        p.error("At least one of --train_event_model / --train_latency_model is required")
    if args.ood_score == "event" and not args.train_event_model:
        p.error("--ood_score event requires --train_event_model")
    if args.ood_score == "latency" and not args.train_latency_model:
        p.error("--ood_score latency requires --train_latency_model")
    if args.early_stopping_patience < 0:
        p.error("--early_stopping_patience must be >= 0")
    if not (0.0 <= args.multitask_lambda <= 1.0):
        p.error("--multitask_lambda must be in [0,1]")
    return args


###############################################################################
# LR schedule — cosine with linear warm-up
###############################################################################

def lr_lambda(step, warmup_steps, total_steps, min_ratio=0.05):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


###############################################################################
# Logging
###############################################################################

class Logger:
    def __init__(self, log_dir, use_wandb, args):
        self.use_wandb = use_wandb and HAS_WANDB
        self.csv_path  = os.path.join(log_dir, "metrics.csv")
        self._writer   = None
        self._file     = None
        if self.use_wandb:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or
                     f"{args.model}_{datetime.now():%Y%m%d_%H%M%S}",
                config=vars(args),
                dir=log_dir,
            )

    def log(self, metrics: dict, step: int):
        if self.use_wandb:
            wandb.log(metrics, step=step)
        
        # Check if we need to initialize or update fieldnames
        current_keys = ["step"] + list(metrics.keys())
        needs_new_header = False
        
        if self._writer is None:
            self._file = open(self.csv_path, "w", newline="", buffering=1)
            self._writer = csv.DictWriter(self._file, fieldnames=current_keys)
            needs_new_header = True
        else:
            missing_keys = [k for k in current_keys if k not in self._writer.fieldnames]
            if missing_keys:
                self._writer.fieldnames.extend(missing_keys)
                needs_new_header = True
                
        if needs_new_header:
            self._writer.writeheader()
            
        row = {"step": step}
        row.update({k: f"{v:.6g}" if isinstance(v, float) else v
                    for k, v in metrics.items()})
        self._writer.writerow(row)

    def save_model(self, model, path):
        torch.save(model.state_dict(), path)
        if self.use_wandb:
            wandb.save(path)

    def close(self):
        if self._file:
            self._file.close()
        if self.use_wandb:
            wandb.finish()


###############################################################################
# Model factory
###############################################################################

def build_model(args, n_syscall, n_process, device):
    kw = dict(
        n_syscall=n_syscall, n_category=args.n_categories, n_process=n_process,
        n_hidden=args.n_hidden, n_layer=args.n_layer, dropout=args.dropout,
        dim_sys=args.dim_sys, dim_entry=args.dim_entry, dim_ret=args.dim_ret,
        dim_proc=args.dim_proc, dim_pid=args.dim_pid, dim_tid=args.dim_tid,
        dim_order=args.dim_order, dim_time=args.dim_time, dim_f_mean=args.dim_f_mean,
        train_event=args.train_event_model, train_latency=args.train_latency_model,
        ordinal_latency=args.ordinal_latency,
    )
    if args.model == "lstm":
        return LSTM(**kw).to(device)
    return Transformer(n_head=args.n_head, activation=args.activation,
                       tfixup=args.tfixup, **kw).to(device)


###############################################################################
# Forward pass helper
###############################################################################

def forward_batch(model, batch, device, args):
    def t(key, dtype=torch.long):
        return batch[key].to(device, dtype=dtype, non_blocking=True)

    call     = t("call")
    entry    = t("entry")
    duration = t("duration")
    proc     = t("proc")
    pid      = t("pid")
    tid      = t("tid")
    ret      = t("ret")
    pad_mask = batch["pad_mask"].to(device, non_blocking=True)

    if args.model == "transformer":
        return model(call, entry, duration, proc, pid, tid, ret,
                     pad_mask=pad_mask, chk=args.chk)
    return model(call, entry, duration, proc, pid, tid, ret)


###############################################################################
# Loss
###############################################################################

def compute_loss(logits_e, logits_l, batch, device, args, crit_e, crit_l):
    tgt_call = batch["tgt_call"].to(device, dtype=torch.long, non_blocking=True)
    tgt_lat  = batch["tgt_lat" ].to(device, dtype=torch.long, non_blocking=True)
    loss = torch.tensor(0.0, device=device)
    loss_e = loss_l = torch.tensor(0.0, device=device)

    has_event = args.train_event_model and crit_e and logits_e.numel() > 0
    has_latency = args.train_latency_model and crit_l and logits_l.numel() > 0

    if has_event:
        B, L, V = logits_e.shape
        loss_e = crit_e(logits_e.reshape(B*L, V), tgt_call.reshape(B*L))

    if has_latency:
        B, L, C = logits_l.shape
        if args.ordinal_latency:
            loss_l = crit_l(logits_l.reshape(B*L, C),
                            tgt_lat.reshape(B*L, 1).float().expand(B*L, C))
        else:
            loss_l = crit_l(logits_l.reshape(B*L, C), tgt_lat.reshape(B*L))

    if has_event and has_latency:
        lam = float(args.multitask_lambda)
        loss = lam * loss_e + (1.0 - lam) * loss_l
    elif has_event:
        loss = loss_e
    elif has_latency:
        loss = loss_l

    return loss, loss_e, loss_l


###############################################################################
# Accuracy
###############################################################################

def token_accuracy(logits, targets):
    if logits.numel() == 0:
        return float("nan")
    pred = logits.argmax(-1)
    mask = targets != 0
    if not mask.any():
        return float("nan")
    return (pred[mask] == targets[mask]).float().mean().item()


def per_sequence_event_ce(le, tgt_call):
    """Mean next-syscall cross-entropy per sequence (non-pad tokens only)."""
    if le.numel() == 0:
        return None
    B, L, V = le.shape
    per_tok = nn.CrossEntropyLoss(ignore_index=0, reduction="none")(
        le.reshape(B * L, V), tgt_call.reshape(B * L)
    ).reshape(B, L)
    mask = tgt_call != 0
    return (per_tok * mask).sum(1) / mask.sum(1).clamp(min=1)


def per_sequence_latency_ce(ll, tgt_lat, ordinal_latency):
    """Mean latency-head loss per sequence (non-pad latency targets only)."""
    if ll.numel() == 0:
        return None
    B, L, C = ll.shape
    mask = tgt_lat != 0
    if ordinal_latency:
        per_tok = (
            nn.BCEWithLogitsLoss(reduction="none")(
                ll.reshape(B * L, C),
                tgt_lat.reshape(B * L, 1).float().expand(B * L, C),
            )
            .mean(-1)
            .reshape(B, L)
        )
    else:
        per_tok = nn.CrossEntropyLoss(ignore_index=0, reduction="none")(
            ll.reshape(B * L, C), tgt_lat.reshape(B * L)
        ).reshape(B, L)
    return (per_tok * mask).sum(1) / mask.sum(1).clamp(min=1)


def _scores_to_numpy(seq_scores):
    if seq_scores is None:
        return np.array([], dtype=np.float64)
    return (
        seq_scores.detach().to(dtype=torch.float32).cpu().numpy().astype(np.float64, copy=False)
    )


def _compute_mad_stats(values: np.ndarray) -> dict[str, float] | None:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    if not np.isfinite(mad) or mad <= 1e-12:
        mad = 1.0
    return {"median": median, "mad": mad}


def _mad_normalize(values: np.ndarray, stats: dict[str, float] | None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or stats is None:
        return arr
    return (arr - stats["median"]) / stats["mad"]


def combine_paper_ood_scores(
    scores_event: np.ndarray,
    scores_latency: np.ndarray,
    args,
    mad_stats_event: dict[str, float] | None = None,
    mad_stats_latency: dict[str, float] | None = None,
) -> np.ndarray:
    """Paper-aligned anomaly score.

    Single-task runs use their own head loss directly.
    Multi-task runs normalize each head loss with MAD and sum them.
    """
    has_event = args.train_event_model and scores_event.size > 0
    has_latency = args.train_latency_model and scores_latency.size > 0

    if has_event and has_latency:
        return _mad_normalize(scores_event, mad_stats_event) + _mad_normalize(
            scores_latency, mad_stats_latency
        )
    if has_event:
        return scores_event.astype(np.float64, copy=False)
    if has_latency:
        return scores_latency.astype(np.float64, copy=False)
    return np.array([], dtype=np.float64)


def combine_full_binary_scores_labels(
    scores_id: np.ndarray, scores_ood: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    scores_id = np.asarray(scores_id, dtype=np.float64)
    scores_ood = np.asarray(scores_ood, dtype=np.float64)
    if scores_id.size == 0 or scores_ood.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.int64)
    scores = np.concatenate([scores_id, scores_ood])
    y = np.concatenate(
        [
            np.zeros(scores_id.size, dtype=np.int64),
            np.ones(scores_ood.size, dtype=np.int64),
        ]
    )
    return scores, y


def combine_balanced_binary_scores_labels(
    scores_id: np.ndarray, scores_ood: np.ndarray, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Create a balanced ID/OOD validation set with deterministic subsampling.

    This is used for threshold tuning so the chosen threshold does not collapse
    to the majority class when pooled validation OOD counts greatly exceed ID.
    """
    scores_id = np.asarray(scores_id, dtype=np.float64)
    scores_ood = np.asarray(scores_ood, dtype=np.float64)
    m = min(scores_id.size, scores_ood.size)
    if m == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.int64)

    rng = np.random.default_rng(seed)
    if scores_id.size > m:
        idx_id = np.sort(rng.choice(scores_id.size, size=m, replace=False))
        scores_id = scores_id[idx_id]
    if scores_ood.size > m:
        idx_ood = np.sort(rng.choice(scores_ood.size, size=m, replace=False))
        scores_ood = scores_ood[idx_ood]

    scores = np.concatenate([scores_id, scores_ood])
    y = np.concatenate(
        [
            np.zeros(m, dtype=np.int64),
            np.ones(m, dtype=np.int64),
        ]
    )
    return scores, y


###############################################################################
# Evaluation
###############################################################################

@torch.no_grad()
def evaluate_split(model, loader, device, args, crit_e, crit_l,
                   return_scores=False):
    model.eval()
    tot_loss = tot_e = tot_l = 0.0
    ae_sum = ae_cnt = al_sum = al_cnt = 0.0
    n = 0
    scores_event, scores_latency, labels = [], [], []

    for batch in loader:
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=args.amp and device.type == "cuda"):
            le, ll = forward_batch(model, batch, device, args)
            loss, loss_e, loss_l = compute_loss(le, ll, batch, device, args, crit_e, crit_l)

        v_loss = loss.item()
        if not math.isnan(v_loss):
            tot_loss += v_loss
            tot_e    += loss_e.item()
            tot_l    += loss_l.item()
            n += 1
        tgt_call  = batch["tgt_call"].to(device, dtype=torch.long, non_blocking=True)
        tgt_lat   = batch["tgt_lat" ].to(device, dtype=torch.long, non_blocking=True)
        if args.train_event_model and le.numel() > 0:
            v = token_accuracy(le, tgt_call)
            if not math.isnan(v): ae_sum += v; ae_cnt += 1
        if args.train_latency_model and ll.numel() > 0:
            v = token_accuracy(ll, tgt_lat)
            if not math.isnan(v): al_sum += v; al_cnt += 1

        if return_scores:
            tgt_call_sc = batch["tgt_call"].to(
                device, dtype=torch.long, non_blocking=True
            )
            tgt_lat_sc = batch["tgt_lat"].to(
                device, dtype=torch.long, non_blocking=True
            )
            seq_sc_e = per_sequence_event_ce(le, tgt_call_sc) if le.numel() > 0 else None
            seq_sc_l = (
                per_sequence_latency_ce(ll, tgt_lat_sc, args.ordinal_latency)
                if ll.numel() > 0
                else None
            )
            scores_event.append(_scores_to_numpy(seq_sc_e))
            scores_latency.append(_scores_to_numpy(seq_sc_l))
            if seq_sc_e is not None or seq_sc_l is not None:
                labels.append(batch["is_anomaly"].numpy())



    out = dict(loss=tot_loss/max(n,1), loss_e=tot_e/max(n,1),
               loss_l=tot_l/max(n,1),
               acc_e=ae_sum/max(ae_cnt,1), acc_l=al_sum/max(al_cnt,1))
    if return_scores:
        out["scores_event"] = (
            np.concatenate(scores_event) if scores_event else np.array([], dtype=np.float64)
        )
        out["scores_latency"] = (
            np.concatenate(scores_latency) if scores_latency else np.array([], dtype=np.float64)
        )
        out["labels"] = np.concatenate(labels) if labels else np.array([])
    return out


###############################################################################
# OOD evaluation
###############################################################################

def _make_ood_loader(preprocessed_dir, subdir, batch, max_seq_len, pin_memory):
    d = os.path.join(preprocessed_dir, subdir)
    if not os.path.isdir(d):
        return None
    ds = SockshopNpzDataset(
        d,
        batch_size=batch,
        max_seq_len=max_seq_len,
        shuffle_shards=False,
    )
    return DataLoader(
        ds,
        batch_size=None,
        collate_fn=sockshop_collate_fn,
        num_workers=2,
        pin_memory=pin_memory,
    )


def run_ood_eval_legacy_weighted(model, args, device, crit_e, crit_l, log_fn):
    if not HAS_SKLEARN:
        log_fn("[OOD] sklearn not available — skipping")
        return {}

    pin = device.type == "cuda"
    base = args.preprocessed_dir
    if not os.path.isdir(os.path.join(base, "test_id")):
        log_fn("[OOD] test_id not found")
        return {}

    valid_id_dir = os.path.join(base, "valid_id")
    has_valid_root = os.path.isdir(valid_id_dir)

    results = {}
    for atype in ["cpu", "disk", "mem", "net"]:
        test_ood_sub = f"test_ood_{atype}"
        if not os.path.isdir(os.path.join(base, test_ood_sub)):
            continue

        ld_tid = _make_ood_loader(base, "test_id", args.batch, args.max_seq_len, pin)
        ld_tood = _make_ood_loader(base, test_ood_sub, args.batch, args.max_seq_len, pin)
        if ld_tid is None or ld_tood is None:
            continue

        res_tid = evaluate_split(
            model, ld_tid, device, args, crit_e, crit_l, return_scores=True
        )
        res_tood = evaluate_split(
            model, ld_tood, device, args, crit_e, crit_l, return_scores=True
        )
        s_tid, s_tood = res_tid["scores"], res_tood["scores"]
        if len(s_tid) == 0 or len(s_tood) == 0:
            log_fn(f"[OOD] {atype}: empty scores — skip")
            continue

        scores_test, y_test = balanced_binary_scores_labels(s_tid, s_tood)
        if len(y_test) == 0 or len(np.unique(y_test)) < 2:
            log_fn(f"[OOD] {atype}: insufficient balanced test data — skip")
            continue

        rm = rank_metrics(y_test, scores_test)
        entry = {
            "auroc": rm["auroc"],
            "aupr": rm["aupr"],
            "n_test_normal_balanced": int((y_test == 0).sum()),
            "n_test_ood_balanced": int((y_test == 1).sum()),
            "n_test_normal_full": int(len(s_tid)),
            "n_test_ood_full": int(len(s_tood)),
            "f1": None,
            "precision": None,
            "recall": None,
            "accuracy": None,
            "best_threshold": None,
            "val_f1_tune": None,
            "n_val_normal_balanced": None,
            "n_val_ood_balanced": None,
            "n_val_normal_full": None,
            "n_val_ood_full": None,
        }

        valid_ood_sub = f"valid_ood_{atype}"
        valid_ood_path = os.path.join(base, valid_ood_sub)
        f1_note = None
        if has_valid_root and os.path.isdir(valid_ood_path):
            ld_vid = _make_ood_loader(base, "valid_id", args.batch, args.max_seq_len, pin)
            ld_vood = _make_ood_loader(
                base, valid_ood_sub, args.batch, args.max_seq_len, pin
            )
            if ld_vid is not None and ld_vood is not None:
                res_vid = evaluate_split(
                    model, ld_vid, device, args, crit_e, crit_l, return_scores=True
                )
                res_vood = evaluate_split(
                    model, ld_vood, device, args, crit_e, crit_l, return_scores=True
                )
                s_vid, s_vood = res_vid["scores"], res_vood["scores"]
                scores_val, y_val = balanced_binary_scores_labels(s_vid, s_vood)
                if len(y_val) > 0 and len(np.unique(y_val)) >= 2:
                    best_t, val_f1 = tune_threshold_max_f1(
                        scores_val, y_val, args.ood_threshold_grid
                    )
                    mt = metrics_at_threshold(y_test, scores_test, best_t)
                    entry.update(
                        {
                            "best_threshold": best_t,
                            "val_f1_tune": val_f1,
                            "f1": mt["f1"],
                            "precision": mt["precision"],
                            "recall": mt["recall"],
                            "accuracy": mt["accuracy"],
                            "n_val_normal_balanced": int((y_val == 0).sum()),
                            "n_val_ood_balanced": int((y_val == 1).sum()),
                            "n_val_normal_full": int(len(s_vid)),
                            "n_val_ood_full": int(len(s_vood)),
                        }
                    )
                else:
                    f1_note = "validation_balanced_set_invalid"
                    log_fn(
                        f"[OOD] {atype}: valid_id + {valid_ood_sub} could not be "
                        "balanced or are single-class — F1 skipped"
                    )
            else:
                f1_note = "validation_loader_missing"
                log_fn(
                    f"[OOD] {atype}: could not load valid_id / {valid_ood_sub} — F1 skipped"
                )
        else:
            f1_note = "validation_dirs_missing"
            log_fn(
                f"[OOD] {atype}: valid_id or {valid_ood_sub} missing — "
                "F1 skipped (AUROC/AUPR on balanced test only)"
            )

        if f1_note:
            entry["f1_note"] = f1_note

        msg = (
            f"[OOD] {atype:6s}  AUROC={rm['auroc']:.4f}  AUPR={rm['aupr']:.4f}  "
            f"bal={entry['n_test_normal_balanced']}/{entry['n_test_ood_balanced']}  "
            f"full={entry['n_test_normal_full']:,}/{entry['n_test_ood_full']:,}"
        )
        if entry["f1"] is not None:
            msg += (
                f"  F1={entry['f1']:.4f}  P={entry['precision']:.4f}  "
                f"R={entry['recall']:.4f}  thr={entry['best_threshold']:.6g}"
            )
        log_fn(msg)
        results[atype] = entry

    return results


def run_ood_eval(model, args, device, crit_e, crit_l, log_fn):
    if not HAS_SKLEARN:
        log_fn("[OOD] sklearn not available - skipping")
        return {}

    pin = device.type == "cuda"
    base = args.preprocessed_dir
    if not os.path.isdir(os.path.join(base, "test_id")):
        log_fn("[OOD] test_id not found")
        return {}

    valid_id_dir = os.path.join(base, "valid_id")
    has_valid_root = os.path.isdir(valid_id_dir)

    ld_tid = _make_ood_loader(base, "test_id", args.batch, args.max_seq_len, pin)
    if ld_tid is None:
        log_fn("[OOD] test_id loader unavailable")
        return {}
    res_tid = evaluate_split(
        model, ld_tid, device, args, crit_e, crit_l, return_scores=True
    )

    res_vid = None
    if has_valid_root:
        ld_vid = _make_ood_loader(base, "valid_id", args.batch, args.max_seq_len, pin)
        if ld_vid is not None:
            res_vid = evaluate_split(
                model, ld_vid, device, args, crit_e, crit_l, return_scores=True
            )

    valid_ood_results = {}
    test_ood_results = {}
    available_types = []
    for atype in ["cpu", "disk", "mem", "net"]:
        test_ood_sub = f"test_ood_{atype}"
        test_loader = _make_ood_loader(base, test_ood_sub, args.batch, args.max_seq_len, pin)
        if test_loader is None:
            continue
        test_ood_results[atype] = evaluate_split(
            model, test_loader, device, args, crit_e, crit_l, return_scores=True
        )
        if has_valid_root:
            valid_ood_sub = f"valid_ood_{atype}"
            valid_loader = _make_ood_loader(
                base, valid_ood_sub, args.batch, args.max_seq_len, pin
            )
            if valid_loader is not None:
                valid_ood_results[atype] = evaluate_split(
                    model, valid_loader, device, args, crit_e, crit_l, return_scores=True
                )
        available_types.append(atype)

    if not available_types:
        log_fn("[OOD] no test OOD splits found")
        return {}

    mad_event = None
    mad_latency = None
    global_best_t = None
    global_val_f1 = None
    global_val_counts = {
        "n_val_normal_balanced": None,
        "n_val_ood_balanced": None,
        "n_val_normal_full": None,
        "n_val_ood_full": None,
    }
    score_method = (
        "paper_mad_sum" if (args.train_event_model and args.train_latency_model) else
        "event_loss" if args.train_event_model else
        "latency_loss"
    )

    if res_vid is not None and valid_ood_results:
        if args.train_event_model and args.train_latency_model:
            mad_event = _compute_mad_stats(res_vid["scores_event"])
            mad_latency = _compute_mad_stats(res_vid["scores_latency"])

        val_id_scores = combine_paper_ood_scores(
            res_vid["scores_event"],
            res_vid["scores_latency"],
            args,
            mad_event,
            mad_latency,
        )
        val_ood_scores_all = []
        for atype in available_types:
            res_vood = valid_ood_results.get(atype)
            if res_vood is None:
                continue
            s_vood = combine_paper_ood_scores(
                res_vood["scores_event"],
                res_vood["scores_latency"],
                args,
                mad_event,
                mad_latency,
            )
            if s_vood.size > 0:
                val_ood_scores_all.append(s_vood)

        if val_id_scores.size > 0 and val_ood_scores_all:
            val_ood_scores = np.concatenate(val_ood_scores_all)
            scores_val, y_val = combine_balanced_binary_scores_labels(
                val_id_scores, val_ood_scores, seed=args.seed
            )
            if len(y_val) > 0 and len(np.unique(y_val)) >= 2:
                global_best_t, global_val_f1 = tune_threshold_max_f1(
                    scores_val, y_val, args.ood_threshold_grid
                )
                global_val_counts = {
                    "n_val_normal_balanced": int((y_val == 0).sum()),
                    "n_val_ood_balanced": int((y_val == 1).sum()),
                    "n_val_normal_full": int(len(val_id_scores)),
                    "n_val_ood_full": int(len(val_ood_scores)),
                }
                log_fn(
                    f"[OOD] global validation threshold ({score_method}) "
                    f"thr={global_best_t:.6g} F1={global_val_f1:.4f} "
                    f"bal={global_val_counts['n_val_normal_balanced']}/"
                    f"{global_val_counts['n_val_ood_balanced']} "
                    f"full={global_val_counts['n_val_normal_full']:,}/"
                    f"{global_val_counts['n_val_ood_full']:,}"
                )
                if mad_event is not None and mad_latency is not None:
                    log_fn(
                        f"[OOD] MAD stats event(median={mad_event['median']:.6g}, "
                        f"mad={mad_event['mad']:.6g}) latency(median="
                        f"{mad_latency['median']:.6g}, mad={mad_latency['mad']:.6g})"
                    )
            else:
                log_fn("[OOD] global validation set invalid for threshold tuning")
        else:
            log_fn("[OOD] validation scores unavailable for global threshold tuning")
    else:
        log_fn("[OOD] valid_id or valid_ood_* missing; threshold-based metrics skipped")

    results = {}
    test_id_scores = combine_paper_ood_scores(
        res_tid["scores_event"],
        res_tid["scores_latency"],
        args,
        mad_event,
        mad_latency,
    )

    for atype in available_types:
        res_tood = test_ood_results[atype]
        s_tood = combine_paper_ood_scores(
            res_tood["scores_event"],
            res_tood["scores_latency"],
            args,
            mad_event,
            mad_latency,
        )
        if test_id_scores.size == 0 or s_tood.size == 0:
            log_fn(f"[OOD] {atype}: empty scores - skip")
            continue

        scores_test, y_test = combine_full_binary_scores_labels(
            test_id_scores, s_tood
        )
        if len(y_test) == 0 or len(np.unique(y_test)) < 2:
            log_fn(f"[OOD] {atype}: insufficient test data - skip")
            continue

        rm = rank_metrics(y_test, scores_test)
        entry = {
            "score_method": score_method,
            "auroc": rm["auroc"],
            "aupr": rm["aupr"],
            "n_test_normal_balanced": int((y_test == 0).sum()),
            "n_test_ood_balanced": int((y_test == 1).sum()),
            "n_test_normal_full": int(len(test_id_scores)),
            "n_test_ood_full": int(len(s_tood)),
            "f1": None,
            "precision": None,
            "recall": None,
            "accuracy": None,
            "best_threshold": global_best_t,
            "val_f1_tune": global_val_f1,
            **global_val_counts,
        }

        if mad_event is not None:
            entry["mad_event_median"] = mad_event["median"]
            entry["mad_event"] = mad_event["mad"]
        if mad_latency is not None:
            entry["mad_latency_median"] = mad_latency["median"]
            entry["mad_latency"] = mad_latency["mad"]

        if global_best_t is not None:
            mt = metrics_at_threshold(y_test, scores_test, global_best_t)
            entry.update(
                {
                    "f1": mt["f1"],
                    "precision": mt["precision"],
                    "recall": mt["recall"],
                    "accuracy": mt["accuracy"],
                }
            )
        else:
            entry["f1_note"] = "global_validation_threshold_unavailable"

        msg = (
            f"[OOD] {atype:6s}  AUROC={rm['auroc']:.4f}  AUPR={rm['aupr']:.4f}  "
            f"bal={entry['n_test_normal_balanced']}/{entry['n_test_ood_balanced']}  "
            f"full={entry['n_test_normal_full']:,}/{entry['n_test_ood_full']:,}"
        )
        if entry["f1"] is not None:
            msg += (
                f"  F1={entry['f1']:.4f}  P={entry['precision']:.4f}  "
                f"R={entry['recall']:.4f}  thr={entry['best_threshold']:.6g}"
            )
        log_fn(msg)
        results[atype] = entry

    return results


###############################################################################
# Main
###############################################################################

def main():
    args = get_args()

    # ── DDP / single-GPU setup ─────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    ddp_enabled = local_rank >= 0

    if ddp_enabled:
        dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()
        rank       = dist.get_rank()
        device     = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank       = 0
        world_size = 1
        device     = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    is_main = (rank == 0)   # only rank 0 logs, saves, and writes WandB

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    if is_main:
        os.makedirs(args.log_dir, exist_ok=True)
    if ddp_enabled:
        dist.barrier()   # wait for rank 0 to create log_dir

    log_file = open(os.path.join(args.log_dir, f"train_rank{rank}.log"), "a", buffering=1)

    def log(msg):
        line = f"[{datetime.now():%H:%M:%S}][rank{rank}] {msg}"
        if is_main:
            print(line, flush=True)
            log_file.write(line + "\n")
            log_file.flush()
        elif "ERROR" in msg or "WARN" in msg:
            # non-main ranks only print errors
            print(line, flush=True)

    use_wandb = bool(args.wandb_project) and HAS_WANDB and is_main
    logger    = Logger(args.log_dir, use_wandb, args) if is_main else None

    # Vocab
    with open(os.path.join(args.preprocessed_dir, "vocab.pkl"), "rb") as f:
        dict_sys, dict_proc = pickle.load(f)
    n_syscall = len(dict_sys)
    n_process = len(dict_proc)
    log(f"Vocab: {n_syscall} syscalls / {n_process} processes")
    if ddp_enabled:
        log(f"DDP: world_size={world_size}  local_rank={local_rank}  device={device}")

    # Datasets + loaders
    def make_loader(split, shuffle=True, workers=None):
        d = os.path.join(args.preprocessed_dir, split)
        if not os.path.isdir(d):
            return None, None
        ds = SockshopNpzDataset(d, batch_size=args.batch,
                                 max_seq_len=args.max_seq_len,
                                 max_samples=args.max_samples,
                                 shuffle_shards=shuffle)
        ld = DataLoader(ds, batch_size=None, collate_fn=sockshop_collate_fn,
                        num_workers=workers if workers is not None else args.num_workers,
                        pin_memory=device.type=="cuda",
                        prefetch_factor=2 if (workers or args.num_workers) > 0 else None,
                        persistent_workers=(workers or args.num_workers) > 0)
        return ds, ld

    train_ds, train_loader = make_loader("train_id",  shuffle=True)
    valid_ds, valid_loader = make_loader("valid_id",  shuffle=False)

    # Assign DDP rank so each GPU reads disjoint shards
    if ddp_enabled:
        train_ds.rank       = rank
        train_ds.world_size = world_size
        if valid_ds is not None:
            valid_ds.rank       = rank
            valid_ds.world_size = world_size

    log(f"Train shards for this rank: {len(train_ds._shards) // world_size}  "
        f"(total={len(train_ds._shards)})  Valid shards: {len(valid_ds._shards) if valid_ds else 0}")

    # Model
    model = build_model(args, n_syscall, n_process, device)
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model, map_location=device))
        log(f"Loaded checkpoint from {args.load_model}")

    # Compile model
    if args.compile and torch.cuda.is_available():
        log("Compiling model with torch.compile ...")
        # Disable CUDAGraphs to prevent the "overwritten by subsequent run" embedding bug
        torch._inductor.config.triton.cudagraphs = False
        model = torch.compile(model, mode="reduce-overhead")

    # Wrap in DDP after compile
    if ddp_enabled:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main:
        log(f"Model: {args.model.upper()}  params={n_params:,}")
        log(f"  n_hidden={args.n_hidden}  n_layer={args.n_layer}  n_head={args.n_head}")
        log(f"  AMP(bf16)={args.amp}  compile={args.compile}  chk={args.chk}  DDP={ddp_enabled}")
        log(f"  batch={args.batch}  accum_steps={args.accum_steps}  world_size={world_size}  "
            f"eff_batch={args.batch * args.accum_steps * world_size}")

    # Loss criteria
    crit_e = (nn.CrossEntropyLoss(ignore_index=0, label_smoothing=args.label_smoothing)
              if args.train_event_model else None)
    crit_l = ((nn.BCEWithLogitsLoss() if args.ordinal_latency
               else nn.CrossEntropyLoss(ignore_index=0))
              if args.train_latency_model else None)
    _crit_e = crit_e or nn.CrossEntropyLoss(ignore_index=0)
    _crit_l = crit_l or nn.CrossEntropyLoss(ignore_index=0)

    # Optimizer + schedule
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01)
    steps_per_epoch = len(train_ds)
    total_steps     = steps_per_epoch * args.n_epochs // max(args.accum_steps, 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: lr_lambda(s, args.warmup_steps, total_steps))
    scaler = torch.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    with open(os.path.join(args.log_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    if ddp_enabled:
        dist.barrier()   # all ranks ready before training

    log("=" * 70)
    log(f"Starting training  device={device}  total_steps~={total_steps}  world_size={world_size}")
    log("=" * 70)

    global_step   = 0
    best_val_loss = float("inf")
    no_improve_evals = 0
    early_stop = False
    t_start       = time.time()

    for epoch in range(1, args.n_epochs + 1):
        train_ds.set_epoch(epoch)
        model.train()

        epoch_loss = 0.0
        acc_e_sum = acc_e_cnt = 0.0
        acc_l_sum = acc_l_cnt = 0.0
        n_batches  = 0
        valid_loss_batches = 0
        optimizer.zero_grad(set_to_none=True)

        pbar = (tqdm(train_loader, desc=f"Ep{epoch:02d}/{args.n_epochs}",
                     dynamic_ncols=True, leave=True)
                if HAS_TQDM else train_loader)

        for batch in pbar:
            is_optim_step = ((n_batches + 1) % args.accum_steps == 0)

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=args.amp and device.type == "cuda"):
                logits_e, logits_l = forward_batch(model, batch, device, args)
                loss, loss_e, loss_l = compute_loss(
                    logits_e, logits_l, batch, device, args, crit_e, crit_l)
                loss = loss / args.accum_steps

            scaler.scale(loss).backward()

            if is_optim_step:
                scaler.unscale_(optimizer)
                if args.clip:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            raw_loss = loss.item() * args.accum_steps
            if not math.isnan(raw_loss):
                epoch_loss += raw_loss
                valid_loss_batches += 1
            tgt_call = batch["tgt_call"].to(device, dtype=torch.long, non_blocking=True)
            tgt_lat  = batch["tgt_lat" ].to(device, dtype=torch.long, non_blocking=True)
            if args.train_event_model and logits_e.numel() > 0:
                v = token_accuracy(logits_e.detach(), tgt_call)
                if not math.isnan(v):
                    acc_e_sum += v; acc_e_cnt += 1
            if args.train_latency_model and logits_l.numel() > 0:
                v = token_accuracy(logits_l.detach(), tgt_lat)
                if not math.isnan(v):
                    acc_l_sum += v; acc_l_cnt += 1
            n_batches += 1

            epoch_acc_e = acc_e_sum / max(acc_e_cnt, 1)
            epoch_acc_l = acc_l_sum / max(acc_l_cnt, 1)

            cur_lr = scheduler.get_last_lr()[0]
            if HAS_TQDM:
                pbar.set_postfix(
                    loss=f"{raw_loss:.3f}",
                    acc_e=f"{epoch_acc_e:.2%}" if args.train_event_model else "-",
                    lr=f"{cur_lr:.2e}",
                    step=global_step)

            # Log every 100 optimizer steps (rank 0 only)
            if is_main and is_optim_step and global_step % 100 == 0:
                metrics = {
                    "train/loss":  epoch_loss / max(valid_loss_batches, 1),
                    "train/acc_e": epoch_acc_e,
                    "train/acc_l": epoch_acc_l,
                    "train/lr":    cur_lr,
                    "train/epoch": epoch,
                }
                if logger: logger.log(metrics, step=global_step)
                if global_step % 500 == 0:
                    elapsed = timedelta(seconds=int(time.time() - t_start))
                    log(f"step={global_step:6d}  epoch={epoch}  "
                        f"loss={epoch_loss/max(valid_loss_batches, 1):.4f}  "
                        f"acc_e={epoch_acc_e:.2%}  "
                        f"lr={cur_lr:.2e}  elapsed={elapsed}")

            # Validation — run on all ranks, but only log from rank 0
            if (valid_loader and is_optim_step and
                    global_step > 0 and global_step % args.eval_every == 0):
                # Sync before eval so all ranks use same model state
                if ddp_enabled: dist.barrier()
                if is_main:
                    log(f"--- Validation @ step {global_step} ---")
                    res = evaluate_split(model, valid_loader, device, args, _crit_e, _crit_l)
                    log(f"  val loss={res['loss']:.4f}  "
                        f"loss_e={res['loss_e']:.4f}  acc_e={res['acc_e']:.2%}  "
                        f"acc_l={res['acc_l']:.2%}")
                    if logger: logger.log({f"val/{k}": v for k, v in res.items()
                                 if not isinstance(v, np.ndarray)},
                                step=global_step)
                    if res["loss"] < best_val_loss:
                        best_val_loss = res["loss"]
                        no_improve_evals = 0
                        best_path = os.path.join(args.log_dir, "model_best.pt")
                        raw = (model.module if ddp_enabled else
                               model._orig_mod if hasattr(model, "_orig_mod") else model)
                        if logger: logger.save_model(raw, best_path)
                        log(f"  New best val loss={best_val_loss:.4f} -> {best_path}")
                    elif args.early_stopping_patience > 0:
                        no_improve_evals += 1
                        log(f"  No improvement count: {no_improve_evals}/"
                            f"{args.early_stopping_patience}")
                        if no_improve_evals >= args.early_stopping_patience:
                            early_stop = True
                            log(f"  Early stopping triggered after "
                                f"{no_improve_evals} validations without improvement.")
                if ddp_enabled:
                    stop_flag = torch.tensor(
                        1 if (is_main and early_stop) else 0,
                        device=device,
                        dtype=torch.int32,
                    )
                    dist.broadcast(stop_flag, src=0)
                    early_stop = bool(stop_flag.item())
                if ddp_enabled: dist.barrier()
                model.train()
                if early_stop:
                    break

            # Periodic checkpoint (rank 0 only)
            if (is_main and is_optim_step and global_step > 0 and
                    global_step % args.save_every == 0):
                ckpt = os.path.join(args.log_dir, f"ckpt_{global_step:07d}.pt")
                raw  = (model.module if ddp_enabled else
                        model._orig_mod if hasattr(model, "_orig_mod") else model)
                if logger: logger.save_model(raw, ckpt)
                log(f"  Checkpoint -> {ckpt}")

        if early_stop:
            elapsed = timedelta(seconds=int(time.time() - t_start))
            log(f"Stopping early at epoch {epoch}/{args.n_epochs}  elapsed={elapsed}")
            break

        # End of epoch
        elapsed = timedelta(seconds=int(time.time() - t_start))
        log(f"Epoch {epoch:3d}/{args.n_epochs}  "
            f"loss={epoch_loss/max(valid_loss_batches,1):.4f}  "
            f"acc_e={epoch_acc_e:.2%}  "
            f"acc_l={epoch_acc_l:.2%}  "
            f"elapsed={elapsed}")
        if is_main:
            raw = (model.module if ddp_enabled else
                   model._orig_mod if hasattr(model, "_orig_mod") else model)
            if logger: logger.save_model(raw, os.path.join(args.log_dir, f"ckpt_epoch{epoch:03d}.pt"))

    # ── OOD eval + final save (rank 0 only) ─────────────────────────────────
    if is_main:
        log("=" * 70)
        log("Training complete — OOD evaluation")
        log("=" * 70)
        raw = (model.module if ddp_enabled else
               model._orig_mod if hasattr(model, "_orig_mod") else model)
        best_path = os.path.join(args.log_dir, "model_best.pt")
        if os.path.isfile(best_path):
            raw.load_state_dict(torch.load(best_path, map_location=device))
            log(f"Loaded best model from {best_path}")

        ood_results = run_ood_eval(raw, args, device, _crit_e, _crit_l, log)

        def _json_safe(o):
            if isinstance(o, dict):
                return {k: _json_safe(v) for k, v in o.items()}
            if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
                return None
            return o

        ood_path = os.path.join(args.log_dir, "ood_results.json")
        with open(ood_path, "w") as f:
            json.dump(_json_safe(ood_results), f, indent=2)
        log(f"OOD results -> {ood_path}")

        if use_wandb and ood_results:
            wb_flat = {}
            for at, metrics in ood_results.items():
                for m, v in metrics.items():
                    if v is None:
                        continue
                    if isinstance(v, str):
                        wb_flat[f"ood/{at}/{m}"] = v
                    elif isinstance(v, bool):
                        wb_flat[f"ood/{at}/{m}"] = int(v)
                    elif isinstance(v, (int, float)):
                        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                            continue
                        wb_flat[f"ood/{at}/{m}"] = v
            if wb_flat:
                wandb.log(wb_flat)

        if logger: logger.close()
        log_file.close()

    if ddp_enabled:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
