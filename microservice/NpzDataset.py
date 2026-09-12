#!/usr/bin/env python3
"""
SockshopNpzDataset — batch-level NPZ shard loader
===================================================

Yields *pre-padded* NumPy mini-batches directly from each shard, eliminating
per-sample Python overhead and enabling zero-copy pinned-memory transfers to
H100 GPUs.

Design:
  • Each DataLoader worker owns a disjoint slice of shards.
  • A worker reads one shard, chops it into (B, L) chunks, yields them.
  • collate_fn just stacks the already-padded chunks → single tensor cat.
  • pin_memory=True + non_blocking H2D copies hide PCIe latency entirely.

Usage
-----
  ds = SockshopNpzDataset("preprocessed/train_id", batch_size=256)
  loader = DataLoader(ds, batch_size=None,        # batch_size=None: dataset handles batching
                      collate_fn=sockshop_collate_fn,
                      num_workers=4,
                      pin_memory=True,
                      prefetch_factor=2)
  for batch in loader:
      call, entry, duration, proc, pid, tid, ret, lat_cat, tgt_call,
          seq_len, req_dur_ms, is_anomaly, pad_mask = batch
"""

from __future__ import annotations

import os
import json
import glob
import random
import numpy as np
import torch
from torch.utils import data


class SockshopNpzDataset(data.IterableDataset):
    """Stream pre-padded mini-batches from NPZ shards.

    Each yielded item is a dict of NumPy arrays with shape (B, L) or (B,).
    The collate_fn (sockshop_collate_fn) converts them to tensors.
    Setting batch_size=None in DataLoader avoids a second collation pass.
    """

    def __init__(
        self,
        split_dir:       str,
        batch_size:      int   = 256,
        max_seq_len:     int   = 512,
        max_samples:     int | None = None,
        shuffle_shards:  bool  = True,
        epoch_seed:      int   = 0,
    ):
        self.split_dir      = split_dir
        self.batch_size     = batch_size
        self.max_seq_len    = max_seq_len
        self.max_samples    = max_samples
        self.shuffle_shards = shuffle_shards
        self.epoch_seed     = epoch_seed   # bump each epoch for different shard order

        self._shards = sorted(glob.glob(os.path.join(split_dir, "shard_*.npz")))
        if not self._shards:
            raise FileNotFoundError(f"No shard_*.npz found in {split_dir}")

        # DDP: set rank / world_size before creating DataLoader
        self.rank       = 0
        self.world_size = 1

        meta_path = os.path.join(split_dir, "meta.json")
        self.meta = {}
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                self.meta = json.load(f)

    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int):
        """Call at the start of each epoch to vary shard shuffling order."""
        self.epoch_seed = epoch

    def _shards_for_worker(self) -> list[str]:
        """Return the shard subset for this (rank, worker_id) pair."""
        worker_info = data.get_worker_info()
        n_workers   = 1  if worker_info is None else worker_info.num_workers
        worker_id   = 0  if worker_info is None else worker_info.id

        shards = list(self._shards)
        if self.shuffle_shards:
            rng = random.Random(self.epoch_seed * 10007 + self.rank * 997)
            rng.shuffle(shards)

        # Distribute across: DDP rank × DataLoader worker
        global_id    = self.rank * n_workers + worker_id
        global_total = self.world_size * n_workers
        return shards[global_id::global_total]

    # ------------------------------------------------------------------

    def __iter__(self):
        samples_yielded = 0

        for shard_path in self._shards_for_worker():
            shard = np.load(shard_path, allow_pickle=False)

            call_arr  = shard["call"]        # (N, L_full) int32
            entry_arr = shard["entry"]       # (N, L_full) int8
            dur_arr   = shard["duration"]    # (N, L_full) int64
            proc_arr  = shard["proc"]        # (N, L_full) int32
            pid_arr   = shard["pid"]         # (N, L_full) int32
            tid_arr   = shard["tid"]         # (N, L_full) int32
            ret_arr   = shard["ret"]         # (N, L_full) int8
            lat_arr   = shard["lat_cat"]     # (N, L_full) uint8
            seq_lens  = shard["seq_len"]     # (N,)        int32
            req_durs  = shard["req_dur_ms"]  # (N,)        float32
            anom_arr  = shard["is_anomaly"]  # (N,)        int8

            N = len(seq_lens)

            # Clip to max_seq_len in the shard axis
            L = min(call_arr.shape[1], self.max_seq_len)
            call_arr  = call_arr[:, :L]
            entry_arr = entry_arr[:, :L]
            dur_arr   = dur_arr[:, :L]
            proc_arr  = proc_arr[:, :L]
            pid_arr   = pid_arr[:, :L]
            tid_arr   = tid_arr[:, :L]
            ret_arr   = ret_arr[:, :L]
            lat_arr   = lat_arr[:, :L]
            seq_lens  = np.minimum(seq_lens, L)

            # Yield mini-batches
            start = 0
            while start < N:
                if self.max_samples is not None and samples_yielded >= self.max_samples:
                    return

                end = min(start + self.batch_size, N)
                sl  = seq_lens[start:end]       # (B,)
                B   = end - start
                pad_len = int(sl.max())          # trim batch to actual max length

                yield {
                    # input = all tokens except last
                    "call":       call_arr [start:end, :pad_len - 1].astype(np.int32),
                    "entry":      entry_arr[start:end, :pad_len - 1].astype(np.int16),
                    "duration":   dur_arr  [start:end, :pad_len - 1].astype(np.int64),
                    "proc":       proc_arr [start:end, :pad_len - 1].astype(np.int32),
                    "pid":        pid_arr  [start:end, :pad_len - 1].astype(np.int32),
                    "tid":        tid_arr  [start:end, :pad_len - 1].astype(np.int32),
                    "ret":        ret_arr  [start:end, :pad_len - 1].astype(np.int16),
                    "lat_cat":    lat_arr  [start:end, :pad_len - 1].astype(np.int32),
                    # target = shifted by 1
                    "tgt_call":   call_arr [start:end, 1:pad_len   ].astype(np.int32),
                    "tgt_lat":    lat_arr  [start:end, 1:pad_len   ].astype(np.int32),
                    # sequence metadata
                    "seq_len":    (sl - 1),                                # (B,) — input length
                    "req_dur_ms": req_durs[start:end].astype(np.float32),
                    "is_anomaly": anom_arr[start:end].astype(np.int32),
                }
                samples_yielded += B
                start = end

    def __len__(self):
        n_seq  = self.meta.get("n_sequences", 0)
        n_shrd = max(len(self._shards) // self.world_size, 1)
        frac   = n_shrd / max(len(self._shards), 1)
        return max(1, int(n_seq * frac / self.batch_size))


# ---------------------------------------------------------------------------
# collate_fn — minimal: just convert dict-of-arrays → dict-of-tensors
# ---------------------------------------------------------------------------

def sockshop_collate_fn(batch_dicts):
    """Convert a list of batch-dicts (from IterableDataset) to tensors.

    DataLoader always calls this with a list, even when batch_size=None.
    Each element is a dict of NumPy arrays with shapes (B, L) or (B,).
    If multiple dicts arrive (unusual), they are concatenated first.
    """
    # Unwrap: if DataLoader gave us a list containing one dict, grab it
    if not isinstance(batch_dicts, list):
        batch_dicts = [batch_dicts]

    if len(batch_dicts) == 1:
        d = batch_dicts[0]
        # d might itself be a list if something went wrong - be robust
        if isinstance(d, list):
            batch_dicts = d
        else:
            out = {}
            for k, v in d.items():
                if isinstance(v, np.ndarray):
                    out[k] = torch.from_numpy(np.ascontiguousarray(v))
                else:
                    out[k] = torch.tensor(v)
            # Build padding mask: True where position >= real sequence length
            sl  = out["seq_len"]               # (B,)
            L   = out["call"].shape[1]
            idx = torch.arange(L).unsqueeze(0) # (1, L)
            out["pad_mask"] = idx >= sl.unsqueeze(1).long()  # (B, L)
            return out

    # Multi-dict: concatenate along batch dim, then recurse once
    keys = batch_dicts[0].keys()
    merged = {}
    for k in keys:
        arrs = [d[k] for d in batch_dicts]
        if isinstance(arrs[0], np.ndarray) and arrs[0].ndim > 1:
            max_l = max(a.shape[1] for a in arrs)
            padded = [np.pad(a, ((0, 0), (0, max_l - a.shape[1]))) for a in arrs]
            merged[k] = np.concatenate(padded, axis=0)
        elif isinstance(arrs[0], np.ndarray):
            merged[k] = np.concatenate(arrs, axis=0)
        else:
            merged[k] = np.concatenate(arrs, axis=0)
    return sockshop_collate_fn([merged])
