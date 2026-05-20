#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metrics.py
────────────────────────────────────────────────────────────────────────────
Compute NDCG@10 / MRR@10 / Recall@10 for a single checkpoint's label-pool
(zero-fill and ignore-list variants), outputting:

  per-event CSV  (metrics_zero_per_event.csv, metrics_ignore_per_event.csv)
  per-task  CSV  (ndcg_per_task.csv)
  overall   TXT  (ndcg_overall.txt)
  retrieval stats (retrieval_stats.json)

Usage:
    from eva.evaluation.metrics import calc_metrics
    calc_metrics(model="PT/.../checkpoint-1000", parent="only_raw")

    python metrics.py --model PT/.../checkpoint-1000 --parent only_raw
"""

# ─── enable relative imports when running as a script ─────────────────────
from __future__ import annotations

import os, sys

if __name__ == "__main__" and __package__ is None:
    here = os.path.abspath(os.path.dirname(__file__))
    pkgroot = os.path.abspath(os.path.join(here, ".."))   # …/eva
    sys.path.insert(0, pkgroot)
    __package__ = "eva.evaluation"
# ───────────────────────────────────────────────────────────────────────────

import json
import math
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd

from eva.utils.config import (
    DEFAULT_TOPK as K,
    LABEL_POOL_DIR, QRELS_DIR, PERF_DIR,
)

LOGGER = logging.getLogger(__name__)

_LABEL_ROOT = Path(LABEL_POOL_DIR)
_QRELS_ROOT = Path(QRELS_DIR)
_PERF_ROOT  = Path(PERF_DIR)

 
# ─── Low-level computation utilities ──────────────────────────────────────
def _dcg(rels: List[int]) -> float:
    """Discounted Cumulative Gain (DCG)"""
    return sum((2 ** r - 1) / math.log2(idx + 2) for idx, r in enumerate(rels))


def _metrics_at_k(retrieved: List[int], all_rels: List[int], k: int = K) -> Tuple[float, float, float]:
    """NDCG, RR, Recall@k"""
    retrieved_k = retrieved[:k]
    ideal_k = sorted(all_rels, reverse=True)[:k]

    dcg = _dcg(retrieved_k)
    idcg = _dcg(ideal_k)
    ndcg = dcg / idcg if idcg else 0.0

    rr = next((1.0 / (rank + 1) for rank, r in enumerate(retrieved_k) if r > 0), 0.0)

    rel_total = sum(1 for r in all_rels if r > 0)
    rel_ret   = sum(1 for r in retrieved_k if r > 0)
    recall    = rel_ret / rel_total if rel_total else 0.0

    return ndcg, rr, recall


# ─── Main computation function ────────────────────────────────────────────
def calc_metrics(model: str, parent: str) -> Path:
    """
    Read retrieval results from LABEL_POOL_DIR/<slug>, compute metrics,
    and write output to performance/<parent>/<ckpt_name>/. Returns the output directory Path.

    Parameters
    ----------
    model  : str
        Checkpoint subdirectory (same relative path used by build_query_emb / exact_search).
    parent : str
        Parent directory name for grouping output.

    Returns
    -------
    Path
        Output directory (performance/<parent>/<ckpt_name>/)
    """
    slug = model.replace("/", "_")
    label_dir = _LABEL_ROOT / slug
    if not label_dir.is_dir():
        raise FileNotFoundError(f"label‑pool directory not found: {label_dir}")

    ckpt_name = slug.split("__")[-1]             # keep consistent with legacy
    out_dir   = _PERF_ROOT / parent / ckpt_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------- load qrels -------------------------------------------------
    qrels_cache: Dict[Tuple[str, str], Dict[str, Dict[str, int]]] = {}
    for fp in _QRELS_ROOT.glob("*_qrels.json"):
        task, *event_parts, _ = fp.stem.split("_")
        event_name = "_".join(event_parts)
        key = (task, event_name)
        arr = json.loads(fp.read_text(encoding="utf-8"))
        per_q = defaultdict(dict)
        for item in arr:
            per_q[item["user_query"]][item["passage"]] = item["score"]
        qrels_cache[key] = per_q

    # ----------- iterate over label-pool files ------------------------------
    rows_zero, rows_ignore = [], []
    total_pairs = unannotated = 0

    for fp in label_dir.glob("*.json"):
        task, event = fp.stem.split("_", 1)
        key = (task, event.replace("_label_pool", "")) if fp.stem.endswith("_label_pool") else (task, event)
        if key not in qrels_cache:
            LOGGER.debug("skip %s – no qrels", fp.name)
            continue
        qrels = qrels_cache[key]
        sums_z = dict(ndcg=0.0, rr=0.0, rec=0.0, nq=0)
        sums_i = dict(ndcg=0.0, rr=0.0, rec=0.0, nq=0)

        data = json.loads(fp.read_text(encoding="utf-8"))
        for item in data:
            q = item["user_query"]
            retrieved_passages = item["passages"][:K]
            total_pairs += len(retrieved_passages)

            all_rels = list(qrels.get(q, {}).values())
            if not all_rels:
                continue

            # unannotated stats
            for p in retrieved_passages:
                if p not in qrels[q]:
                    unannotated += 1

            # zero‑fill
            rels_z = [qrels.get(q, {}).get(p, 0) for p in retrieved_passages]
            n_z, r_z, rec_z = _metrics_at_k(rels_z, all_rels, K)
            sums_z["ndcg"] += n_z; sums_z["rr"] += r_z; sums_z["rec"] += rec_z; sums_z["nq"] += 1

            # ignore‑list
            rels_i = [qrels[q][p] for p in retrieved_passages if p in qrels[q]]
            n_i, r_i, rec_i = _metrics_at_k(rels_i, all_rels, K)
            sums_i["ndcg"] += n_i; sums_i["rr"] += r_i; sums_i["rec"] += rec_i; sums_i["nq"] += 1

        if sums_z["nq"]:
            rows_zero.append(
                dict(
                    model=slug,
                    search_task=task,
                    event_type=event,
                    num_queries=sums_z["nq"],
                    **{k.upper() + "@10": sums_z[k] / sums_z["nq"] for k in ("ndcg", "rr", "rec")},
                )
            )
        if sums_i["nq"]:
            rows_ignore.append(
                dict(
                    model=slug,
                    search_task=task,
                    event_type=event,
                    num_queries=sums_i["nq"],
                    **{k.upper() + "@10": sums_i[k] / sums_i["nq"] for k in ("ndcg", "rr", "rec")},
                )
            )

    # ----------- write CSV / TXT / JSON -------------------------------------
    df_z = pd.DataFrame(rows_zero).sort_values(["search_task", "event_type"])
    df_i = pd.DataFrame(rows_ignore).sort_values(["search_task", "event_type"])

    path_zero   = out_dir / "metrics_zero_per_event.csv"
    path_ignore = out_dir / "metrics_ignore_per_event.csv"
    df_z.to_csv(path_zero, index=False)
    df_i.to_csv(path_ignore, index=False)
    LOGGER.info("saved per‑event metrics ➜ %s  %s", path_zero, path_ignore)

    # per-task mean NDCG@10
    per_task = df_z.groupby("search_task")["NDCG@10"].mean().reset_index()
    path_task = out_dir / "ndcg_per_task.csv"
    per_task.to_csv(path_task, index=False)

    # overall
    overall_mean = per_task["NDCG@10"].mean() if not per_task.empty else 0.0
    path_overall = out_dir / "ndcg_overall.txt"
    path_overall.write_text(f"Overall mean NDCG@10: {overall_mean:.4f}\n", encoding="utf-8")
    LOGGER.info("overall mean NDCG@10 = %.4f", overall_mean)

    # un‑annotated stats
    stats = dict(
        total_retrieved_pairs=total_pairs,
        unannotated_pairs=unannotated,
        unannotated_ratio=unannotated / total_pairs if total_pairs else 0.0,
    )
    path_stats = out_dir / "retrieval_stats.json"
    path_stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    LOGGER.info("saved all metrics ➜ %s", out_dir)
    return out_dir


# ─── CLI wrapper ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Calculate IR metrics for one checkpoint")
    parser.add_argument("--model",  required=True, help="checkpoint relative path, matching the label-pool slug")
    parser.add_argument("--parent", required=True, help="parent directory name (performance/<parent>/...)")
    args = parser.parse_args()

    calc_metrics(args.model, args.parent)
