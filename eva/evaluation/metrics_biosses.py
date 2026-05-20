#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metrics_biosses.py
────────────────────────────────────────────────────────────────────────────
Compute NDCG@10 / MRR@10 / Recall@10 for BIOSSES (merged.jsonl):
- query:  text_1
- corpus: all text_2
- qrels:  label (float, 0.0~5.0), keyed by (query_id, document_id)

Output:
  per-event CSV  (metrics_zero_per_event.csv, metrics_ignore_per_event.csv)
  per-task  CSV  (ndcg_per_task.csv)  -- task fixed to "STS"
  overall   TXT  (ndcg_overall.txt)
  retrieval stats (retrieval_stats.json)
"""

from __future__ import annotations
import json
import math
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd

from eva.utils.config import DEFAULT_TOPK as K, LABEL_POOL_DIR, PERF_DIR, RAW_DATA_DIR

LOGGER = logging.getLogger(__name__)

_LABEL_ROOT = Path(LABEL_POOL_DIR)
_PERF_ROOT  = Path(PERF_DIR)
_DATA_FP    = Path(RAW_DATA_DIR) / "biosses" / "merged.jsonl"

def _dcg(rels: List[float]) -> float:
    return sum((2 ** r - 1) / math.log2(idx + 2) for idx, r in enumerate(rels))

def _metrics_at_k(retrieved: List[float], all_rels: List[float], k: int = K) -> Tuple[float, float, float]:
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

def _load_json_any(fp: Path):
    txt = fp.read_text(encoding="utf-8")
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        items = []
        for ln in txt.splitlines():
            s = ln.strip()
            if not s:
                continue
            items.append(json.loads(s))
        return items

def _build_qrels_from_biosses(src_fp: Path) -> Dict[str, Dict[str, float]]:
    """
    Returns: qrels[qid][docid] = label (float)
    """
    arr = _load_json_any(src_fp)
    qrels: Dict[str, Dict[str, float]] = defaultdict(dict)
    for it in arr:
        if not isinstance(it, dict):
            continue
        qid   = str(it.get("id", "")).strip()
        did   = str(it.get("document_id", "")).strip()
        label_raw = it.get("label", 0)
        try:
            rel = float(label_raw)
        except Exception:
            rel = 0.0
        if qid and did:
            qrels[qid][did] = rel
    return qrels

def calc_metrics_biosses(model: str, parent: str) -> Path:
    """
    Read retrieval results from label_pools/<slug>/STS_biosses_label_pool.json,
    compute metrics, and write output to performance/<parent>/<ckpt_name>/.
    """
    slug = model.replace("/", "_")
    label_dir = _LABEL_ROOT / slug
    if not label_dir.is_dir():
        raise FileNotFoundError(f"label-pool directory not found: {label_dir}")

    ckpt_name = slug.split("__")[-1]
    out_dir   = _PERF_ROOT / parent / ckpt_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # qrels (built from source data)
    qrels: Dict[str, Dict[str, float]] = _build_qrels_from_biosses(_DATA_FP)

    # only process STS_biosses_label_pool.json
    lp_fp = label_dir / "STS_biosses_label_pool.json"
    if not lp_fp.exists():
        raise FileNotFoundError(f"label-pool not found: {lp_fp}")

    rows_zero, rows_ignore = [], []
    total_pairs = unannotated = 0

    data = json.loads(lp_fp.read_text(encoding="utf-8"))
    sums_z = dict(ndcg=0.0, rr=0.0, rec=0.0, nq=0)
    sums_i = dict(ndcg=0.0, rr=0.0, rec=0.0, nq=0)

    for item in data:
        qid = str(item.get("query_id","")).strip()
        doc_ids = item.get("doc_ids") or []   # written by exact_search
        total_pairs += len(doc_ids)

        if not qid or qid not in qrels:
            continue
        all_rels = list(qrels[qid].values())
        if not all_rels:
            continue

        # unannotated stats
        for did in doc_ids:
            if did not in qrels[qid]:
                unannotated += 1

        # zero-fill
        rels_z: List[float] = [qrels.get(qid, {}).get(did, 0.0) for did in doc_ids]
        n_z, r_z, rec_z = _metrics_at_k(rels_z, all_rels, K)
        sums_z["ndcg"] += n_z; sums_z["rr"] += r_z; sums_z["rec"] += rec_z; sums_z["nq"] += 1

        # ignore-list
        rels_i: List[float] = [qrels[qid][did] for did in doc_ids if did in qrels[qid]]
        n_i, r_i, rec_i = _metrics_at_k(rels_i, all_rels, K)
        sums_i["ndcg"] += n_i; sums_i["rr"] += r_i; sums_i["rec"] += rec_i; sums_i["nq"] += 1

    # per-event rows (task fixed to STS / event: biosses)
    if sums_z["nq"]:
        rows_zero.append(
            dict(
                model=slug,
                search_task="STS",
                event_type="biosses",
                num_queries=sums_z["nq"],
                **{k.upper() + "@10": sums_z[k] / sums_z["nq"] for k in ("ndcg", "rr", "rec")},
            )
        )
    if sums_i["nq"]:
        rows_ignore.append(
            dict(
                model=slug,
                search_task="STS",
                event_type="biosses",
                num_queries=sums_i["nq"],
                **{k.upper() + "@10": sums_i[k] / sums_i["nq"] for k in ("ndcg", "rr", "rec")},
            )
        )

    # ----------- write CSV / TXT / JSON -------------------------------------
    df_z = pd.DataFrame(rows_zero).sort_values(["search_task", "event_type"]) if rows_zero else pd.DataFrame()
    df_i = pd.DataFrame(rows_ignore).sort_values(["search_task", "event_type"]) if rows_ignore else pd.DataFrame()

    path_zero   = out_dir / "metrics_zero_per_event.csv"
    path_ignore = out_dir / "metrics_ignore_per_event.csv"
    df_z.to_csv(path_zero, index=False)
    df_i.to_csv(path_ignore, index=False)
    LOGGER.info("saved per-event metrics ➜ %s  %s", path_zero, path_ignore)

    # per-task mean NDCG@10 (only STS)
    if not df_z.empty:
        per_task = df_z.groupby("search_task")["NDCG@10"].mean().reset_index()
    else:
        per_task = pd.DataFrame([{"search_task": "STS", "NDCG@10": 0.0}])
    path_task = out_dir / "ndcg_per_task.csv"
    per_task.to_csv(path_task, index=False)

    # overall
    overall_mean = per_task["NDCG@10"].mean() if not per_task.empty else 0.0
    path_overall = out_dir / "ndcg_overall.txt"
    path_overall.write_text(f"Overall mean NDCG@10: {overall_mean:.4f}\n", encoding="utf-8")
    LOGGER.info("overall mean NDCG@10 = %.4f", overall_mean)

    # un-annotated stats
    stats = dict(
        total_retrieved_pairs=total_pairs,
        unannotated_pairs=unannotated,
        unannotated_ratio=unannotated / total_pairs if total_pairs else 0.0,
    )
    path_stats = out_dir / "retrieval_stats.json"
    path_stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    LOGGER.info("saved all metrics ➜ %s", out_dir)
    return out_dir


# CLI
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Calculate IR metrics for BIOSSES")
    parser.add_argument("--model",  required=True, help="checkpoint relative path, matching the label-pool slug")
    parser.add_argument("--parent", required=True, help="parent directory name (performance/<parent>/...)")
    args = parser.parse_args()
    calc_metrics_biosses(args.model, args.parent)
