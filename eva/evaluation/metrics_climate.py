#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metrics_climate.py
────────────────────────────────────────────────────────────────────────────
Evaluate Climate-FEVER (qrels-based: query_id ↔ doc_id ↦ score), computing:
  NDCG@10 / MRR@10 / Recall@10 (zero-fill and ignore-list variants)
Output:
  performance/<parent>/<ckpt_name>/, aligned with the main metrics layout
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import json
import math
import logging
import pandas as pd

from eva.utils.config import DEFAULT_TOPK as K, LABEL_POOL_DIR, PERF_DIR, RAW_DATA_DIR
from eva.utils.io_utils import load_json_any

LOGGER = logging.getLogger(__name__)

_LABEL_ROOT = Path(LABEL_POOL_DIR)
_PERF_ROOT  = Path(PERF_DIR)
_CLI_ROOT   = Path(RAW_DATA_DIR) / "climate_fever"
_QRELS_FP   = _CLI_ROOT / "qrels.jsonl"

def _dcg(rels: List[int]) -> float:
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))

def _metrics_at_k(retrieved_rels: List[int], all_rels: List[int], k: int = K) -> Tuple[float, float, float]:
    rk = retrieved_rels[:k]
    ideal_k = sorted(all_rels, reverse=True)[:k]
    dcg = _dcg(rk)
    idcg = _dcg(ideal_k)
    ndcg = dcg / idcg if idcg else 0.0
    rr = next((1.0 / (r + 1) for r, v in enumerate(rk) if v > 0), 0.0)
    total_pos = sum(1 for v in all_rels if v > 0)
    got_pos   = sum(1 for v in rk if v > 0)
    recall = (got_pos / total_pos) if total_pos else 0.0
    return ndcg, rr, recall

def _slug(model: str | Path) -> str:
    return str(model).replace("/", "_")

def _load_qrels(fp: Path) -> Dict[str, Dict[str, int]]:
    """
    qrels: {qid_str: {docid: score}}
    """
    arr = load_json_any(fp)  # supports both JSON and JSONL
    tab: Dict[str, Dict[str, int]] = {}
    for it in arr:
        qid = str(it["query-id"])
        did = str(it["corpus-id"])
        sc  = int(it.get("score", 0))
        tab.setdefault(qid, {})[did] = sc
    return tab

def _find_climate_label_file(label_dir: Path) -> Path:
    """
    Find the climate label_pool file (containing doc_ids).
    """
    cands = sorted(label_dir.glob("*_label_pool.json"))
    if not cands:
        raise FileNotFoundError(f"no label_pool json under: {label_dir}")
    for fp in cands:
        if "climate" in fp.stem or "climate_fever" in fp.stem:
            return fp
    return cands[0]

def calc_metrics_climate(model: str, parent: str) -> Path:
    """
    Read climate_fever label_pool and qrels, compute metrics, and write output to
    performance/<parent>/<ckpt_name>/. Returns the output directory Path.
    """
    slug = _slug(model)
    out_dir = _PERF_ROOT / parent / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    label_dir = _LABEL_ROOT / slug
    if not label_dir.is_dir():
        raise FileNotFoundError(f"label-pool directory not found: {label_dir}")

    qrels = _load_qrels(_QRELS_FP)

    lab_fp = _find_climate_label_file(label_dir)
    data = json.loads(lab_fp.read_text(encoding="utf-8"))

    rows_zero, rows_ignore = [], []
    sums_z = dict(ndcg=0.0, rr=0.0, rec=0.0, nq=0)
    sums_i = dict(ndcg=0.0, rr=0.0, rec=0.0, nq=0)

    total_pairs = 0
    unannotated = 0

    for item in data:
        qid = str(item.get("query_id", "")) or str(item.get("qid", ""))
        if not qid:
            continue
        retrieved_ids: List[str] = [str(x) for x in item.get("doc_ids", [])][:K]
        total_pairs += len(retrieved_ids)

        all_rels = list(qrels.get(qid, {}).values())
        if not all_rels:
            continue

        # unannotated stats
        for did in retrieved_ids:
            if did not in qrels.get(qid, {}):
                unannotated += 1

        # zero-fill: treat unknown as 0
        rels_z = [qrels.get(qid, {}).get(did, 0) for did in retrieved_ids]
        n_z, r_z, rec_z = _metrics_at_k(rels_z, all_rels, K)
        sums_z["ndcg"] += n_z; sums_z["rr"] += r_z; sums_z["rec"] += rec_z; sums_z["nq"] += 1

        # ignore-list: skip unannotated
        rels_i = [qrels[qid][did] for did in retrieved_ids if did in qrels.get(qid, {})]
        n_i, r_i, rec_i = _metrics_at_k(rels_i, all_rels, K)
        sums_i["ndcg"] += n_i; sums_i["rr"] += r_i; sums_i["rec"] += rec_i; sums_i["nq"] += 1

    def _row(sums):
        return dict(
            model=slug,
            search_task="QA",
            event_type="climate_fever",
            num_queries=sums["nq"],
            **{k.upper() + "@10": (sums[k] / sums["nq"] if sums["nq"] else 0.0)
               for k in ("ndcg", "rr", "rec")},
        )

    df_z = pd.DataFrame([_row(sums_z)])
    df_i = pd.DataFrame([_row(sums_i)])

    path_zero   = out_dir / "metrics_zero_per_event.csv"
    path_ignore = out_dir / "metrics_ignore_per_event.csv"
    df_z.to_csv(path_zero, index=False)
    df_i.to_csv(path_ignore, index=False)

    per_task = df_z.groupby("search_task")["NDCG@10"].mean().reset_index()
    (out_dir / "ndcg_per_task.csv").write_text(per_task.to_csv(index=False), encoding="utf-8")

    overall_mean = per_task["NDCG@10"].mean() if not per_task.empty else 0.0
    (out_dir / "ndcg_overall.txt").write_text(f"Overall mean NDCG@10: {overall_mean:.4f}\n", encoding="utf-8")

    stats = dict(
        total_retrieved_pairs=total_pairs,
        unannotated_pairs=unannotated,
        unannotated_ratio=(unannotated / total_pairs) if total_pairs else 0.0,
    )
    (out_dir / "retrieval_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    LOGGER.info("saved climate metrics ➜ %s", out_dir)
    return out_dir

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Calculate Climate-FEVER metrics for one checkpoint")
    parser.add_argument("--model",  required=True, help="checkpoint relative path, matching the label-pool slug")
    parser.add_argument("--parent", required=True, help="parent directory name (performance/<parent>/...)")
    args = parser.parse_args()
    calc_metrics_climate(args.model, args.parent)
