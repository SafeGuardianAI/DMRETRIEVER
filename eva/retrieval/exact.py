# eva/retrieval/exact.py
# ---------------------------------------------------------------------------
"""Exact dot product retrieval that produces **label pools**.

New feature
-----------
" `use_encode` flag: forward propagated to `build_corpus_emb`, allowing the
  caller to decide whether to use the model’s `.encode()` shortcut.
" Added `backbone_type` flag: explicitly choose the backbone class to use
  (decoder_only vs qwen3bi), propagated to `build_corpus_emb`.
" NEW: Added `padding_side` flag: explicitly set tokenizer padding side
  (left/right), propagated to `build_corpus_emb`.
" NEW: Support eva_test='trec' (TREC-COVID) & 'climate' (Climate-FEVER)
       - automatically build ordered corpus + id map
       - add "query_id" and "doc_ids" to label_pool
" NEW: Support eva_test='biosses' (BIOSSES STS)
       - use text_2 as corpus; write query_id / doc_ids
"""

from __future__ import annotations

import os
import sys

# Ensure standalone execution can still find the package
if __name__ == "__main__" and __package__ is None:
    here = os.path.abspath(os.path.dirname(__file__))              # eva/retrieval
    code_root = os.path.abspath(os.path.join(here, "..", ".."))
    if code_root not in sys.path:
        sys.path.insert(0, code_root)
    __package__ = "eva.retrieval"

import gc
import json
from pathlib import Path
from typing import Iterable, Literal, List

import numpy as np
import torch

from eva.embed.corpus import build_corpus_emb
from eva.utils.config import (
    QUERY_EMB_DIR, LABEL_POOL_DIR, CORPUS_DIR, TEST_QUERY_DIR, RAW_DATA_DIR,
    DEFAULT_TOPK, USE_FP16,
)
from eva.utils.io_utils import (
    build_ordered_corpus,
    load_ordered_corpus,
    build_trec_covid_corpus,
    build_climate_fever_corpus,
    build_biosses_corpus,
    load_json_any,
)

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DTYPE_TORCH = torch.float16 if (USE_FP16 and _DEVICE == "cuda") else torch.float32
_DTYPE_NP = np.float16 if USE_FP16 else np.float32

_CORPUS = Path(CORPUS_DIR)
_TQ = Path(TEST_QUERY_DIR)
_RAW = Path(RAW_DATA_DIR)

_EVA_TEST_CFG = {
    "eva": {
        "TEST_QUERY_PATH": _TQ,
        "CORPUS_JSON":     _CORPUS / "ordered_corpus_eva.json",
        "CORPUS_BASE":     _CORPUS,
        "IDMAP_JSON":      None,
    },
    "test": {
        "TEST_QUERY_PATH": _TQ.parent / "test_query_test",
        "CORPUS_JSON":     _CORPUS / "ordered_corpus_full.json",
        "CORPUS_BASE":     _CORPUS,
        "IDMAP_JSON":      None,
    },
    "trec": {
        "TEST_QUERY_PATH": _RAW / "trec_covid_queries.json",
        "CORPUS_JSON":     _RAW / "ordered_corpus_trec.json",
        "CORPUS_BASE":     _RAW,
        "TREC_SRC":        _RAW / "trec_covid_corpus.json",
        "IDMAP_JSON":      _RAW / "ordered_corpus_trec_ids.json",
    },
    "climate": {
        "TEST_QUERY_PATH": _RAW / "climate_fever" / "queries.jsonl",
        "CORPUS_JSON":     _RAW / "climate_fever" / "ordered_corpus_climate.json",
        "CORPUS_BASE":     _RAW / "climate_fever",
        "SRC":             _RAW / "climate_fever" / "corpus.jsonl",
        "IDMAP_JSON":      _RAW / "climate_fever" / "ordered_corpus_climate_ids.json",
    },
    "biosses": {
        "TEST_QUERY_PATH": _RAW / "biosses" / "merged.jsonl",
        "CORPUS_JSON":     _RAW / "biosses" / "ordered_corpus_biosses.json",
        "CORPUS_BASE":     _RAW / "biosses",
        "SRC":             _RAW / "biosses" / "merged.jsonl",
        "IDMAP_JSON":      _RAW / "biosses" / "ordered_corpus_biosses_ids.json",
    },
}

def _slug(model_name: str) -> str:
    """Replace `/` with `_` so it is safe for filenames."""
    return model_name.replace("/", "_")

def _iter_tasks(test_query_path: Path, eva_test: str) -> set[str]:
    """Return the set of search task names appearing in the query path."""
    if eva_test in ("trec", "climate"):
        return {"QA"}  # always use QA
    if eva_test == "biosses":
        return {"STS"}
    if test_query_path.is_file():
        stem = test_query_path.stem
        return {stem.split("_", 1)[0]}
    return {fp.stem.split("_", 1)[0] for fp in test_query_path.glob("*.json")}

def _list_json_files(p: Path) -> List[Path]:
    if p.is_dir():
        return list(p.glob("*.json"))
    elif p.is_file():
        return [p]
    else:
        raise FileNotFoundError(f"query path not found: {p}")

# Public API
def exact_search(
    model_name: str | Path,
    pool: str = "cls",
    eva_test: Literal["eva", "test", "trec", "climate", "biosses"] = "eva",
    *,
    tasks: Iterable[str] | None = None,
    ckpt_type: Literal["auto", "full", "lora"] = "auto",
    backbone: str | Path | None = None,
    backbone_type: Literal["decoder_only", "qwen3bi"] = "decoder_only",
    rebuild_corpus_emb: bool = False,
    topk: int = DEFAULT_TOPK,
    use_encode: bool = False,  # NEW
    padding_side: Literal["left", "right"] = "right",  # NEW
) -> Path:
    """
    Run dense retrieval and save label pools.
    """
    if eva_test not in _EVA_TEST_CFG:
        raise ValueError("eva_test must be 'eva' or 'test' or 'trec' or 'climate' or 'biosses'")

    cfg = _EVA_TEST_CFG[eva_test]
    test_query_path: Path = cfg["TEST_QUERY_PATH"]
    corpus_json: Path = cfg["CORPUS_JSON"]

    # ---------- ordered corpus -------------------------------------------------
    if not corpus_json.exists():
        if eva_test == "trec":
            build_trec_covid_corpus(cfg["TREC_SRC"], corpus_json, cfg["IDMAP_JSON"])
        elif eva_test == "climate":
            build_climate_fever_corpus(cfg["SRC"], corpus_json, cfg["IDMAP_JSON"])
        elif eva_test == "biosses":
            build_biosses_corpus(cfg["SRC"], corpus_json, cfg["IDMAP_JSON"])
        else:
            print(f"[exact] building ordered_corpus → {corpus_json}")
            build_ordered_corpus(str(cfg["CORPUS_BASE"]), str(corpus_json))
    corpus: list[str] = load_ordered_corpus(str(corpus_json))
    print(f"[exact] loaded corpus: {len(corpus):,} passages")

    # optional id map (trec/climate/biosses)
    idmap: List[str] | None = None
    if cfg.get("IDMAP_JSON"):
        idmap = json.loads(Path(cfg["IDMAP_JSON"]).read_text(encoding="utf-8"))
        if len(idmap) != len(corpus):
            raise RuntimeError("IDMAP length mismatch with corpus")

    # ---------- corpus embeddings ---------------------------------------------
    corpus_emb_fp = build_corpus_emb(
        model_name=model_name,
        pool=pool,
        eva_test=eva_test,
        ckpt_type=ckpt_type,
        backbone=backbone,
        backbone_type=backbone_type,   # NEW
        rebuild=rebuild_corpus_emb,
        use_encode=use_encode,         # propagate
        padding_side=padding_side,     # NEW
    )
    corpus_emb = np.load(corpus_emb_fp, mmap_mode="r")
    corpus_emb_T = torch.tensor(corpus_emb, dtype=_DTYPE_TORCH, device=_DEVICE)

    # ---------- label-pool output directory -----------------------------------
    slug = _slug(str(model_name))
    label_out_dir = Path(LABEL_POOL_DIR) / slug
    label_out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- task filtering -------------------------------------------------
    all_tasks = _iter_tasks(test_query_path, eva_test)
    chosen_tasks = set(tasks) if tasks else all_tasks
    bad_tasks = chosen_tasks - all_tasks
    if bad_tasks:
        raise ValueError(f"tasks {bad_tasks} not found in {test_query_path}")

    # ---------- main loop ------------------------------------------------------
    files = _list_json_files(test_query_path)
    for fp in files:
        # parse task/event
        if eva_test == "trec":
            task = "QA";   event_tag = "trec_covid"
        elif eva_test == "climate":
            task = "QA";   event_tag = "climate_fever"
        elif eva_test == "biosses":
            task = "STS";  event_tag = "biosses"
        else:
            task = fp.stem.split("_", 1)[0]
            event_tag = fp.stem.split("_", 1)[1] if "_" in fp.stem else fp.stem

        if task not in chosen_tasks:
            continue

        arr = load_json_any(fp)  # JSON / JSONL

        if eva_test == "trec":
            queries = [ (it.get("text") or "").strip() for it in arr ]
            qids    = [ str(it.get("_id","")).strip() for it in arr ]
        elif eva_test == "climate":
            queries = [ (it.get("text") or "").strip() for it in arr ]
            qids    = [ str(it.get("_id","")).strip() for it in arr ]
        elif eva_test == "biosses":
            queries = [ (it.get("text_1") or "").strip() for it in arr ]
            qids    = [ str(it.get("id","")).strip() for it in arr ]
        else:
            queries = [ it["user_query"].strip() for it in arr ]
            qids    = None

        q_emb_fp = Path(QUERY_EMB_DIR) / f"{slug}__{eva_test}" / f"{fp.stem}.npy"
        if not q_emb_fp.exists():
            print(f"[exact] WARN skip {fp.name} missing query_embeddings")
            continue

        q_emb = np.load(q_emb_fp).astype(_DTYPE_NP)
        q_emb_t = torch.tensor(q_emb, dtype=_DTYPE_TORCH, device=_DEVICE)

        # batched dot product + topk #
        text_lists: list[list[str]] = []
        id_lists:   list[list[str]] = []
        B = min(128, q_emb_t.size(0))      # micro batch
        with torch.no_grad():
            for i in range(0, q_emb_t.size(0), B):
                sims = q_emb_t[i : i + B] @ corpus_emb_T.T  # [b, |C|]
                _, idx = sims.topk(topk, dim=1)
                for inds in idx.cpu().numpy():
                    inds_list = list(inds)
                    text_lists.append([corpus[k] for k in inds_list])
                    if idmap is not None:
                        id_lists.append([idmap[k] for k in inds_list])

        # write label_pool JSON #
        out_items = []
        for j, texts in enumerate(text_lists):
            item = {"user_query": queries[j], "passages": texts}
            if qids is not None:
                item["query_id"] = qids[j]
            if idmap is not None:
                item["doc_ids"] = id_lists[j]
            out_items.append(item)

        out_fp = (
            label_out_dir / f"{task}_{event_tag}_label_pool.json"
            if eva_test in ("trec","climate","biosses")
            else label_out_dir / f"{fp.stem}_label_pool.json"
        )

        out_fp.write_text(json.dumps(out_items, ensure_ascii=False, indent=2),
                          encoding="utf-8")

    # ---------- cleanup --------------------------------------------------------
    del corpus_emb, corpus_emb_T, q_emb, q_emb_t
    gc.collect()
    if _DEVICE == "cuda":
        torch.cuda.empty_cache()

    print(f"[exact] finished label pools saved to {label_out_dir}")
    return label_out_dir


# Lightweight CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Exact dense retrieval (label pool)")
    parser.add_argument("--model",  required=True, help="checkpoint or adapter directory")
    parser.add_argument("--pool",   default="cls", help="pooling method cls/mean/last")
    parser.add_argument("--eva_test", choices=["eva", "test", "trec", "climate", "biosses"], default="eva")
    parser.add_argument("--task",  nargs="*", help="only run the specified searchTasks")
    parser.add_argument("--ckpt_type", choices=["auto", "full", "lora"], default="auto",
                        help="checkpoint type: auto/full/lora")
    parser.add_argument("--backbone", help="base model dir when ckpt_type=lora")
    parser.add_argument("--backbone_type", choices=["decoder_only","qwen3bi"], default="decoder_only",
                        help="choose backbone class to instantiate")
    parser.add_argument("--rebuild_corpus_emb", action="store_true",
                        help="recompute corpus embeddings")
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK,
                        help="number of passages returned per query")
    parser.add_argument("--use_encode", action="store_true",
                        help="use model.encode() when available")  # NEW
    parser.add_argument("--padding_side", choices=["left", "right"], default="right",
                        help="set tokenizer padding side (default: right)")  # NEW

    args = parser.parse_args()
    exact_search(
        model_name=args.model,
        pool=args.pool,
        eva_test=args.eva_test,
        tasks=args.task,
        ckpt_type=args.ckpt_type,
        backbone=args.backbone,
        backbone_type=args.backbone_type,
        rebuild_corpus_emb=args.rebuild_corpus_emb,
        topk=args.topk,
        use_encode=args.use_encode,       # propagate
        padding_side=args.padding_side,   # propagate
    )
