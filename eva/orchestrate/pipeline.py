# eva/orchestrate/pipeline.py
# ---------------------------------------------------------------------------
"""Three step evaluation pipeline for a single checkpoint.

New feature
-----------
" Added `use_encode` flag; passed through to query/corpus embedding stages.
" Added `backbone_type` flag; passed through to query/corpus embedding stages.
" NEW: Added `padding_side` flag; passed through to query/corpus embedding stages.
" NEW: Support eva_test='trec' and 'climate' and route to respective metrics.
" NEW: Support eva_test='biosses' and route to metrics_biosses.
"""

from __future__ import annotations
import os, sys
from pathlib import Path
from typing import Iterable, Literal

# Allow standalone execution to locate the package
if __name__ == "__main__" and __package__ is None:
    here = os.path.abspath(os.path.dirname(__file__))
    code_root = os.path.abspath(os.path.join(here, "..", ".."))
    if code_root not in sys.path:
        sys.path.insert(0, code_root)
    __package__ = "eva.orchestrate"

from eva.embed.query     import build_query_emb
from eva.retrieval.exact import exact_search
from eva.evaluation.metrics import calc_metrics
from eva.evaluation.metrics_trec import calc_metrics_trec   # NEW
from eva.evaluation.metrics_climate import calc_metrics_climate  # NEW
from eva.evaluation.metrics_biosses import calc_metrics_biosses  # NEW

from eva.utils.config import (
    DEFAULT_BATCH,
    DEFAULT_MAXLEN,
    DEFAULT_TOPK,
    QUERY_EMB_DIR,
    LABEL_POOL_DIR,
    BASELINE_INDEX_DIR,
)

import logging
LOGGER = logging.getLogger(__name__)

def _slug(model: str | Path) -> str:
    return str(model).replace("/", "_")

def _cleanup_intermediate(slug: str, eva_test: str) -> None:
    """Delete intermediate artefacts (query_emb / corpus_emb / label_pool)."""
    from shutil import rmtree
    # query embeddings
    qdir = Path(QUERY_EMB_DIR) / f"{slug}__{eva_test}" 
    if qdir.exists():
        rmtree(qdir)
    # label pool
    lp_dir = Path(LABEL_POOL_DIR) / slug
    if lp_dir.exists():
        rmtree(lp_dir)
    # corpus embeddings
    for ext in ("fp16.npy", "fp32.npy"):
        for fp in (
            Path(BASELINE_INDEX_DIR) / f"{slug}.{ext}",
            Path(BASELINE_INDEX_DIR) / f"{slug}.eva.{ext}",
            Path(BASELINE_INDEX_DIR) / f"{slug}.test.{ext}",
            Path(BASELINE_INDEX_DIR) / f"{slug}.trec.{ext}",
            Path(BASELINE_INDEX_DIR) / f"{slug}.climate.{ext}",
            Path(BASELINE_INDEX_DIR) / f"{slug}.biosses.{ext}",  # NEW
        ):
            if fp.exists():
                fp.unlink()

def run_pipeline(
    model: str | Path,
    parent: str,
    *,
    eva_test: Literal["eva", "test", "trec", "climate", "biosses"] = "eva",
    pool: str = "cls",
    batch: int = DEFAULT_BATCH,
    max_len: int = DEFAULT_MAXLEN,
    tasks: Iterable[str] | None = None,
    ckpt_type: Literal["auto", "full", "lora"] = "auto",
    backbone: str | Path | None = None,
    backbone_type: Literal["decoder_only", "qwen3bi"] = "decoder_only",
    rebuild_query_emb: bool = False,
    rebuild_corpus_emb: bool = False,
    topk: int = DEFAULT_TOPK,
    keep_intermediate: bool = True,
    use_encode: bool = False,            # NEW
    padding_side: Literal["left", "right"] = "right",   # NEW
) -> None:
    """
    Execute the full evaluation pipeline on a single checkpoint.

    Parameters
    ----------
    use_encode : bool, default False
        When True, embedding helpers will rely on model.encode() if available.
    backbone_type : {"decoder_only","qwen3bi"}, default "decoder_only"
        Explicitly choose the backbone class to instantiate when loading.
    padding_side : {"left","right"}, default "right"
        Pass through to set tokenizer padding side in downstream stages.
    """
    model_str = str(model)

    print("¶ STEP 1  build/load query embeddings")
    build_query_emb(
        model_name=model_str,
        pool=pool,
        eva_test=eva_test,
        ckpt_type=ckpt_type,
        backbone=backbone,
        backbone_type=backbone_type,   # NEW
        batch=batch,
        max_len=max_len,
        rebuild=rebuild_query_emb,
        use_encode=use_encode,         # propagate
        padding_side=padding_side,     # NEW
    )

    print("¶ STEP 2  exact dense retrieval")
    exact_search(
        model_name=model_str,
        pool=pool,
        eva_test=eva_test,
        tasks=tasks,
        ckpt_type=ckpt_type,
        backbone=backbone,
        backbone_type=backbone_type,   # NEW
        rebuild_corpus_emb=rebuild_corpus_emb,
        topk=topk,
        use_encode=use_encode,         # propagate
        padding_side=padding_side,     # NEW
    )

    print("¶ STEP 3  calc metrics")
    if eva_test == "trec":
        out_dir = calc_metrics_trec(model=model_str, parent=parent)
    elif eva_test == "climate":
        out_dir = calc_metrics_climate(model=model_str, parent=parent)
    elif eva_test == "biosses":
        out_dir = calc_metrics_biosses(model=model_str, parent=parent)
    else:
        out_dir = calc_metrics(model=model_str, parent=parent)

    # print overall metric #
    overall_fp = out_dir / "ndcg_overall.txt"
    if overall_fp.is_file():
        print(f"[RESULT] {model_str} · {overall_fp.read_text().strip()}")

    # optional cleanup #
    if not keep_intermediate:
        _cleanup_intermediate(_slug(model_str), eva_test)

    print(f"ipeline finished for {model_str}")


# CLI wrapper
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="One shot pipeline for a single checkpoint"
    )
    parser.add_argument("--model",  required=True, help="checkpoint or adapter directory")
    parser.add_argument("--parent", required=True, help="group name under performance/")
    parser.add_argument("--eva_test", choices=["eva", "test", "trec", "climate", "biosses"], default="eva")
    parser.add_argument("--pool", default="cls")
    parser.add_argument("--batch",   type=int, default=DEFAULT_BATCH)
    parser.add_argument("--max_len", type=int, default=DEFAULT_MAXLEN)
    parser.add_argument("--tasks",   nargs="*", metavar="TASK")
    parser.add_argument("--ckpt_type", choices=["auto", "full", "lora"], default="auto",
                        help="checkpoint type: auto/full/lora")
    parser.add_argument("--backbone", help="base model dir when ckpt_type=lora")
    parser.add_argument("--backbone_type", choices=["decoder_only","qwen3bi"], default="decoder_only",
                        help="choose backbone class to instantiate")
    parser.add_argument("--rebuild_query_emb",   action="store_true")
    parser.add_argument("--rebuild_corpus_emb",  action="store_true")
    parser.add_argument("--topk",  type=int, default=DEFAULT_TOPK)
    parser.add_argument("--discard_intermediate", action="store_true",
                        help="delete intermediate artefacts after run")
    parser.add_argument("--use_encode", action="store_true",
                        help="use model.encode() when available")   # NEW
    parser.add_argument("--padding_side", choices=["left", "right"], default="right",
                        help="set tokenizer padding side (default: right)")   # NEW

    args = parser.parse_args()
    run_pipeline(
        model=args.model,
        parent=args.parent,
        eva_test=args.eva_test,
        pool=args.pool,
        batch=args.batch,
        max_len=args.max_len,
        tasks=args.tasks,
        ckpt_type=args.ckpt_type,
        backbone=args.backbone,
        backbone_type=args.backbone_type,
        rebuild_query_emb=args.rebuild_query_emb,
        rebuild_corpus_emb=args.rebuild_corpus_emb,
        topk=args.topk,
        keep_intermediate=not args.discard_intermediate,
        use_encode=args.use_encode,      # propagate
        padding_side=args.padding_side,  # NEW
    )
