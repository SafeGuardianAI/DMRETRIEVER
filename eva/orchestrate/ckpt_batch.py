# eva/orchestrate/ckpt_batch.py
# ---------------------------------------------------------------------------
"""Batch traversal to evaluate every checkpoint under a directory.

New feature
-----------
* Added `use_encode` flag; forwarded to `run_pipeline`.
* **NEW**: Added `skip_done` flag – when enabled, skip checkpoints that already
  have a `ndcg_overall.txt` metric file under performance/<parent>/<ckpt>/.
* **NEW**: Added `backbone_type` flag – explicitly choose backbone class
  (decoder_only vs qwen3bi) and forward to `run_pipeline`.
* **NEW**: Added `padding_side` flag – explicitly set tokenizer padding side
  (left/right) and forward to `run_pipeline`.
* **NEW**: `eva_test` supports "trec" & "climate" & "biosses"
"""

from __future__ import annotations
import os, sys, logging, shutil
from pathlib import Path
from typing import Iterable, Literal, Sequence

# Adjust import path for standalone execution
if __name__ == "__main__" and __package__ is None:
    here = os.path.abspath(os.path.dirname(__file__))
    code_root = os.path.abspath(os.path.join(here, "..", ".."))
    sys.path.insert(0, code_root)
    __package__ = "eva.orchestrate"

from .pipeline import run_pipeline
from eva.utils.config import (
    DEFAULT_BATCH, DEFAULT_MAXLEN,
    CHECKPOINT_ROOT, PERF_DIR, QUERY_EMB_DIR, BASELINE_INDEX_DIR, LABEL_POOL_DIR,
)

LOGGER = logging.getLogger(__name__)

_D_TRAIN_ROOT = Path(CHECKPOINT_ROOT)
_PERF_ROOT    = Path(PERF_DIR)
_QE_ROOT      = Path(QUERY_EMB_DIR)
_CE_ROOT      = Path(BASELINE_INDEX_DIR)
_LP_ROOT      = Path(LABEL_POOL_DIR)

def _slug(path: str | Path) -> str:
    return str(path).replace("/", "_")

# Main entry
def run_ckpt_batch(
    model_dir: str | Path,
    *,
    eva_test: Literal["eva", "test", "trec", "climate", "biosses"] = "eva",
    pool: str | None = None,
    batch: int = DEFAULT_BATCH,
    max_len: int = DEFAULT_MAXLEN,
    tasks: Iterable[str] | None = None,
    ckpt_type: Literal["auto", "full", "lora"] = "auto",
    backbone: str | Path | None = None,
    backbone_type: Literal["decoder_only","qwen3bi"] = "decoder_only",
    rebuild_query_emb: bool = False,
    rebuild_corpus_emb: bool = False,
    start_idx: int | None = None,
    end_idx: int | None = None,
    min_ckpt: int | None = None,
    max_ckpt: int | None = None,
    only_ckpts: Iterable[int] | None = None,
    keep_intermediate: bool = True,
    use_encode: bool = False,
    skip_done: bool = False,              # NEW
    padding_side: Literal["left","right"] = "right",  # NEW
) -> None:
    """
    Batch-evaluate all checkpoints within *model_dir*.
    """
    # resolve dataset_dir
    dataset_dir = Path(model_dir)
    if not dataset_dir.is_dir():
        dataset_dir = _D_TRAIN_ROOT / dataset_dir
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"model_dir not found: {model_dir}")

    parent = dataset_dir.name
    LOGGER.info("Batch on %s (parent=%s)", dataset_dir, parent)

    # collect checkpoint subdirs
    all_ckpts: Sequence[Path] = sorted(
        (d for d in dataset_dir.iterdir()
         if d.is_dir() and d.name.startswith("checkpoint-")),
        key=lambda d: int(d.name.split("checkpoint-")[1]),
    )

    # filtering
    if only_ckpts:
        only_set = {int(x) for x in only_ckpts}
        all_ckpts = [d for d in all_ckpts
                     if int(d.name.split("checkpoint-")[1]) in only_set]

    if min_ckpt is not None:
        all_ckpts = [d for d in all_ckpts
                     if int(d.name.split("checkpoint-")[1]) >= min_ckpt]
    if max_ckpt is not None:
        all_ckpts = [d for d in all_ckpts
                     if int(d.name.split("checkpoint-")[1]) <= max_ckpt]

    ckpts = all_ckpts[slice(start_idx, end_idx)]
    if not ckpts:
        LOGGER.warning("no checkpoints selected – abort")
        return

    # pre-create grouped output dir
    (_PERF_ROOT / parent).mkdir(parents=True, exist_ok=True)

    # main loop
    for ckpt in ckpts:
        rel_model = str(ckpt)
        ckpt_name = ckpt.name
        slug = _slug(rel_model)
        perf_dir = _PERF_ROOT / parent / slug
        overall_file = perf_dir / "ndcg_overall.txt"
        print(overall_file)

        # --- skip-done ---------------------------------------------------------
        if skip_done and overall_file.is_file():
            print(f"[SKIP] {ckpt_name} – metrics already present, skipping.")
            continue
        # ----------------------------------------------------------------------
        LOGGER.info("¶ RUN   %s", rel_model)

        run_pipeline(
            model=rel_model,
            parent=parent,
            eva_test=eva_test,
            pool=pool or "cls",
            batch=batch,
            max_len=max_len,
            tasks=tasks,
            ckpt_type=ckpt_type,
            backbone=backbone,
            backbone_type=backbone_type,
            rebuild_query_emb=rebuild_query_emb,
            rebuild_corpus_emb=rebuild_corpus_emb,
            keep_intermediate=keep_intermediate,
            use_encode=use_encode,
            padding_side=padding_side,      # NEW
        )

        # move / clean intermediates
        if keep_intermediate:
            for root in (_QE_ROOT, _LP_ROOT):
                src = root / slug
                dst = root / parent / ckpt_name
                if src.exists():
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.move(str(src), str(dst))

            for ext in ("fp16.npy", "fp32.npy"):
                for src in ((_CE_ROOT / f"{slug}.{ext}"),
                            (_CE_ROOT / f"{slug}.{eva_test}.{ext}")):
                    if src.exists():
                        dst_name = f"{ckpt_name}.{ext}" if src.name.endswith(ext) else src.name
                        dst = _CE_ROOT / parent / dst_name
                        if dst.exists():
                            dst.unlink()
                        shutil.move(str(src), str(dst))
        else:
            for root in (_QE_ROOT, _LP_ROOT):
                tmp = root / slug
                if tmp.exists():
                    shutil.rmtree(tmp)
            for ext in ("fp16.npy", "fp32.npy"):
                for tmp in ((_CE_ROOT / f"{slug}.{ext}"),
                            (_CE_ROOT / f"{slug}.{eva_test}.{ext}")):
                    if tmp.exists():
                        tmp.unlink()

        if overall_file.is_file():
            print(f"[RESULT] {ckpt_name} – {overall_file.read_text().strip()}")
        else:
            print(f"[WARN] {ckpt_name} – ndcg_overall.txt missing")

    LOGGER.info("batch finished – all results under E_test_res/*")


# CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch run over all checkpoints")
    parser.add_argument("--model_dir", required=True, help="root dir containing checkpoints")
    parser.add_argument("--eva_test", choices=["eva", "test", "trec", "climate", "biosses"], default="eva")
    parser.add_argument("--pool")
    parser.add_argument("--batch",   type=int, default=DEFAULT_BATCH)
    parser.add_argument("--max_len", type=int, default=DEFAULT_MAXLEN)
    parser.add_argument("--tasks",   nargs="*", metavar="TASK")
    parser.add_argument("--ckpt_type", choices=["auto", "full", "lora"], default="auto",
                        help="checkpoint type: auto/full/lora")
    parser.add_argument("--backbone", help="basemodel dir when ckpt_type=lora")
    parser.add_argument("--backbone_type", choices=["decoder_only","qwen3bi"], default="decoder_only",
                        help="choose backbone class to instantiate")
    parser.add_argument("--rebuild_query_emb",  action="store_true")
    parser.add_argument("--rebuild_corpus_emb", action="store_true")
    parser.add_argument("--start_idx", type=int)
    parser.add_argument("--end_idx",   type=int)
    parser.add_argument("--min_ckpt", type=int, help="only evaluate checkpoint ≥ min_ckpt")
    parser.add_argument("--max_ckpt", type=int, help="only evaluate checkpoint ≤ max_ckpt")
    parser.add_argument("--ckpt", nargs="+", type=int,
                        help="evaluate only the listed checkpoint numbers")
    parser.add_argument("--discard_intermediate", action="store_true",
                        help="do not keep query emb / corpus emb / label_pool")
    parser.add_argument("--use_encode", action="store_true",
                        help="use model.encode() when available")
    parser.add_argument("--skip_done", action="store_true",
                        help="skip checkpoints that already have performance metrics")
    parser.add_argument("--padding_side", choices=["left", "right"], default="right",
                        help="set tokenizer padding side (default: right)")  # NEW

    args = parser.parse_args()
    run_ckpt_batch(
        model_dir=args.model_dir,
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
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        min_ckpt=args.min_ckpt,
        max_ckpt=args.max_ckpt,
        only_ckpts=args.ckpt,
        keep_intermediate=not args.discard_intermediate,
        use_encode=args.use_encode,        # propagate
        skip_done=args.skip_done,          # propagate
        padding_side=args.padding_side,    # NEW
    )
