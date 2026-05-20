# eva/embed/corpus.py
# ---------------------------------------------------------------------------
"""Unified entry for generating or loading **Passage Embeddings**.
 
Changes:
    " Added *use_encode* argument so that callers can decide whether to rely on
      the model’s `.encode()` shortcut (if available).
    " Added *backbone_type* argument to explicitly choose the backbone class
      when loading models (e.g., decoder_only vs qwen3bi).
    " NEW: Added *padding_side* argument to set tokenizer padding side
      (left/right), default remains 'right'.
    " NEW: Support eva_test='trec' and 'climate' (TREC-COVID / Climate-FEVER).
    " NEW: Support eva_test='biosses' (BIOSSES STS): text_2 as corpus.
"""

from __future__ import annotations
import gc
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from eva.utils.config import (
    BASELINE_INDEX_DIR, CHECKPOINT_ROOT, CORPUS_DIR, RAW_DATA_DIR,
    DEFAULT_BATCH, DEFAULT_MAXLEN, USE_FP16,
)
from eva.utils.embed_utils import embed_texts
from eva.utils.model_utils import load_model_and_tokenizer

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DTYPE_TORCH = torch.float16 if (USE_FP16 and _DEVICE == "cuda") else torch.float32

_CHECKPOINT_ROOT = Path(CHECKPOINT_ROOT)
_CORPUS_DIR = Path(CORPUS_DIR)
_RAW = Path(RAW_DATA_DIR)

_EVA_TEST_CFG = {
    "eva": {
        "CORPUS_JSON": _CORPUS_DIR / "ordered_corpus_eva.json",
        "CORPUS_BASE": _CORPUS_DIR,
    },
    "test": {
        "CORPUS_JSON": _CORPUS_DIR / "ordered_corpus_full.json",
        "CORPUS_BASE": _CORPUS_DIR,
    },
    "trec": {
        "CORPUS_JSON": _RAW / "ordered_corpus_trec.json",
        "CORPUS_BASE": _RAW,
        "TREC_SRC":    _RAW / "trec_covid_corpus.json",
        "TREC_IDMAP":  _RAW / "ordered_corpus_trec_ids.json",
    },
    "climate": {
        "CORPUS_JSON": _RAW / "climate_fever" / "ordered_corpus_climate.json",
        "CORPUS_BASE": _RAW / "climate_fever",
        "SRC":         _RAW / "climate_fever" / "corpus.jsonl",
        "IDMAP":       _RAW / "climate_fever" / "ordered_corpus_climate_ids.json",
    },
    "biosses": {
        "CORPUS_JSON": _RAW / "biosses" / "ordered_corpus_biosses.json",
        "CORPUS_BASE": _RAW / "biosses",
        "SRC":         _RAW / "biosses" / "merged.jsonl",
        "IDMAP":       _RAW / "biosses" / "ordered_corpus_biosses_ids.json",
    },
}

# Helpers
def _slug(model_name: str) -> str:
    return model_name.replace("/", "_")

def _resolve_ckpt_path(model_name: str | Path) -> Path:
    """Return absolute checkpoint path, falling back to `_CHECKPOINT_ROOT`."""
    p = Path(model_name)
    return p if p.exists() else _CHECKPOINT_ROOT / model_name

# Core API
def build_corpus_emb(
    model_name: str | Path,
    pool: str = "cls",
    eva_test: Literal["eva", "test", "trec", "climate", "biosses"] = "eva",
    *,
    ckpt_type: Literal["auto", "full", "lora"] = "auto",
    backbone: str | Path | None = None,
    backbone_type: Literal["decoder_only", "qwen3bi"] = "decoder_only",
    rebuild: bool = False,
    use_encode: bool = False,  # NEW
    padding_side: Literal["left", "right"] = "right",  # NEW
) -> Path:
    """
    Encode or load passage embeddings.
    """
    if eva_test not in _EVA_TEST_CFG:
        raise ValueError("eva_test must be 'eva' or 'test' or 'trec' or 'climate' or 'biosses'")

    cfg = _EVA_TEST_CFG[eva_test]
    corpus_json = cfg["CORPUS_JSON"]
    corpus_base = cfg["CORPUS_BASE"]

    # ---------- ordered corpus -------------------------------------------------
    if not corpus_json.exists():
        if eva_test == "trec":
            from eva.utils.io_utils import build_trec_covid_corpus
            build_trec_covid_corpus(cfg["TREC_SRC"], corpus_json, cfg["TREC_IDMAP"])
        elif eva_test == "climate":
            from eva.utils.io_utils import build_climate_fever_corpus
            build_climate_fever_corpus(cfg["SRC"], corpus_json, cfg["IDMAP"])
        elif eva_test == "biosses":
            from eva.utils.io_utils import build_biosses_corpus
            build_biosses_corpus(cfg["SRC"], corpus_json, cfg["IDMAP"])
        else:
            print(f"[corpus_emb] building ordered_corpus → {corpus_json}")
            from eva.utils.io_utils import build_ordered_corpus
            build_ordered_corpus(corpus_base, str(corpus_json))
    from eva.utils.io_utils import load_ordered_corpus
    corpus = load_ordered_corpus(str(corpus_json))

    # ---------- cache path -----------------------------------------------------
    slug = _slug(str(model_name))
    emb_dir = Path(BASELINE_INDEX_DIR)
    emb_dir.mkdir(parents=True, exist_ok=True)
    cache_fp = emb_dir / f"{slug}.{eva_test}.{ 'fp16' if USE_FP16 else 'fp32' }.npy"

    if cache_fp.exists() and not rebuild:
        print(f"[corpus_emb] cache hit for {model_name} ({eva_test})")
        return cache_fp

    # ---------- model loading --------------------------------------------------
    ckpt_path = _resolve_ckpt_path(model_name)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    tok, mdl = load_model_and_tokenizer( 
        ckpt_path,
        device=_DEVICE,
        torch_dtype=_DTYPE_TORCH,
        ckpt_type=ckpt_type,
        backbone_path=backbone,
        trust_remote_code=True,
        backbone_type=backbone_type,   # NEW
        padding_side=padding_side,     # NEW
    )

    # ---------- encoding -------------------------------------------------------
    embs, _ = embed_texts(
        mdl,
        tok,
        corpus,
        max_len=DEFAULT_MAXLEN,
        batch_size=DEFAULT_BATCH,
        pool_tag=pool,
        device=_DEVICE,
        dtype=_DTYPE_TORCH,
        use_encode=use_encode,  # propagate
        desc=f"{slug}-corpus-{eva_test}",
    )

    if USE_FP16:
        embs = embs.astype(np.float16)
    np.save(cache_fp, embs)

    # ---------- cleanup --------------------------------------------------------
    del mdl, tok
    gc.collect()
    if _DEVICE == "cuda":
        torch.cuda.empty_cache()

    return cache_fp
