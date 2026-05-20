#!/usr/bin/env python3
# -*- coding: utf-8 -*-
## eva/utils/embed_utils.py
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from typing import List, Tuple

# ——— pooling helpers ———
def cls_pool(h: Tensor, m: Tensor) -> Tensor:
    return h[:, 0]

def last_token_pool(h: Tensor, m: Tensor) -> Tensor:
    left_padding = (m[:, -1].sum() == m.shape[0])
    if left_padding:
        return h[:, -1]
    else:
        sequence_lengths = m.sum(dim=1) - 1
        batch_size = h.shape[0]
        return h[torch.arange(batch_size, device=h.device), sequence_lengths]

def mean_pool(h: Tensor, m: Tensor) -> Tensor:
    m = m.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1)

POOL_FN = {"cls": cls_pool, "last": last_token_pool, "mean": mean_pool}
SUPPORTED_POOLS = set(POOL_FN.keys()) | {"pooler"}  # also supports pooler


def embed_texts(
    model,
    tokenizer,
    texts: List[str],
    max_len: int,
    batch_size: int,
    pool_tag: str,
    device: str,
    dtype: torch.dtype,
    *,
    normalize: bool = True,
    use_encode: bool = False,
    desc: str = "embed",
    show_progress: bool = True,
) -> Tuple[np.ndarray, List[int]]:
    """
    Generate embeddings for `texts` in batches.

    Parameters
    ----------
    pool_tag : {"cls","mean","last","pooler"}
        "pooler" uses `model(..., return_dict=True).pooler_output` (e.g. SimCSE/BERT).
    normalize : bool, default True
        Whether to L2-normalize each embedding.
    use_encode : bool, default False
        If the model implements `.encode()`, set True (pool_tag is ignored).
    show_progress : bool, default True
        Whether to show a tqdm progress bar.

    Returns
    -------
    embs : np.ndarray  shape = (len(valid_idxs), hidden_size)
    valid_idxs : List[int]
    """
    if pool_tag not in SUPPORTED_POOLS:
        raise ValueError(f"Unknown pool_tag: {pool_tag}. "
                         f"Supported: {sorted(SUPPORTED_POOLS)}")

    all_embs: List[np.ndarray] = []
    valid_idxs: List[int] = []
    total = len(texts)

    iterator = tqdm(
        range(0, total, batch_size),
        desc=desc,
        unit="batch",
        disable=not show_progress,
    )

    for start in iterator:
        end = min(start + batch_size, total)
        chunk = texts[start:end]
        # try:
        if use_encode:
            # Use the model's built-in encode; pool_tag is ignored
            arr = model.encode(chunk, max_length=max_len)
            t = (
                arr.to(device).to(dtype)
                if isinstance(arr, torch.Tensor)
                else torch.from_numpy(arr).to(device).to(dtype)
            )
            emb = F.normalize(t, p=2, dim=1) if normalize else t

        else:
            tok = tokenizer(
                chunk,
                max_length=max_len,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                if pool_tag == "pooler":
                    out = model(**tok, output_hidden_states=True, return_dict=True)
                    if not hasattr(out, "pooler_output") or out.pooler_output is None:
                        raise RuntimeError(
                            "Model output has no pooler_output. "
                            "Make sure you are using a backbone with a pooler "
                            "(e.g. BERT/SimCSE), or switch to --pool cls/mean/last."
                        )
                    emb = out.pooler_output
                    emb = emb.to(dtype)
                    if normalize:
                        emb = F.normalize(emb, p=2, dim=1)
                else:
                    out = model(**tok)
                    h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
                    pool_fn = POOL_FN[pool_tag]
                    emb = pool_fn(h, tok["attention_mask"])
                    if normalize:
                        emb = F.normalize(emb, p=2, dim=1)
                    emb = emb.to(dtype)

        all_embs.append(emb.cpu().numpy())
        valid_idxs.extend(range(start, end))

        # except Exception as e:
        #     print(f" ✗ skip batch {start}-{end} due to: {e}")
        #     continue

    if all_embs:
        embs = np.concatenate(all_embs, axis=0)
    else:
        hidden_size = (
            model.config.hidden_size
            if hasattr(model, "config") and hasattr(model.config, "hidden_size")
            else 0
        )
        embs = np.empty((0, hidden_size), dtype=np.float32)

    return embs, valid_idxs
