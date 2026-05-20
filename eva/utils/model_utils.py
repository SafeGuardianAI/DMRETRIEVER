# eva/utils/model_utils.py
# ---------------------------------------------------------------------------
"""Unified loader for HuggingFace Tokenizer / Model.

Supports three modes:
1. ckpt_type="full"         -> standard checkpoint directory
2. ckpt_type="lora"         -> LoRA adapter only, requires a backbone
3. ckpt_type="auto"(default)-> auto-detect; if adapter is found and the user
                               also provides a backbone, the user-specified
                               backbone takes priority

Additional features:
- `backbone_type` selects the base model class explicitly:
    - "decoder_only" -> uses AutoModel (backward-compatible)
    - "qwen3bi"      -> uses custom Qwen3BiModel (must be importable)
- `padding_side` sets the tokenizer padding direction ("left"/"right", default "right")

Dependencies:
    pip install peft
"""

from __future__ import annotations
from pathlib import Path
from typing import Literal, Tuple

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

try:
    from peft import PeftModel, PeftConfig
except ImportError as e:
    # Keep original behavior: raise immediately if peft is not installed
    raise RuntimeError("PEFT is required: pip install peft") from e

try:
    from DMRetriever.models import Qwen3BiModel
except ImportError:
    Qwen3BiModel = None  # type: ignore


def _str2path(p) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _load_tokenizer(
    path: str | Path,
    *,
    trust_remote_code: bool = True,
    padding_side: Literal["left", "right"] = "right",
) -> AutoTokenizer:
    """Unified tokenizer loading with fallback defaults."""
    tok = AutoTokenizer.from_pretrained(
        path, trust_remote_code=trust_remote_code, local_files_only=False, use_fast=False
    )
    # -- Fallback: ensure batch padding works --
    if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token
    if hasattr(tok, "padding_side"):
        tok.padding_side = padding_side  # set per user preference
    return tok


def load_model_and_tokenizer(
    ckpt_path: str | Path,
    *,
    device: str = "cpu",
    torch_dtype: torch.dtype | None = None,
    ckpt_type: Literal["auto", "full", "lora"] = "auto",
    backbone_path: str | Path | None = None,
    trust_remote_code: bool = True,
    backbone_type: Literal["decoder_only", "qwen3bi"] = "decoder_only",
    padding_side: Literal["left", "right"] = "right",
) -> Tuple[AutoTokenizer, torch.nn.Module]:
    """
    Load model and tokenizer based on `ckpt_type` and `backbone_type`.

    Parameters
    ----------
    ckpt_path : str | Path
        Checkpoint or adapter directory.
    device : "cpu" | "cuda"
    torch_dtype : torch.dtype | None
    ckpt_type : {"auto","full","lora"}
    backbone_path : str | Path | None
        Base-model directory for ckpt_type="lora" (or inferred from adapter).
    trust_remote_code : bool
        Allow loading custom model implementations from packages.
    backbone_type : {"decoder_only","qwen3bi"}
        Explicitly select base model class; "decoder_only" is backward-compatible.
    padding_side : {"left","right"}, default "right"
        Tokenizer padding direction.

    Returns
    -------
    (tokenizer, model)
    """
    ckpt_path = _str2path(ckpt_path)
    backbone_path = _str2path(backbone_path) if backbone_path else None

    # ---------- Detect LoRA adapter directory --------------------------------
    is_adapter_dir = (ckpt_path / "adapter_config.json").exists() \
        or (ckpt_path / "adapter_model.safetensors").exists()

    if ckpt_type == "full":
        use_adapter = False
    elif ckpt_type == "lora":
        use_adapter = True
        # backbone can be inferred from adapter_config.json later
    else:  # auto
        use_adapter = is_adapter_dir

    # ---------- Load path and objects -----------------------------------------
    if use_adapter:
        # *LoRA*: resolve backbone, instantiate base model, then apply PEFT adapter
        if backbone_path is None:
            # In auto mode, extract base_model_name_or_path from adapter_config
            peft_cfg = PeftConfig.from_pretrained(ckpt_path)
            backbone_path = _str2path(peft_cfg.base_model_name_or_path)
            # print("======backbone_path========")
            # print(backbone_path)
            # print(backbone_path.shape())

        tokenizer = _load_tokenizer(  # pass padding_side
            backbone_path,
            trust_remote_code=trust_remote_code,
            padding_side=padding_side,
        )

        # -- Instantiate base model by backbone_type --
        if backbone_type == "qwen3bi":
            if Qwen3BiModel is None:
                raise RuntimeError(
                    "Qwen3BiModel not found. Ensure DMRetriever.models is importable."
                )
            base_model = Qwen3BiModel.from_pretrained(  # type: ignore
                backbone_path,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                local_files_only=False,
            )
        else:
            base_model = AutoModel.from_pretrained(
                backbone_path,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                local_files_only=False,
            )

        model = (
            PeftModel.from_pretrained(base_model, ckpt_path)
            .to(device)
            .eval()
        )

    else:
        # *Non-LoRA*: build model directly from ckpt_path
        tokenizer = _load_tokenizer(  # pass padding_side
            ckpt_path,
            trust_remote_code=trust_remote_code,
            padding_side=padding_side,
        )
        if backbone_type == "qwen3bi":
            if Qwen3BiModel is None:
                raise RuntimeError(
                    "Qwen3BiModel not found. Ensure DMRetriever.models is importable."
                )
            model = (
                Qwen3BiModel.from_pretrained(  # type: ignore
                    ckpt_path,
                    torch_dtype=torch_dtype,
                    trust_remote_code=trust_remote_code,
                    local_files_only=False,
                )
                .to(device)
                .eval()
            )
            # print("=====model=======")
            # print(model)
            
        else:
            config = AutoConfig.from_pretrained(
                ckpt_path,
                trust_remote_code=True,
                local_files_only=False,
            )
            config.local_path = str(ckpt_path)
            model = (
                AutoModel.from_pretrained(
                    ckpt_path,
                    config=config,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                    local_files_only=False,
                )
                .to(device)
                .eval()
            )
            # print("=====model=======")
            # print(model)

    return tokenizer, model
