# -*- coding: utf-8 -*-
"""
utils.lora_proj_encoder
————————————————————————————————————————————
Unified encoder wrapping Backbone + LoRA (+ optional MLP projection),
with batch-level progress bar. If no proj_mlp weights file is provided,
the MLP is skipped and raw backbone hidden_states are output directly.
"""
from __future__ import annotations

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel
from tqdm import tqdm
from pathlib import Path


class LoraProjEncoder:
    """Unified encoding interface: Backbone + LoRA (+ optional MLP projection)"""

    def __init__(
        self,
        base_model: str,
        checkpoint: str,
        proj_mlp_path: str | None,
        mlp_hidden: int,
        mlp_out: int,
        device: str = "cpu",
        normalize: bool = True,
    ) -> None:
        self.device = torch.device(device)

        # 1. tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=True, use_fast=False
        )
        self.tokenizer.padding_side = "right"

        # 2. backbone + LoRA
        cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
        base = AutoModel.from_pretrained(base_model, config=cfg, trust_remote_code=True)
        peft = PeftModel.from_pretrained(base, checkpoint, is_trainable=False)
        self.backbone = peft.merge_and_unload().to(self.device).eval()

        self.normalize = normalize
        self.has_proj = False  # whether an MLP projection layer exists

        # 3. proj MLP (optional)
        if proj_mlp_path and Path(proj_mlp_path).is_file():
            # Weights file found: build 2-layer MLP and load parameters
            self.proj = nn.Sequential(
                nn.Linear(self.backbone.config.hidden_size, mlp_hidden, bias=True),
                nn.GELU(),
                nn.Linear(mlp_hidden, mlp_out, bias=True),
            ).to(self.device).eval()
            state = torch.load(proj_mlp_path, map_location=self.device)
            self.proj.load_state_dict(state)
            self.out_dim = mlp_out
            self.has_proj = True
        else:
            # No weights file: use Identity (pass-through)
            if proj_mlp_path:
                print(
                    f"[INFO] proj_mlp weights not found or unavailable: {proj_mlp_path}. "
                    "Skipping MLP projection; output dim remains hidden_size."
                )
            self.proj = nn.Identity()
            self.out_dim = self.backbone.config.hidden_size  # same as hidden_size

    # ───────────────────────────── private helpers ──────────────────────────
    @staticmethod
    @torch.no_grad()
    def _last_token_pool(hidden_states: torch.Tensor, mask: torch.Tensor):
        """Pool the last valid token of each sequence as the sentence vector."""
        idx = mask.long().sum(dim=1) - 1
        return hidden_states[torch.arange(mask.size(0), device=mask.device), idx]

    # ────────────────────────────────── API ──────────────────────────────────
    @torch.no_grad()
    def encode(
        self,
        texts: list[str],
        max_length: int = 512,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Args:
            texts: List of strings to encode.
            max_length: Tokenizer truncation length.
            batch_size: Batch size.

        Returns:
            Tensor of shape (len(texts), out_dim),
            where out_dim = mlp_out (with MLP) or hidden_size (without MLP).
        """
        reps = []
        total = len(texts)

        for i in tqdm(
            range(0, total, batch_size),
            desc="Encoding batches",
            unit="batch",
            leave=False,
        ):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)

            out = self.backbone(**enc, return_dict=True)
            pooled = self._last_token_pool(out.last_hidden_state, enc.attention_mask)
            vec = self.proj(pooled)  # Identity or MLP

            if self.normalize:
                vec = nn.functional.normalize(vec, dim=-1)
            reps.append(vec.cpu())

        return torch.cat(reps, dim=0)
