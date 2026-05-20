from __future__ import annotations

from typing import Optional, Any

import torch
from torch import nn
from transformers.cache_utils import Cache  # noqa: F401  (kept for future use)
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3Model,
    Qwen3ForCausalLM,
    Qwen3PreTrainedModel
)
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.modeling_utils import PreTrainedModel 

try:
    from peft import PeftModel  # type: ignore 
except ImportError:
    PeftModel = Any  # type: ignore

logger = logging.get_logger(__name__)

# ---------------------------------------------------------------------------
# 1.  Bidirectional Attention (no causal mask, no sliding window)
# ---------------------------------------------------------------------------
class ModifiedQwen3Attention(Qwen3Attention):
    """Full‑context attention for Qwen‑3 (no causal masking)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False     # disable autoregressive masking
        self.sliding_window = None # disable SWA entirely


# ---------------------------------------------------------------------------
# 2.  Decoder layer that swaps in the bidirectional attention
# ---------------------------------------------------------------------------
class ModifiedQwen3DecoderLayer(Qwen3DecoderLayer):
    """Qwen‑3 decoder layer using bidirectional attention."""

    def __init__(self, config: PretrainedConfig, layer_idx: int):
        # Call parent constructor *first* to keep gradient‑ckpt flags, etc.
        super().__init__(config, layer_idx)

        # Replace the causal attention with our full‑context version
        self.self_attn = ModifiedQwen3Attention(config=config, layer_idx=layer_idx)
        self.attention_type = "full_attention"
        self.sliding_window = None  # make sure SWA is disabled


# ---------------------------------------------------------------------------
# 3.  Backbone – Qwen‑3 with bidirectional self‑attention
# ---------------------------------------------------------------------------
class Qwen3BiModel(Qwen3Model):
    """Qwen‑3 backbone whose self‑attention is *bidirectional*."""

    _no_split_modules = ["ModifiedQwen3DecoderLayer"]

    def __init__(self, config: PretrainedConfig):
        # Build a regular Qwen‑3 model first (initialises important HF fields)
        super().__init__(config)

        # Swap every decoder layer for the modified, bidirectional variant
        self.layers = nn.ModuleList(
            [ModifiedQwen3DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )

        # Ensure the model never attempts sliding‑window masks again
        self.has_sliding_layers = False

    # ---------------------------- helper ----------------------------------
    @staticmethod
    def _build_pad_bias(pad_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Turn [B,L] bool mask (1=keep, 0=pad) → additive bias [B,1,1,L]."""
        neg_inf = torch.finfo(dtype).min
        bias = (~pad_mask.bool()).to(dtype) * neg_inf
        return bias[:, None, None, :]

    # ---------------------------- forward ---------------------------------
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Build a default *keep‑all* mask if caller omitted one
        if attention_mask is None:
            if input_ids is None:
                raise ValueError("Either attention_mask or input_ids must be provided.")
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        pad_bias = self._build_pad_bias(attention_mask, self.embed_tokens.weight.dtype)
        # Hand a *dict* mask to parent so it skips causal‑mask generation
        attn_mask_dict = {"full_attention": pad_bias}

        return super().forward(
            input_ids=input_ids,
            attention_mask=attn_mask_dict,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# 4.  Task head – Masked‑Next‑Token Prediction (a.k.a. MLM)
# ---------------------------------------------------------------------------
class Qwen3BiForMNTP(Qwen3ForCausalLM):
    """Bidirectional Qwen‑3 with LM head for masked token modelling tasks."""

    def __init__(self, config: PretrainedConfig):
        # Use PreTrainedModel.__init__ directly to set core HF attributes
        Qwen3PreTrainedModel.__init__(self, config)

        self.model = Qwen3BiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # HF helper: initialises weights & final dtype conversions
        self.post_init()

    # -------- generation is undefined for bidirectional models ------------
    def generate(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError(
            "generate() is disabled: this backbone is bidirectional and not autoregressive."
        )

    # --------------------- PEFT convenience helpers -----------------------
    def get_model_for_peft(self):
        """Return backbone for LoRA / adapters."""
        return self.model

    def set_model_for_peft(self, model: PeftModel):  # type: ignore[override]
        """Replace backbone with PEFT‑wrapped version.""" 
        self.model = model

    def save_peft_model(self, path: str):
        if isinstance(self.model, PeftModel):  # type: ignore[arg-type]
            self.model.save_pretrained(path)
        else:
            raise ValueError("Backbone is not a PEFT model; nothing to save.")
