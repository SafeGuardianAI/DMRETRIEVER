import os
import re
import torch
import logging
from typing import Optional

from transformers import AutoConfig, AutoModel, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

try:
    from .models import Qwen3BiModel
except ImportError:
    Qwen3BiModel = None

logger = logging.getLogger(__name__)


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    checkpoint_pattern = re.compile(r"checkpoint-(\d+)")
    max_number, max_chkpt = -1, None
    for file in os.listdir(checkpoint_dir):
        m = checkpoint_pattern.search(file)
        if m:
            n = int(m.group(1))
            if n > max_number:
                max_number, max_chkpt = n, file
    return os.path.join(checkpoint_dir, max_chkpt) if max_chkpt else None


def load_tokenizer(model_cfg):
    if model_cfg.arch_type == "encoder":
        tokenizer = AutoTokenizer.from_pretrained(
            model_cfg.tokenizer_name or model_cfg.model_name_or_path,
            cache_dir=model_cfg.cache_dir,
            token=model_cfg.token,
            trust_remote_code=model_cfg.trust_remote_code,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_cfg.tokenizer_name or model_cfg.model_name_or_path,
            token=model_cfg.token,
            cache_dir=model_cfg.cache_dir,
            use_fast=not model_cfg.use_slow_tokenizer,
            add_eos_token=True,
            trust_remote_code=model_cfg.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
            tokenizer.pad_token_id = (
                tokenizer.unk_token_id if tokenizer.unk_token else tokenizer.eos_token_id
            )
        tokenizer.padding_side = "right"

    resize = False
    if model_cfg.additional_special_tokens:
        added = tokenizer.add_special_tokens(
            {"additional_special_tokens": model_cfg.additional_special_tokens}
        )
        if added > 0:
            resize = True
            logger.info(f"Added {added} special tokens: {model_cfg.additional_special_tokens}")

    return tokenizer, resize


def load_backbone(model_cfg, output_dir: str, tokenizer, resize: bool = False):
    config = AutoConfig.from_pretrained(
        model_cfg.config_name or model_cfg.model_name_or_path,
        token=model_cfg.token,
        cache_dir=model_cfg.cache_dir,
        trust_remote_code=model_cfg.trust_remote_code,
    )
    config.use_cache = False

    if model_cfg.arch_type == "qwen3bi":
        if Qwen3BiModel is None:
            raise ImportError(
                "Qwen3BiModel not available. Requires transformers>=4.40 with Qwen3 support."
            )
        model = Qwen3BiModel.from_pretrained(
            model_cfg.model_name_or_path,
            token=model_cfg.token,
            cache_dir=model_cfg.cache_dir,
            trust_remote_code=True,
            config=config,
        )
    elif model_cfg.arch_type == "encoder":
        model = AutoModel.from_pretrained(
            model_cfg.model_name_or_path,
            cache_dir=model_cfg.cache_dir,
            token=model_cfg.token,
            trust_remote_code=model_cfg.trust_remote_code,
        )
    else:
        model = AutoModel.from_pretrained(
            model_cfg.model_name_or_path,
            token=model_cfg.token,
            cache_dir=model_cfg.cache_dir,
            from_tf=bool(".ckpt" in model_cfg.model_name_or_path),
            config=config,
            trust_remote_code=model_cfg.trust_remote_code,
        )

    if model_cfg.raw_peft:
        model.set_input_embeddings(
            torch.load(os.path.join(model_cfg.raw_peft, "embedding", "emb.pth"))
        )
        model = PeftModel.from_pretrained(model, model_cfg.raw_peft)
        model = model.merge_and_unload()

    if resize:
        model.resize_token_embeddings(len(tokenizer))
        os.makedirs(os.path.join(output_dir, "embedding"), exist_ok=True)
        torch.save(model.get_input_embeddings(), os.path.join(output_dir, "embedding", "emb.pth"))
        target_modules = model_cfg.target_modules
    else:
        target_modules = [t for t in model_cfg.target_modules if t != "embed_tokens"]

    if model_cfg.from_peft:
        if os.path.exists(os.path.join(model_cfg.from_peft, "embedding")):
            model.set_input_embeddings(
                torch.load(os.path.join(model_cfg.from_peft, "embedding", "emb.pth"))
            )
            torch.save(
                model.get_input_embeddings(),
                os.path.join(output_dir, "embedding", "emb.pth"),
            )
        model = PeftModel.from_pretrained(model, model_cfg.from_peft, is_trainable=True)
    elif model_cfg.use_lora:
        peft_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=model_cfg.lora_rank,
            target_modules=target_modules,
            modules_to_save=model_cfg.modules_to_save,
            lora_alpha=model_cfg.lora_alpha,
            lora_dropout=model_cfg.lora_dropout,
        )
        model = get_peft_model(model, peft_cfg)

    return model


def save_merged_lora(model_cfg, output_dir: str):
    if model_cfg.arch_type == "qwen3bi":
        if Qwen3BiModel is None:
            raise ImportError(
                "Qwen3BiModel not available. Requires transformers>=4.40 with Qwen3 support."
            )
        base_model = Qwen3BiModel.from_pretrained(
            model_cfg.model_name_or_path,
            token=model_cfg.token,
            cache_dir=model_cfg.cache_dir,
            trust_remote_code=True,
        )
    else:
        base_model = AutoModel.from_pretrained(
            model_cfg.model_name_or_path,
            token=model_cfg.token,
            cache_dir=model_cfg.cache_dir,
            trust_remote_code=model_cfg.trust_remote_code,
        )

    emb_path = os.path.join(output_dir, "embedding", "emb.pth")
    if os.path.exists(emb_path):
        base_model.set_input_embeddings(torch.load(emb_path))

    try:
        base_model = PeftModel.from_pretrained(base_model, output_dir)
        base_model = base_model.merge_and_unload()
    except Exception:
        base_model = PeftModel.from_pretrained(base_model, find_latest_checkpoint(output_dir))
        base_model = base_model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
    tokenizer.save_pretrained(os.path.join(output_dir, "merged_model"))

    base_model.config.vocab_size = len(tokenizer)
    base_model.save_pretrained(os.path.join(output_dir, "merged_model"))
