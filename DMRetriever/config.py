import os
from typing import Optional, List
from dataclasses import dataclass, field

from transformers import TrainingArguments


def _default_target_modules() -> List[str]:
    return ["v_proj", "q_proj", "k_proj", "gate_proj", "down_proj", "o_proj", "up_proj"]


@dataclass
class ModelConfig:
    model_name_or_path: str = field(
        metadata={"help": "The model checkpoint for initialization."}
    )
    config_name: str = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name."}
    )
    tokenizer_name: str = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name."}
    )
    cache_dir: str = field(
        default=None,
        metadata={"help": "Where to store the pre-trained models."}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Trust remote code"}
    )
    token: str = field(
        default_factory=lambda: os.getenv('HF_TOKEN', None),
        metadata={"help": "The token to use when accessing the model."}
    )

    arch_type: str = field(
        default="encoder",
        metadata={
            "help": "Architecture type: 'encoder', 'decoder', or 'qwen3bi'.",
            "choices": ["encoder", "decoder", "qwen3bi"],
        },
    )

    peft_model_path: str = field(
        default="",
        metadata={"help": "Path to existing LoRA weights to resume training from."}
    )
    use_lora: bool = field(default=False, metadata={"help": "Enable LoRA fine-tuning."})
    lora_rank: int = field(default=64, metadata={"help": "LoRA rank r."})
    lora_alpha: float = field(default=16, metadata={"help": "LoRA scaling alpha."})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout."})
    target_modules: List[str] = field(
        default_factory=_default_target_modules,
        metadata={"help": "Module names to apply LoRA to."}
    )

    use_flash_attn: bool = field(
        default=False,
        metadata={"help": "Enable Flash-Attention 2 if supported by backbone."}
    )
    use_slow_tokenizer: bool = field(
        default=False,
        metadata={"help": "Force slow tokenizer."}
    )

    from_peft: Optional[str] = field(
        default=None,
        metadata={"help": "Load this PEFT checkpoint first, then add new LoRA adapter."}
    )
    modules_to_save: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated PEFT module whitelist."}
    )
    raw_peft: Optional[str] = field(
        default=None,
        metadata={"help": "Load and merge arbitrary LoRA weights as initialization."}
    )
    additional_special_tokens: Optional[str] = field(
        default=None,
        metadata={"help": "Extra special tokens to add to tokenizer.", "nargs": "+"}
    )
    save_merged_lora_model: bool = field(
        default=False,
        metadata={"help": "Merge LoRA + base model and save after training."}
    )
    only_merge_lora_model: bool = field(
        default=False,
        metadata={"help": "Skip training, only merge existing LoRA to base."}
    )

    mlp_out_dim: Optional[int] = field(
        default=None,
        metadata={"help": "If set, add 2-layer MLP to project backbone output to this dimension."}
    )
    mlp_hidden_dim: Optional[int] = field(
        default=None,
        metadata={"help": "MLP hidden dimension. Defaults to backbone hidden_size."}
    )

    def __post_init__(self):
        if self.arch_type not in ("encoder", "decoder", "qwen3bi"):
            raise ValueError(f"Invalid arch_type: {self.arch_type}. Must be 'encoder', 'decoder', or 'qwen3bi'.")
        if self.arch_type == "encoder" and self.use_lora:
            raise ValueError(
                "LoRA is not supported for arch_type='encoder'. "
                "Set use_lora=False or use arch_type='decoder'/'qwen3bi'."
            )


@dataclass
class DataConfig:
    train_data: str = field(
        default=None, metadata={
            "help": "One or more paths to training data.",
            "nargs": "+"
        }
    )
    cache_path: Optional[str] = field(
        default=None, metadata={"help": "Where to store cached data."}
    )
    train_group_size: int = field(default=8)

    query_max_len: int = field(
        default=32,
        metadata={"help": "Max query sequence length after tokenization."},
    )
    passage_max_len: int = field(
        default=128,
        metadata={"help": "Max passage sequence length after tokenization."},
    )
    pad_to_multiple_of: Optional[int] = field(
        default=None,
        metadata={"help": "Pad sequence length to be a multiple of this value."},
    )
    max_example_num_per_dataset: int = field(
        default=100000000, metadata={"help": "Max examples per dataset."}
    )

    query_instruction_for_retrieval: str = field(
        default=None, metadata={"help": "Instruction prepended to queries."}
    )
    query_instruction_format: str = field(
        default="{}{}", metadata={"help": "Format string for query instruction."}
    )
    knowledge_distillation: bool = field(
        default=True,
        metadata={"help": "Use knowledge distillation when teacher scores are available."}
    )
    passage_instruction_for_retrieval: Optional[str] = field(
        default=None, metadata={"help": "Instruction prepended to passages."}
    )
    passage_instruction_format: Optional[str] = field(
        default="{}{}", metadata={"help": "Format string for passage instruction."}
    )
    shuffle_ratio: float = field(
        default=0.0, metadata={"help": "Ratio of text shuffling augmentation."}
    )

    same_dataset_within_batch: bool = field(
        default=False, metadata={"help": "All samples in a batch come from the same dataset."}
    )
    small_threshold: int = field(
        default=0,
        metadata={"help": "Threshold for merging small datasets in same directory."}
    )
    drop_threshold: int = field(
        default=0,
        metadata={"help": "Drop merged small dataset if size below this threshold."}
    )

    def __post_init__(self):
        if self.query_instruction_format and "\\n" in self.query_instruction_format:
            self.query_instruction_format = self.query_instruction_format.replace("\\n", "\n")
        if self.passage_instruction_format and "\\n" in self.passage_instruction_format:
            self.passage_instruction_format = self.passage_instruction_format.replace("\\n", "\n")

        if self.train_data is not None:
            for train_dir in self.train_data:
                if not os.path.exists(train_dir):
                    raise FileNotFoundError(f"Training data not found: {train_dir}")


@dataclass
class TrainConfig(TrainingArguments):
    negatives_cross_device: bool = field(
        default=False, metadata={"help": "Share negatives across devices."}
    )
    temperature: Optional[float] = field(
        default=0.02, metadata={"help": "Temperature for similarity score scaling."}
    )
    fix_position_embedding: bool = field(
        default=False, metadata={"help": "Freeze position embedding parameters."}
    )
    sentence_pooling_method: str = field(
        default='cls',
        metadata={
            "help": "Pooling method: cls, mean, or last_token.",
            "choices": ['cls', 'mean', 'last_token']
        }
    )
    normalize_embeddings: bool = field(
        default=True, metadata={"help": "Normalize embedding vectors."}
    )
    sub_batch_size: Optional[int] = field(
        default=None, metadata={"help": "Sub-batch size during encoding for memory efficiency."}
    )
    kd_loss_type: str = field(
        default='kl_div',
        metadata={
            "help": "Knowledge distillation loss type: kl_div or m3_kd_loss.",
            "choices": ['kl_div', 'm3_kd_loss']
        }
    )
    distill_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for KD loss. Final loss = CE + distill_loss_weight * KD. Set to 0 to disable."}
    )
    no_in_batch_neg_flag: bool = field(
        default=False,
        metadata={"help": "Force disable in-batch negatives for all batches."},
    )
