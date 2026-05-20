import logging
from pathlib import Path

from transformers import HfArgumentParser, set_seed

from .config import ModelConfig, DataConfig, TrainConfig
from .model_loader import load_tokenizer, load_backbone, save_merged_lora
from .biencoder import BiEncoderModel
from .dataset import (
    PairDataset, PairBatchDataset,
    PairCollator, BatchCollator,
    EpochRefreshCallback,
)
from .trainer import BiEncoderTrainer

logger = logging.getLogger(__name__)

POOLING_DEFAULTS = {
    "encoder": "cls",
    "decoder": "last_token",
    "qwen3bi": "mean",
}


def main():
    parser = HfArgumentParser((ModelConfig, DataConfig, TrainConfig))
    model_cfg, data_cfg, train_cfg = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if train_cfg.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, fp16: %s",
        train_cfg.local_rank, train_cfg.device, train_cfg.n_gpu,
        bool(train_cfg.local_rank != -1), train_cfg.fp16,
    )
    logger.info("Model config: %s", model_cfg)
    logger.info("Data config: %s", data_cfg)
    logger.info("Training config: %s", train_cfg)

    set_seed(train_cfg.seed)

    tokenizer, resize = load_tokenizer(model_cfg)

    if model_cfg.only_merge_lora_model:
        if train_cfg.process_index == 0:
            save_merged_lora(model_cfg, train_cfg.output_dir)
        return

    backbone = load_backbone(model_cfg, train_cfg.output_dir, tokenizer, resize)

    pooling = train_cfg.sentence_pooling_method
    if not pooling or pooling == "cls":
        pooling = POOLING_DEFAULTS.get(model_cfg.arch_type, "cls")
    train_cfg.sentence_pooling_method = pooling

    model = BiEncoderModel(
        backbone=backbone,
        tokenizer=tokenizer,
        negatives_cross_device=train_cfg.negatives_cross_device,
        temperature=train_cfg.temperature,
        sub_batch_size=train_cfg.sub_batch_size,
        kd_loss_type=train_cfg.kd_loss_type,
        distill_loss_weight=train_cfg.distill_loss_weight,
        sentence_pooling_method=pooling,
        normalize_embeddings=train_cfg.normalize_embeddings,
    )

    if train_cfg.gradient_checkpointing:
        model.enable_input_require_grads()

    if train_cfg.fix_position_embedding:
        for name, param in model.named_parameters():
            if "position_embeddings" in name:
                logger.info(f"Freeze position embedding: {name}")
                param.requires_grad = False

    if data_cfg.same_dataset_within_batch:
        train_dataset = PairBatchDataset(
            args=data_cfg,
            default_batch_size=train_cfg.per_device_train_batch_size,
            seed=train_cfg.seed,
            tokenizer=tokenizer,
            process_index=train_cfg.process_index,
            num_processes=train_cfg.world_size,
        )
        train_cfg.per_device_train_batch_size = 1
        train_cfg.dataloader_num_workers = 0
        CollatorClass = BatchCollator
    else:
        train_dataset = PairDataset(args=data_cfg, tokenizer=tokenizer)
        CollatorClass = PairCollator

    data_collator = CollatorClass(
        tokenizer=tokenizer,
        query_max_len=data_cfg.query_max_len,
        passage_max_len=data_cfg.passage_max_len,
        sub_batch_size=train_cfg.sub_batch_size,
        pad_to_multiple_of=data_cfg.pad_to_multiple_of,
        padding=True,
        return_tensors="pt",
        default_no_in_batch_neg_flag=train_cfg.no_in_batch_neg_flag,
    )

    trainer = BiEncoderTrainer(
        model=model,
        args=train_cfg,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    if data_cfg.same_dataset_within_batch:
        trainer.add_callback(EpochRefreshCallback(train_dataset))

    Path(train_cfg.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.train(resume_from_checkpoint=train_cfg.resume_from_checkpoint)
    trainer.save_model()

    if model_cfg.save_merged_lora_model and train_cfg.process_index == 0:
        save_merged_lora(model_cfg, train_cfg.output_dir)


if __name__ == "__main__":
    main()
