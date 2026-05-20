#!/bin/bash
# Pre-train encoder (e.g., BERT / XLM-RoBERTa) with contrastive learning.
# Uses in-batch negatives + cross-device negatives.
set -e

NUM_GPUS=3
MODEL=bert-base-uncased                      # or xlm-roberta-base, etc.
DATA_DIR=data/pretrain                        # directory of JSONL files (see README for format)
OUTPUT_DIR=output/pretrain/encoder

torchrun --nproc_per_node $NUM_GPUS \
  -m DMRetriever.train \
    --arch_type encoder \
    --model_name_or_path $MODEL \
    --cache_dir ./cache/model \
    --train_data $DATA_DIR \
    --cache_path ./cache/data \
    --train_group_size 10 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --query_instruction_for_retrieval 'Represent this sentence for searching relevant passages: ' \
    --query_instruction_format '{}: {}' \
    --knowledge_distillation False \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --learning_rate 3e-4 \
    --bf16 \
    --num_train_epochs 2 \
    --dataloader_drop_last True \
    --warmup_ratio 0.01 \
    --logging_steps 10 \
    --save_steps 10 \
    --temperature 0.01 \
    --sentence_pooling_method mean \
    --normalize_embeddings True \
    --kd_loss_type kl_div \
    --report_to none \
    --gradient_accumulation_steps 1 \
    --per_device_train_batch_size 244 \
    --weight_decay 0.01 \
    --negatives_cross_device \
    --same_dataset_within_batch True \
    --deepspeed configs/ds_stage1.json \
    --gradient_checkpointing
