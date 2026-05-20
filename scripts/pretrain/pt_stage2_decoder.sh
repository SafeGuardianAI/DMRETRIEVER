#!/bin/bash
# Pre-train Stage 2: Contrastive learning for the bidirectional decoder model.
# Requires a model that has been through Stage 1 (MNTP), with LoRA merged.
set -e

NUM_GPUS=4
MODEL=output/pretrain/stage1_mntp/merged      # merged Qwen3Bi model after MNTP
DATA_DIR=data/pretrain                         # directory of JSONL files
OUTPUT_DIR=output/pretrain/stage2_contrastive

torchrun --nproc_per_node $NUM_GPUS --master_port 29601 \
  -m DMRetriever.train \
    --arch_type qwen3bi \
    --model_name_or_path $MODEL \
    --cache_dir ./cache/model \
    --train_data $DATA_DIR \
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
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
    --learning_rate 4e-5 \
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
    --per_device_train_batch_size 48 \
    --weight_decay 0.01 \
    --negatives_cross_device \
    --same_dataset_within_batch True \
    --deepspeed configs/ds_stage1.json \
    --gradient_checkpointing
