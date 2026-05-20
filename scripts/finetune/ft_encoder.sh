#!/bin/bash
# Fine-tune an encoder model on domain-specific data.
# Uses no in-batch negatives (group negatives only).
set -e

NUM_GPUS=2
MODEL=output/pretrain/encoder/checkpoint-best  # pre-trained encoder checkpoint
DATA_DIR=data/finetune                          # directory of JSONL files
OUTPUT_DIR=output/finetune/encoder

python -m torch.distributed.run --nproc_per_node $NUM_GPUS --master_port 29606 \
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
    --learning_rate 4e-5 \
    --bf16 \
    --num_train_epochs 40 \
    --dataloader_drop_last True \
    --warmup_ratio 0.01 \
    --logging_steps 50 \
    --save_steps 10 \
    --temperature 0.01 \
    --sentence_pooling_method mean \
    --normalize_embeddings True \
    --kd_loss_type kl_div \
    --report_to none \
    --gradient_accumulation_steps 8 \
    --per_device_train_batch_size 64 \
    --weight_decay 0.005 \
    --no_in_batch_neg_flag True
