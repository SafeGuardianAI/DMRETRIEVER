#!/bin/bash
# Fine-tune a bidirectional decoder model on domain-specific data.
# Uses LoRA + no in-batch negatives (group negatives only).
set -e

NUM_GPUS=2
MODEL=output/pretrain/stage2_contrastive/merged   # merged Qwen3Bi after PT Stage 2
DATA_DIR=data/finetune                             # directory of JSONL files
OUTPUT_DIR=output/finetune/decoder

python -m torch.distributed.run --nproc_per_node $NUM_GPUS --master_port 29607 \
  -m DMRetriever.train \
    --arch_type qwen3bi \
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
    --learning_rate 5.9e-5 \
    --bf16 \
    --num_train_epochs 12 \
    --dataloader_drop_last True \
    --warmup_ratio 0.01 \
    --logging_steps 2 \
    --save_steps 10 \
    --temperature 0.01 \
    --sentence_pooling_method mean \
    --normalize_embeddings True \
    --kd_loss_type kl_div \
    --report_to none \
    --gradient_accumulation_steps 64 \
    --per_device_train_batch_size 4 \
    --weight_decay 0.005 \
    --no_in_batch_neg_flag True \
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --deepspeed configs/ds_stage1.json
