#!/usr/bin/env bash
# Pre-train Stage 1: Masked Next Token Prediction (MNTP) for decoder models.
# Converts a causal LM into a bidirectional encoder using LoRA.
set -euo pipefail

MODEL=Qwen/Qwen3-4B                          # base causal LM
TRAIN_FILE=data/pretrain_text/corpus.txt      # plain text, one sentence per line
OUTPUT_DIR=output/pretrain/stage1_mntp

python -m pretrain.run_mntp \
  --model_name_or_path $MODEL \
  --train_file $TRAIN_FILE \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 16 \
  --do_train \
  --do_eval \
  --max_seq_length 512 \
  --mask_token_type blank \
  --data_collator_type default \
  --mlm_probability 0.2 \
  --overwrite_output_dir \
  --output_dir $OUTPUT_DIR \
  --save_steps 25 \
  --lora_r 16 \
  --eval_strategy steps \
  --eval_steps 25 \
  --torch_dtype bfloat16 \
  --attn_implementation eager \
  --report_to none \
  --num_train_epochs 10 \
  --logging_steps 25 \
  --learning_rate 5e-5 \
  --warmup_ratio 0.01 \
  --bf16
