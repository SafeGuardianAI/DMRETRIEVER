#!/bin/bash
# Evaluate encoder pre-training checkpoints across multiple GPUs.
# Splits checkpoint range into windows and runs them in parallel.

MODEL_DIR=output/pretrain/encoder             # directory containing checkpoint-* subdirs
POOL=mean
CKPT_TYPE=full                                # "full" for encoder, "lora" for LoRA checkpoints

START=0
END=200
NUM_WINDOWS=3
NUM_GPUS_AVAILABLE=3

TOTAL=$((END - START))
STEP=$((TOTAL / NUM_WINDOWS))

mkdir -p logs

for ((i=0; i<$NUM_WINDOWS; i++)); do
  MIN_CKPT=$((START + i * STEP))

  if [ $i -eq $((NUM_WINDOWS - 1)) ]; then
    MAX_CKPT=$END
  else
    MAX_CKPT=$((MIN_CKPT + STEP))
  fi

  PHYSICAL_GPU_ID=$((i % NUM_GPUS_AVAILABLE))

  echo "Launching window $i: $MIN_CKPT to $MAX_CKPT on GPU $PHYSICAL_GPU_ID"

  CUDA_VISIBLE_DEVICES=$PHYSICAL_GPU_ID \
  python eva/orchestrate/ckpt_batch.py \
    --model_dir $MODEL_DIR \
    --pool $POOL \
    --eva_test eva \
    --discard_intermediate \
    --ckpt_type $CKPT_TYPE \
    --min_ckpt $MIN_CKPT \
    --max_ckpt $MAX_CKPT \
    --batch 8 \
    --skip_done \
    > logs/eva_window_${MIN_CKPT}_${MAX_CKPT}.log 2>&1 &

  sleep 7
done

wait
echo "All evaluation windows completed."
