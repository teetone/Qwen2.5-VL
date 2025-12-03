#!/usr/bin/env bash
# Fine-tune Qwen3-VL-4B-Instruct on RoboReward (77k examples) on a single GPU

############################
# Distributed / DeepSpeed
############################
NPROC_PER_NODE=1                 # single GPU (H100, etc.)
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

DEEPSPEED_CFG="./scripts/zero3.json"

############################
# Model & data paths
############################
LLM="Qwen/Qwen3-VL-4B-Instruct"
DATASETS="roboreward"
ENTRY="qwenvl/train/train_qwen.py"

############################
# Hyperparameters
############################
LR=3e-6                          # conservative LR, similar to your Qwen2.5 run
BATCH_SIZE=4                     # per-GPU batch size
GRAD_ACCUM=8                     # global batch = 4 * 8 = 32
NUM_TRAIN_EPOCHS=2              # ~4800 steps over 77k examples
WDECAY=0.05

############################
# Output / tracking
############################
RUN_NAME="qwen3vl-4b-roboreward"
OUTPUT_DIR="./output_qwen3vl_4b_roboreward"

############################
# Argument string
############################
ARGS="
 --deepspeed ${DEEPSPEED_CFG} \
 --model_name_or_path ${LLM} \
 --dataset_use ${DATASETS} \
 --data_flatten True \
 --tune_mm_vision False \
 --tune_mm_mlp True \
 --tune_mm_llm True \
 --bf16 \
 --output_dir ${OUTPUT_DIR} \
 --num_train_epochs ${NUM_TRAIN_EPOCHS} \
 --per_device_train_batch_size ${BATCH_SIZE} \
 --per_device_eval_batch_size $((BATCH_SIZE * 2)) \
 --gradient_accumulation_steps ${GRAD_ACCUM} \
 --max_pixels 50176 \
 --min_pixels 784 \
 --eval_strategy no \
 --save_strategy steps \
 --save_steps 1000 \
 --save_total_limit 2 \
 --learning_rate ${LR} \
 --weight_decay ${WDECAY} \
 --warmup_ratio 0.05 \
 --max_grad_norm 1 \
 --lr_scheduler_type cosine \
 --logging_steps 10 \
 --model_max_length 8192 \
 --gradient_checkpointing True \
 --dataloader_num_workers 4 \
 --run_name ${RUN_NAME} \
 --report_to wandb \
"

############################
# Launch
############################
torchrun \
  --nproc_per_node=${NPROC_PER_NODE} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  ${ENTRY} ${ARGS}
