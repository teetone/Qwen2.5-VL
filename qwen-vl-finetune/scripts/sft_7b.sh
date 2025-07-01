#!/usr/bin/env bash
# Fine-tune Qwen-2.5-VL-7B on Robo-Reward (≈4 258 pairs)

############################
# Distributed / DeepSpeed
############################
NPROC_PER_NODE=1                       # 1 GPU
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
DEEPSPEED_CFG="./scripts/zero3.json"   # same ZeRO-3 config as before

############################
# Model & data paths
############################
MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"
DATASETS="robo_reward_bench%100"       # use full 4 k set
ENTRY="qwenvl/train/train_qwen.py"

############################
# Hyper-parameters
############################
LR=3e-6                   # slightly lower than 3B run
PER_GPU_BATCH=2           # 7 B + video fits comfortably in 24 GB RAM
GRAD_ACCUM=32             # → global batch =   64
SCHED="cosine_with_restarts"
NUM_CYCLES=2
MAX_STEPS=3000            # ≈45 epochs, same as good 3B run
EVAL_STEPS=250            # evaluate/save ~12×
WDECAY=0.02
WARMUP=0.05
LOG_STEPS=25
RUN_NAME="qwen2vl-7b-robo-ft"
OUTPUT_DIR="./output_7b_6_29"

############################
# Argument string
############################
ARGS="
 --deepspeed ${DEEPSPEED_CFG} \
 --model_name_or_path ${MODEL_ID} \
 --dataset_use ${DATASETS} \
 --data_flatten True \
 --tune_mm_vision False \
 --tune_mm_mlp True \
 --tune_mm_llm True \
 --bf16 \
 --output_dir ${OUTPUT_DIR} \
 --max_steps ${MAX_STEPS} \
 --per_device_train_batch_size ${PER_GPU_BATCH} \
 --per_device_eval_batch_size $((PER_GPU_BATCH * 2)) \
 --gradient_accumulation_steps ${GRAD_ACCUM} \
 --max_pixels 50176 \
 --min_pixels 784 \
 --evaluation_strategy steps \
 --eval_steps ${EVAL_STEPS} \
 --save_strategy steps \
 --save_steps ${EVAL_STEPS} \
 --save_total_limit 3 \
 --load_best_model_at_end True \
 --metric_for_best_model train_loss \
 --greater_is_better False \
 --learning_rate ${LR} \
 --weight_decay ${WDECAY} \
 --warmup_ratio ${WARMUP} \
 --max_grad_norm 1 \
 --lr_scheduler_type ${SCHED} \
 --num_cycles ${NUM_CYCLES} \
 --logging_steps ${LOG_STEPS} \
 --model_max_length 8192 \
 --gradient_checkpointing True \
 --dataloader_num_workers 4 \
 --run_name ${RUN_NAME} \
 --report_to wandb
"

############################
# Launch
############################
torchrun \
  --nproc_per_node=${NPROC_PER_NODE} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  ${ENTRY} ${ARGS}
