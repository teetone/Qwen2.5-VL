#!/usr/bin/env bash
# Fine-tune Qwen-2.5-VL-3B on Robo-Reward bench (400 examples)

############################
# Distributed / DeepSpeed
############################
NPROC_PER_NODE=1                 # one GPU; raise if you have more
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

DEEPSPEED_CFG="./scripts/zero3.json"

############################
# Model & data paths
############################
MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct"
DATASETS="robo_reward_bench%100"              # 100 % of the 400-example set
ENTRY="qwenvl/train/train_qwen.py"

############################
# Hyper-parameters
############################
LR=1e-5                        # full-parameter / adapter FT; use 2e-4 if LoRA-only
PER_GPU_BATCH=4
GRAD_ACCUM=4                   # → global batch = 16
MAX_STEPS=3000                 # ~120 “epochs” with global batch 16
EVAL_STEPS=100
SAVE_STEPS=${EVAL_STEPS}

############################
# Output / tracking
############################
RUN_NAME="qwen2vl-3b-robo-ft"
OUTPUT_DIR="./output_robo_reward"

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
 --save_steps ${SAVE_STEPS} \
 --save_total_limit 2 \
 --load_best_model_at_end \
 --metric_for_best_model eval_loss --greater_is_better False \
 --learning_rate ${LR} \
 --weight_decay 0 \
 --warmup_ratio 0.03 \
 --max_grad_norm 1 \
 --lr_scheduler_type cosine \
 --logging_steps 10 \
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
