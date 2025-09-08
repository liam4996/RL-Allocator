#!/bin/bash
# file: resume_stage3.sh
set -e

# ==== 你自己的路径 ====
DATA_DIR="/home/ec2-user/AdaptPruner/processed_datasets_llama3_1b_3_steps_repeat_1_strategy_linear_instruct_1/tokenized_dataset"
EXP_DIR="/home/ec2-user/experiments/iterative_prune_train_llama3_1b_3_steps_repeat_1_strategy_linear_4_microbatch_size_WSD_max_lr_1e-5"
CKPT="checkpoint-5966"            # 上一次中断时的 checkpoint 目录名
TOTAL_STEPS=6027                  # 目标总步数（原脚本写死的）
WARMUP_STEPS=$(( TOTAL_STEPS * 5 / 100 ))   # 按 warm-up 5 % 算
BATCH_SIZE=16
MICRO_BATCH=4

accelerate launch --config_file fsdp_config.yaml \
  ../utils/post_train.py \
    --data_path  "${DATA_DIR}/tokenized_dataset_stage_3_of_3" \
    --prune_model "${EXP_DIR}/trained_model/train_stage_3_of_3/${CKPT}" \
    --dataset_tokenized \
    --output_dir  "${EXP_DIR}/trained_model/train_stage_3_of_3" \
    --save_log_name "${EXP_DIR}/log" \
    --learning_rate 1e-5 \
    --min_learning_rate 1e-6 \
    --lr_scheduler WSD \
    --batch_size ${BATCH_SIZE} \
    --micro_batch_size ${MICRO_BATCH} \
    --total_training_steps ${TOTAL_STEPS} \
    --base_training_steps ${CKPT#checkpoint-} \
    --total_warmup_steps ${WARMUP_STEPS} \
    --resume_from_checkpoint "${CKPT}"
