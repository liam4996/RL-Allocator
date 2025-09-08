#!/bin/bash
set -e

save_dir="$1"                 # Root directory to save results
processed_dataset_dir="$2"    # Root output dir from your process_datasets.sh (contains tokenized_dataset/)

# ------- Resources & training scale (single GPU g5.2x) -------
num_gpu=1
iterative_prune_train_step=2      # Match your data split: stage_1_of_2 / stage_2_of_2
repeat_dataset=1
seperate_strategy=linear
warmup_ratio_int=5

# Conservative settings for 24GB VRAM:
batch_size=16
micro_batch_size=4
gradient_accumulation_steps=$(( ${batch_size} / ${micro_batch_size} ))

lr_scheduler=WSD
learning_rate=2e-5
min_learning_rate=2e-6
max_grad_norm=1
cutoff_len=1024

# ------- Paths & model -------
base_dir="${save_dir}/iterative_prune_train_llama3_1b_g5_2x_${iterative_prune_train_step}_steps_${seperate_strategy}_${lr_scheduler}_maxlr_${learning_rate}"
save_log_name="${base_dir}/log"
save_log_pth="${save_log_name}/log.out"
tmp_result_name="${base_dir}/tmp_result.out"
pruned_model_save_dir="${base_dir}/pruned_model"
trained_model_save_dir="${base_dir}/trained_model"
output_model_save_dir="${base_dir}/output_model"

# Calibration data used for importance scoring (Adapt-Pruner)
calibration_data_path='slimpajama'

# 1B model path: local path is more reliable; if using HF Hub, change to meta-llama/Llama-3.2-1B (requires login)
model_path='/home/ec2-user/models/Llama-3.2-1B'

# Approximate parameter counts (to compute per-iteration target_param_num; adjust as needed)
original_param_num=1240000000       # ~1.24B (rough estimate is fine)
target_param_num=980000000          # Target ~0.7B (example: smaller = more aggressive pruning)

tokenized_dataset_dir="${processed_dataset_dir}/tokenized_dataset"
# Total samples you preprocessed earlier (for estimating training steps); per logs it's 737703
num_samples=737703

mkdir -p "$(dirname "${save_log_pth}")"
exec &> >(tee -a "${save_log_pth}")

echo "Logging path: ${save_log_pth}"
echo "Model path: ${model_path}"
echo "Tokenized dataset: ${tokenized_dataset_dir}"
echo "Original params: ${original_param_num} | Target params: ${target_param_num}"
echo "Iterative steps: ${iterative_prune_train_step} | GPUs: ${num_gpu}"
echo "Batch ${batch_size} = micro ${micro_batch_size} x grad_acc ${gradient_accumulation_steps}"

# Estimate training steps (use *num_gpu instead of a fixed 4)
denominator=$(( ${micro_batch_size} * ${num_gpu} * ${gradient_accumulation_steps} ))
if [ $(( ${num_samples} % ${denominator} )) -ne 0 ]; then remainder=1; else remainder=0; fi
quotient=$(( ${num_samples} / ${denominator} ))
total_training_steps=$(( ${quotient} + ${remainder} ))
base_training_steps=0
total_warmup_steps=$(( (${warmup_ratio_int} * ${total_training_steps}) / 100 ))

echo "Per-iter training steps: ${total_training_steps} (warmup ${total_warmup_steps})"

# ---------- Iterative pruning + brief training ----------
for stage in $(seq 1 ${iterative_prune_train_step}); do
  echo "[START] Prune Stage ${stage}/${iterative_prune_train_step}"
  output_path="${pruned_model_save_dir}/adaptive_prune_stage_${stage}_of_${iterative_prune_train_step}/output_model.bin"
  cur_target_param_num=$(( original_param_num - (original_param_num - target_param_num) * stage / iterative_prune_train_step ))
  echo "Target params this stage: ${cur_target_param_num}"

  # python ../utils/hf_prune.py \
  #   --adpative_prune \
  #   --layer_prune_distribution_amplitude 0.02 \
  #   --iterative_steps 50 \
  #   --base_model ${model_path} \
  #   --calibration_data_path ${calibration_data_path} \
  #   --pruning_ratio 1.00 \
  #   --target_param_num ${cur_target_param_num} \
  #   --device cuda \
  #   --block_wise \
  #   --block_mlp_layer_start 0 \
  #   --block_mlp_layer_end 16 \  # Llama-3.2-1B is commonly ~16 layers; if an error occurs, change to the actual layer count
  #   --block_attention_layer_start 0 \
  #   --block_attention_layer_end 16 \
  #   --save_log_name ${save_log_name} \
  #   --output_pth ${output_path} \
  #   --pruner_type taylor \
  #   --taylor param_first \
  #   --taylor_seq_len 64 \
  #   --num_examples 512 \
  #   --save_model \
  # 
  #   # ---- Enable RL: per-layer fine-tuning with FLOPs reward (±1%), keeping overall sparsity fixed ----
  #   --rl_flops_tune \
  #   --rl_steps 400 --rl_moves_per_step 8 --rl_lr 1e-2 \
  #   --rl_context_len 4096 --rl_group_size 64 \
  #   --rl_max_layer_delta 0.01 \
  #   --rl_validate_every 0  # Do not enable external threshold yet; set to 32 to enable and implement eval_hook in hf_prune.py
  output_path="${pruned_model_save_dir}/adaptive_prune_stage_${stage}_of_${iterative_prune_train_step}/output_model.bin"
  pruned_hf_dir="${pruned_model_save_dir}/adaptive_prune_stage_${stage}_of_${iterative_prune_train_step}/hf_pruned"

  ABS_PY="$HOME/AdaptPruner1/utils/hf_prune.py"
  args=(
    --adpative_prune
    --layer_prune_distribution_amplitude 0.02
    --iterative_steps 50
    --base_model "${model_path}"
    --calibration_data_path "${calibration_data_path}"
    --pruning_ratio 1.00
    --target_param_num "${cur_target_param_num}"
    --device cuda
    --block_wise
    --block_mlp_layer_start 0
    --block_mlp_layer_end 16
    --block_attention_layer_start 0
    --block_attention_layer_end 16
    --save_log_name "${save_log_name}"
    --output_pth "${output_path}"                  # Optional .bin backup
    --save_pretrained_dir "${pruned_hf_dir}"      # ★ Ensure the Hugging Face directory is exported
    --pruner_type taylor
    --taylor param_first
    --taylor_seq_len 64
    --num_examples 512
    --batch_size 16
    --save_model
    --rl_flops_tune
    --rl_steps 400
    --rl_moves_per_step 8
    --rl_lr 1e-2
    --rl_context_len 4096
    --rl_group_size 64
    --rl_max_layer_delta 0.01
    --rl_validate_every 0
  )
  python "$ABS_PY" "${args[@]}"

  echo "[FINISH] Prune Stage ${stage}"

  echo "[START] Train Stage ${stage}"
  train_data_path="${tokenized_dataset_dir}/tokenized_dataset_stage_${stage}_of_${iterative_prune_train_step}"
  train_model_path="${pruned_hf_dir}"   # ★ Use the HF directory, not the .bin
  output_dir="${trained_model_save_dir}/train_stage_${stage}_of_${iterative_prune_train_step}"

  # Single GPU: use accelerate with default config
  accelerate launch --num_processes 1 ../utils/post_train.py \
    --data_path ${train_data_path} \
    --prune_model ${train_model_path} \
    --dataset_tokenized \
    --save_log_name ${save_log_name} \
    --output_dir ${output_dir} \
    --return_pth ${tmp_result_name} \
    --learning_rate ${learning_rate} \
    --min_learning_rate ${min_learning_rate} \
    --lr_scheduler ${lr_scheduler} \
    --batch_size ${batch_size} \
    --micro_batch_size ${micro_batch_size} \
    --train_epochs 1 \
    --resume_previous_stages \
    --total_training_steps ${total_training_steps} \
    --base_training_steps ${base_training_steps} \
    --total_warmup_steps ${total_warmup_steps}

  cur_training_steps=$(cat ${tmp_result_name})
  rm -f ${tmp_result_name}
  echo "Returned steps this iter: ${cur_training_steps}"
  base_training_steps=$(( ${base_training_steps} + ${cur_training_steps} ))
  echo "Accumulated base steps: ${base_training_steps}"

  latest_checkpoint=$(ls -vd ${output_dir}/checkpoint-* | tail -n 1)
  if [ -z "$latest_checkpoint" ]; then
    echo "Error: No checkpoint found in ${output_dir}"
    exit 1
  fi
  model_path="${latest_checkpoint}"
  echo "Use latest checkpoint for next stage: ${model_path}"
  echo "[FINISH] Train Stage ${stage}"
done

echo "[START] Export final"
mkdir -p "${output_model_save_dir}"
cp -r "${model_path}/." "${output_model_save_dir}/"
echo "[FINISH] Final model: ${output_model_save_dir}"
