#!/usr/bin/env bash
# ==============================================================
# Polar Contrastive Transformer (PCT)
# 一键训练 + 测试脚本 (L40 优化 + 强鲁棒性版 Robust V1)
# ==============================================================

# ---------------- 基础配置 ----------------
name="PCT_Robust_V2"

# 路径配置：train_dir = 训练集, test_dir = 测试集
train_dir="/home/ps/Code/wxm_project/Polar_Contrastive_Transfomer/main_project/datasets/University-Release/train"
test_dir="/home/ps/Code/wxm_project/Polar_Contrastive_Transfomer/main_project/datasets/University-Release/test"
pretrain_path="/home/ps/Code/wxm_project/Polar_Contrastive_Transfomer/main_project/pretrain_model/vit_small_patch16_rope_224_naver_in1k.pth"

# ---------------- 训练参数设置 (针对 L40 调优) ----------------
gpu_ids=7
num_worker=16          
lr=0.08
sample_num=1
block=3
batchsize=96            
triplet_loss=0.3
num_epochs=120
pad=0
views=2
warmup_epochs=10
backbone="VIT-S-RoPE-OFFICIAL"
h=384
w=384

# === 核心鲁棒性增强参数 ===
erasing_p=0.25         # 强遮挡增强
drop_path_rate=0.25   # DropPath 随机深度增强         
use_autocast=false      
# ========================

# ---------------- 日志输出 ----------------
log_root="$(pwd)/logs/${name}"
mkdir -p "$log_root"

ts="$(date +'%Y%m%d_%H%M')"
train_log="${log_root}/train_${ts}.log"
test_log="${log_root}/test_${ts}.log"
summary_log="${log_root}/summary_${ts}.log"

start_time=$(date +%s)

# ==============================================================
#                环境信息记录
# ==============================================================
echo "===== Environment Info =====" | tee -a "$summary_log"
{
    echo "Date: $(date)"
    echo "Host: $(hostname)"
    echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -n 1)"
    if command -v nvcc >/dev/null 2>&1; then
        echo "CUDA: $(nvcc --version | grep release | sed 's/.*release //')"
    else
        echo "CUDA: (nvcc not found)"
    fi
    echo "PyTorch: $(python -c 'import torch;print(torch.__version__)')"
    echo "Config: DropPath=${drop_path_rate}, Erase=${erasing_p}"
    echo "-----------------------------"
} | tee -a "$summary_log"

# ==============================================================
#                       TRAIN
# ==============================================================

echo "======= [1/2] TRAINING START =======" | tee -a "$summary_log"
train_start=$(date +%s)

python train.py \
  --name "$name" \
  --data_dir "$train_dir" \
  --gpu_ids "$gpu_ids" \
  --num_worker "$num_worker" \
  --views "$views" \
  --lr "$lr" \
  --sample_num "$sample_num" \
  --block "$block" \
  --batchsize "$batchsize" \
  --triplet_loss "$triplet_loss" \
  --num_epochs "$num_epochs" \
  --pretrain_path "$pretrain_path" \
  --warmup_epochs "$warmup_epochs" \
  --backbone "$backbone" \
  --erasing_p "$erasing_p" \
  --drop_path_rate "$drop_path_rate" \
  --h "$h" \
  --w "$w" \
  $( [ "$use_autocast" = true ] && echo "--autocast" ) \
  2>&1 | tee "$train_log"

train_end=$(date +%s)
train_dur=$((train_end - train_start))
printf "✅ Training completed in %d minutes.\n\n" $((train_dur/60)) | tee -a "$summary_log"

# 如果训练失败，checkpoints 目录有可能没创建
if [ ! -d "checkpoints/$name" ]; then
  echo "❌ checkpoints/$name 不存在，训练失败。" | tee -a "$summary_log"
  exit 1
fi

# ==============================================================
#                       TEST (Modified)
# ==============================================================

echo "======= [2/2] TESTING START (Best Loss Model) =======" | tee -a "$summary_log"
test_start=$(date +%s)

cd "checkpoints/$name" || { echo "❌ 路径 checkpoints/$name 不存在"; exit 1; }

# 确保 pretrain_model 可访问
[ ! -d "pretrain_model" ] && ln -s ../../pretrain_model pretrain_model

# [修改] 仅测试 net_best_loss.pth
echo "======= Testing Best Loss Checkpoint: net_best_loss.pth =======" | tee -a "$test_log"

if [ ! -f "net_best_loss.pth" ]; then
  echo "⚠️  net_best_loss.pth 不存在，可能训练未完成或未触发保存条件。" | tee -a "$test_log"
else
  # 测试 mode 1 和 2 (根据你原本的脚本)
  for ((j=1; j<3; j++)); do
    echo "  >> Testing Mode $j ..."
    python ../../test_server.py \
      --test_dir "$test_dir" \
      --checkpoint "net_best_loss.pth" \
      --mode "$j" \
      --gpu_ids "$gpu_ids" \
      --num_worker "$num_worker" \
      --pad "$pad" 2>&1 | tee -a "$test_log"
  done
fi

test_end=$(date +%s)
test_dur=$((test_end - test_start))
printf "✅ Best Loss Model test finished. Time cost: %d minutes.\n\n" $((test_dur/60)) | tee -a "$summary_log"

# ==============================================================
#                  总结
# ==============================================================
echo "Logs saved to: $log_root" | tee -a "$summary_log"
echo "===== Job Finished at $(date) =====" | tee -a "$summary_log"
