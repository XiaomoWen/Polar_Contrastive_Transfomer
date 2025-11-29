#!/usr/bin/env bash
# ==============================================================
# Polar Contrastive Transformer (PCT)
# 一键训练 + 测试脚本 (L40 优化 + 完整日志版，去掉 --use_amp)
# ==============================================================

# ---------------- 基础配置 ----------------
name="PCT"

# 路径配置：train_dir = 训练集, test_dir = 测试集
train_dir="/home/ps/Code/wxm_project/Polar_Contrastive_Transfomer/main_project/datasets/University-Release/train"
test_dir="/home/ps/Code/wxm_project/Polar_Contrastive_Transfomer/main_project/datasets/University-Release/test"
pretrain_path="/home/ps/Code/wxm_project/Polar_Contrastive_Transfomer/main_project/pretrain_model/vit_small_patch16_rope_224_naver_in1k.pth"

# ---------------- 训练参数设置 (针对 L40 调优) ----------------
gpu_ids=0
num_worker=16          # L40 机器一般 CPU 也不错，可以开多点
lr=0.0015
sample_num=1
block=3
batchsize=192            # 如果显存有压力可以改为 48 或 32
triplet_loss=0.3
num_epochs=120
pad=0
views=2
warmup_epochs=10
backbone="VIT-S-RoPE-OFFICIAL"

use_autocast=true       # 使用 torch.autocast 混合精度（train.py 支持 --autocast）

# ---------------- 日志输出 (修复：使用绝对路径) ----------------
# 使用 $(pwd) 获取当前绝对路径，防止后续 cd 命令导致相对路径失效
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
    # 有些环境没有 nvcc，不要因为这一行挂掉脚本
    if command -v nvcc >/dev/null 2>&1; then
        echo "CUDA: $(nvcc --version | grep release | sed 's/.*release //')"
    else
        echo "CUDA: (nvcc not found, likely CUDA runtime only)"
    fi
    echo "PyTorch: $(python -c 'import torch;print(torch.__version__)')"
    echo "Python: $(python -V)"
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
  $( [ "$use_autocast" = true ] && echo "--autocast" ) \
  2>&1 | tee "$train_log"

train_end=$(date +%s)
train_dur=$((train_end - train_start))
printf "✅ Training completed in %d minutes.\n\n" $((train_dur/60)) | tee -a "$summary_log"

# 如果训练失败，checkpoints 目录有可能没创建，这里做个存在性检查
if [ ! -d "checkpoints/$name" ]; then
  echo "❌ checkpoints/$name 不存在，说明训练阶段没有成功产生模型，测试阶段跳过。" | tee -a "$summary_log"
  exit 1
fi

# ==============================================================
#                       TEST
# ==============================================================

echo "======= [2/2] TESTING START =======" | tee -a "$summary_log"
test_start=$(date +%s)

cd "checkpoints/$name" || { echo "❌ 路径 checkpoints/$name 不存在"; exit 1; }

# 确保 pretrain_model 可访问（软链接一次自动处理）
[ ! -d "pretrain_model" ] && ln -s ../../pretrain_model pretrain_model

for ((i=109; i<=num_epochs; i+=10)); do
  echo "======= Testing checkpoint: net_${i}.pth =======" | tee -a "$test_log"
  if [ ! -f "net_${i}.pth" ]; then
    echo "⚠️  net_${i}.pth 不存在，跳过该 checkpoint。" | tee -a "$test_log"
    continue
  fi
  for ((j=1; j<3; j++)); do
    python ../../test_server.py \
      --test_dir "$test_dir" \
      --checkpoint "net_${i}.pth" \
      --mode "$j" \
      --gpu_ids "$gpu_ids" \
      --num_worker "$num_worker" \
      --pad "$pad" 2>&1 | tee -a "$test_log"
  done
done

test_end=$(date +%s)
test_dur=$((test_end - test_start))
printf "✅ All tests finished. Time cost: %d minutes.\n\n" $((test_dur/60)) | tee -a "$summary_log"

# ==============================================================
#                  总结
# ==============================================================
echo "Logs saved to: $log_root" | tee -a "$summary_log"
echo "===== Job Finished at $(date) =====" | tee -a "$summary_log"