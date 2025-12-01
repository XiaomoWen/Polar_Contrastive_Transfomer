#!/usr/bin/env bash
# ==============================================================
# Polar Contrastive Transformer (PCT)
# 一键训练 + 测试脚本 (L40 优化 + 完整日志版，去掉 --use_amp)
# ==============================================================

# ---------------- 基础配置 ----------------
name="PCT_1"

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
triplet_loss=2
num_epochs=160
pad=0
views=2
warmup_epochs=10
backbone="VIT-S-RoPE-OFFICIAL"
erasing_p=0.1
triplet_weight=4.5
use_autocast=false      # 使用 torch.autocast 混合精度（train.py 支持 --autocast）


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
