#!/usr/bin/env bash
# ==============================================================
# Polar Contrastive Transformer (PCT)
# 一键训练 + 测试脚本（绝对路径稳定版）
# ==============================================================
# 作者：fishros，整理 by assistant
# ==============================================================

# ---------------- 基础配置 ----------------
name="PCT"
data_dir="/home/fishros/Code/Polar_Contrastive_Transfomer/main_project/datasets/University-Release/train"
test_dir="/home/fishros/Code/Polar_Contrastive_Transfomer/main_project/datasets/University-Release/test"
pretrain_path="/home/fishros/Code/Polar_Contrastive_Transfomer/main_project/pretrain_model/vit_small_patch16_rope_224_naver_in1k.pth"

gpu_ids=0
num_worker=1
lr=0.0003
sample_num=1
block=3
batchsize=16
triplet_loss=0.3
num_epochs=120
pad=0
views=2
warmup_epochs=10
backbone="VIT-S-RoPE-OFFICIAL"

# ==============================================================
#                       TRAIN
# ==============================================================

echo "======= [1/2] TRAINING START ======="
python train.py \
  --name "$name" \
  --data_dir "$data_dir" \
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
  --backbone "$backbone"

echo "✅ Training completed for ${num_epochs} epochs."
echo " "

# ==============================================================
#                       TEST
# ==============================================================

echo "======= [2/2] TESTING START ======="

cd checkpoints/$name || { echo "❌ 路径 checkpoints/$name 不存在"; exit 1; }

# 确保 pretrain_model 可访问（软链接一次自动处理）
[ ! -d "pretrain_model" ] && ln -s ../../pretrain_model pretrain_model

for ((i=109; i<=num_epochs; i+=10)); do
  echo "======= Testing checkpoint: net_${i}.pth ======="
  for ((j=1; j<3; j++)); do
    python ../../test_server.py \
      --test_dir "$test_dir" \
      --checkpoint "net_${i}.pth" \
      --mode "$j" \
      --gpu_ids "$gpu_ids" \
      --num_worker "$num_worker" \
      --pad "$pad"
  done
done

echo "✅ All tests finished successfully."