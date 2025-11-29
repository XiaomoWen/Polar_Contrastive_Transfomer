#!/bin/bash

# ================= 配置区域 =================
# 任务名称 (会自动创建 logs/PCT_DDP 文件夹)
name="PCT_DDP_8GPU"

# 数据集路径
data_dir="datasets/University-Aug/train"
test_dir="datasets/University-Aug/test"

# 预训练权重路径 (确保路径正确)
pretrain_path="/home/ps/Code/wxm_project/Polar_Contrastive_Transfomer/main_project/pretrain_model/vit_small_p16_224-15ec54c9.pth"

# 显卡设置
# 这里的 gpu_ids 用于指定 torchrun 使用哪些卡，例如 "0,1,2,3,4,5,6,7"
gpu_ids="0,1,2,3,4,5,6,7"
num_gpus=8  # 显卡数量，必须和 gpu_ids 对应

# DDP 训练参数
# 注意：这里的 batchsize 是【单张卡】的 BatchSize
# L40 显存很大，建议 64。总 BatchSize = 64 * 8 = 512
batchsize=64 
num_worker=4

# 学习率策略 (Linear Scaling Rule)
# Base: BS=8 -> LR=0.01
# Now:  BS=512 -> LR=0.64 (理论值)
# 实战建议：0.3 ~ 0.4 (配合 Warmup)
lr=0.3 

# 其他参数
sample_num=1
block=3
triplet_loss=0.3
num_epochs=120
pad=0
views=2

# ================= 1. 开始 DDP 训练 =================
echo "------------------------------------------------------------------"
echo "🚀 Start Distributed Training on $num_gpus GPUs (L40)"
echo "Global Batch Size: $(($batchsize * $num_gpus))"
echo "Per-GPU Batch Size: $batchsize"
echo "Learning Rate: $lr"
echo "------------------------------------------------------------------"

# 设置可见显卡
export CUDA_VISIBLE_DEVICES=$gpu_ids

# 使用 torchrun 启动
# --nproc_per_node: 使用的 GPU 数量
# --master_port: 防止端口冲突，随便写个 29500 左右的
torchrun --nproc_per_node=$num_gpus --master_port=29505 train_ddp.py \
  --name $name \
  --data_dir $data_dir \
  --num_worker $num_worker \
  --views $views \
  --lr $lr \
  --sample_num $sample_num \
  --block $block \
  --batchsize $batchsize \
  --triplet_loss $triplet_loss \
  --num_epochs $num_epochs \
  --pretrain_path $pretrain_path \
  --gpu_ids $gpu_ids # 虽然 train_ddp 不用这个参数来选卡了，但传进去防止报错

# ================= 2. 训练结束，开始评估 =================
# 评估通常不需要多卡并行，用第一张卡（GPU 0）跑就行了
echo "------------------------------------------------------------------"
echo "✅ Training Finished. Start Evaluation on GPU ${gpu_ids:0:1}..."
echo "------------------------------------------------------------------"

cd checkpoints/$name || exit

# 这里稍微修改了原来的逻辑：
# 1. 现在的 test_server.py 应该也是指定单卡运行
# 2. 我们只测试最后几个保存的模型
for ((i=110; i<=$num_epochs; i+=10));
do
    # 检查文件是否存在
    if [ -f "net_$i.pth" ]; then
        echo "Testing model: net_$i.pth"
        
        # Mode 1: Drone -> Satellite
        # Mode 2: Satellite -> Drone
        for ((j = 1; j < 3; j++));
        do
             # 注意：这里显式指定只用第一张卡跑测试，避免多卡冲突
             CUDA_VISIBLE_DEVICES=${gpu_ids:0:1} python test_server.py \
                --test_dir $test_dir \
                --name $name \
                --gpu_ids 0 \
                --batchsize 32 \
                --num_worker $num_worker \
                --pad $pad \
                --mode $j \
                --checkpoint net_$i.pth
        done
    else
        echo "Model net_$i.pth not found, skipping..."
    fi
done
echo "------------------------------------------------------------------"
echo "🎉 All Done! Congratulations! "
echo "------------------------------------------------------------------"