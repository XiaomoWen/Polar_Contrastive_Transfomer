# -*- coding: utf-8 -*-
# train_ddp.py

from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import random
import numpy as np

from torch.autograd import Variable
from torch.cuda.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn
import time
from optimizers.make_optimizer import make_optimizer
from models.model import make_model
# 注意：这里需要改一下引用，后面会说 make_dataloader 怎么改
from datasets.make_dataloader_ddp import make_dataset_ddp as make_dataset 
from tool.utils_server import save_network, copyfiles2checkpoints
import warnings
from losses.triplet_loss import Tripletloss
from losses.cal_loss import cal_kl_loss, cal_loss, cal_triplet_loss

warnings.filterwarnings("ignore")

def setup_distributed():
    # 从环境变量初始化 DDP
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
        dist.barrier()
        return local_rank, rank, world_size
    else:
        print("Not using DDP, falling back to single GPU")
        return 0, 0, 1

def cleanup():
    dist.destroy_process_group()

def get_parse():
    parser = argparse.ArgumentParser(description='Training')
    # gpu_ids 不再需要，由 torchrun 控制
    parser.add_argument('--name', default='test_ddp', type=str, help='output model name')
    parser.add_argument('--data_dir', default='University-Release/train', type=str, help='training dir path')
    parser.add_argument('--train_all', action='store_true', help='use all training data')
    parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training')
    parser.add_argument('--num_worker', default=4, type=int, help='num_workers per GPU') # 建议设为 4
    parser.add_argument('--batchsize', default=32, type=int, help='batchsize PER GPU') # 单卡 BS
    parser.add_argument('--pad', default=0, type=int, help='padding')
    parser.add_argument('--h', default=256, type=int, help='height')
    parser.add_argument('--w', default=256, type=int, help='width')
    parser.add_argument('--views', default=2, type=int, help='the number of views')
    parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability')
    parser.add_argument('--warm_epoch', default=0, type=int, help='warm up epochs')
    parser.add_argument('--lr', default=0.01, type=float, help='base learning rate') # 这里可以稍微调大，比如 0.02 或 0.04
    parser.add_argument('--moving_avg', default=1.0, type=float, help='moving average')
    parser.add_argument('--DA', action='store_true', help='use Color Data Augmentation')
    parser.add_argument('--fp16', action='store_true', default=False, help='use float16') # 建议用 autocast，这里保持 False
    parser.add_argument('--autocast', action='store_true', default=True, help='use mix precision')
    parser.add_argument('--block', default=1, type=int, help='')
    parser.add_argument('--kl_loss', action='store_true', default=False, help='kl_loss')
    parser.add_argument('--triplet_loss', default=0.3, type=float, help='')
    parser.add_argument('--sample_num', default=1, type=float, help='num of repeat sampling')
    parser.add_argument('--num_epochs', default=120, type=int, help='')
    parser.add_argument('--steps', default=[70, 110], type=int, help='')
    parser.add_argument('--backbone', default="VIT-S", type=str, help='')
    parser.add_argument('--pretrain_path', default="", type=str, help='')
    # DDP 必须参数
    parser.add_argument('--local_rank', default=-1, type=int) 
    opt = parser.parse_args()
    return opt

def train_model(model, opt, optimizer, scheduler, dataloaders, dataset_sizes, local_rank):
    num_epochs = opt.num_epochs
    
    # 只在主进程初始化 loss 和 scaler
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss().cuda(local_rank)
    loss_kl = nn.KLDivLoss(reduction='batchmean').cuda(local_rank)
    triplet_loss = Tripletloss(margin=opt.triplet_loss).cuda(local_rank)

    # 训练循环
    for epoch in range(num_epochs):
        # 关键：如果是 DDP 采样器，每个 epoch 需要 set_epoch
        # 但因为我们用的是自定义 Sampler，我们需要确保它每次 shuffle 的随机性
        # 在 make_dataloader 修改版中，我们会让 Sampler 自动处理
        
        if local_rank == 0:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

        model.train(True)  # Set model to training mode

        running_cls_loss = 0.0
        running_triplet = 0.0
        running_kl_loss = 0.0
        running_loss = 0.0
        running_corrects = 0.0
        running_corrects2 = 0.0
        
        # 计时
        start_time = time.time()

        for i, (data, data2, data3) in enumerate(dataloaders):
            inputs, labels = data
            inputs2, labels2 = data2
            inputs3, labels3 = data3
            
            # 放到当前 GPU
            inputs = inputs.cuda(local_rank, non_blocking=True)
            inputs3 = inputs3.cuda(local_rank, non_blocking=True)
            labels = labels.cuda(local_rank, non_blocking=True)
            labels3 = labels3.cuda(local_rank, non_blocking=True)
            
            now_batch_size = inputs.shape[0]

            optimizer.zero_grad()

            with autocast(enabled=opt.autocast):
                outputs, outputs2 = model(inputs, inputs3)
                
                # Triplet Loss
                f_triplet_loss = torch.tensor(0.0).cuda(local_rank)
                if opt.triplet_loss > 0:
                    features = outputs[1]
                    features2 = outputs2[1]
                    # 注意：这里的 batchsize 是单卡的
                    split_num = opt.batchsize // opt.sample_num 
                    f_triplet_loss = cal_triplet_loss(features, features2, labels, triplet_loss, split_num)
                    outputs = outputs[0]
                    outputs2 = outputs2[0]

                # CE Loss
                cls_loss = cal_loss(outputs, labels, criterion) + cal_loss(outputs2, labels3, criterion)
                
                # KL Loss
                kl_loss = torch.tensor(0.0).cuda(local_rank)
                if opt.kl_loss:
                    kl_loss = cal_kl_loss(outputs, outputs2, loss_kl)

                loss = kl_loss + cls_loss + f_triplet_loss

            # Backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 统计 (只在 rank 0 打印或者聚合后打印，这里简化为只统计 rank 0 的大概情况)
            # 严谨的做法是用 dist.all_reduce 聚合所有卡 loss，但这会拖慢速度，只看 rank 0 足够了
            if local_rank == 0:
                running_loss += loss.item() * now_batch_size
                running_cls_loss += cls_loss.item() * now_batch_size
                running_triplet += f_triplet_loss.item() * now_batch_size
                
                # 计算精度 (简化)
                _, preds = torch.max(outputs.data, 1)
                _, preds2 = torch.max(outputs2.data, 1)
                running_corrects += float(torch.sum(preds == labels.data))
                running_corrects2 += float(torch.sum(preds2 == labels3.data))

        # Epoch 结束后的操作
        scheduler.step()
        
        if local_rank == 0:
            epoch_loss = running_loss / dataset_sizes['satellite'] # 注意：这里分母其实不准了，因为只统计了单卡
            epoch_acc = running_corrects / dataset_sizes['satellite']
            epoch_acc2 = running_corrects2 / dataset_sizes['satellite']
            
            lr_backbone = optimizer.state_dict()['param_groups'][0]['lr']
            
            print('Train Loss: {:.4f} Cls: {:.4f} Trip: {:.4f} Sat_Acc: {:.4f} Drone_Acc: {:.4f} LR: {:.6f} Time: {:.0f}s'.format(
                epoch_loss, epoch_cls_loss/dataset_sizes['satellite'], epoch_triplet_loss/dataset_sizes['satellite'],
                epoch_acc, epoch_acc2, lr_backbone, time.time() - start_time))

            # 保存模型
            if epoch % 10 == 9 and epoch >= 10: # 改早一点保存测试
                save_network(model.module, opt.name, epoch) # 注意保存 model.module

if __name__ == '__main__':
    opt = get_parse()
    
    # 1. DDP 初始化
    local_rank, rank, world_size = setup_distributed()
    
    # 2. 调整 BatchSize 和 LR
    # 这里的 opt.batchsize 是单卡的，比如 64
    # 我们自动根据 GPU 数量缩放 LR (Linear Scaling Rule)
    base_lr = opt.lr # 假设 base_lr 是针对单卡 batch=8 设计的
    # 简单的缩放策略：
    # 如果你原来 BS=8, LR=0.01. 现在单卡 BS=64, 8卡并行. Global BS = 512.
    # 建议在这里写死一个新的 LR，或者用公式：
    # opt.lr = opt.lr * (opt.batchsize * world_size / 8.0) 
    # 但为了稳妥，建议你手动指定 --lr 0.1 左右先试试
    
    cudnn.benchmark = True

    # 3. 数据集 (重点修改了这里)
    # 传入 world_size 和 rank 供 Sampler 切分数据
    opt.world_size = world_size
    opt.rank = rank
    dataloaders, class_names, dataset_sizes = make_dataset(opt)
    opt.nclasses = len(class_names)

    # 4. 模型构建
    model = make_model(opt)
    
    # 5. 加上 SyncBN (这对 L40 大 Batch 非常重要)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda(local_rank)
    
    # 6. DDP 包装
    # find_unused_parameters=True 有时能解决报错，但会稍微慢一点点
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # 7. 优化器
    optimizer_ft, exp_lr_scheduler = make_optimizer(model, opt)

    # 8. 开始训练
    if local_rank == 0:
        print(f"Start Training with {world_size} GPUs. Total BatchSize = {opt.batchsize * world_size}")
        copyfiles2checkpoints(opt)

    train_model(model, opt, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes, local_rank)
    
    cleanup()