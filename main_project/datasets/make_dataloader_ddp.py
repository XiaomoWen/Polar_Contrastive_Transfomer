# -*- coding: utf-8 -*-
# datasets/make_dataloader_ddp.py

import torch
from torchvision import transforms
# 注意这里用了相对引用，确保此文件在 datasets 文件夹下
from .Dataloader_University import Sampler_University, Dataloader_University, train_collate_fn
from .random_erasing import RandomErasing
from .autoaugment import ImageNetPolicy

def make_dataset_ddp(opt):
    """
    DDP 版本的 Dataset 构建函数
    区别：为了配合 Triplet Loss 的 PK 采样，我们不使用 DistributedSampler，
    而是保留 Sampler_University，让每个 GPU 独立进行 PK 采样。
    """

    ######################################################################
    # 1. 定义数据增强 transforms (保持原版一致)
    ######################################################################
    
    transform_train_list = [
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.Pad(opt.pad, padding_mode='edge'),
        transforms.RandomCrop((opt.h, opt.w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    transform_satellite_list = [
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.Pad(opt.pad, padding_mode='edge'),
        transforms.RandomAffine(90),
        transforms.RandomCrop((opt.h, opt.w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    transform_val_list = [
        transforms.Resize(size=(opt.h, opt.w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    # 随机擦除
    if opt.erasing_p > 0:
        transform_train_list = transform_train_list + [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]

    # 颜色抖动
    if opt.color_jitter:
        transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list
        transform_satellite_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_satellite_list

    # AutoAugment
    if opt.DA:
        transform_train_list = [ImageNetPolicy()] + transform_train_list

    # 打印一次 transforms 信息 (只在主进程打印，防止刷屏，外部 train_ddp.py 控制)
    # print(transform_train_list) 

    data_transforms = {
        'train': transforms.Compose(transform_train_list),
        'val': transforms.Compose(transform_val_list),
        'satellite': transforms.Compose(transform_satellite_list)
    }

    ######################################################################
    # 2. 加载数据集 (Custom Dataset)
    ######################################################################

    image_datasets = Dataloader_University(opt.data_dir, transforms=data_transforms)
    
    # 关键点：保留 Sampler_University
    # 注意：这里的 opt.batchsize 指的是 Per-GPU BatchSize (例如 64)
    # 这里的 opt.sample_num 是 K (每个 ID 采几张图)
    sampler = Sampler_University(image_datasets, batchsize=opt.batchsize, sample_num=opt.sample_num)
    
    # 构建 DataLoader
    # 注意：
    # 1. shuffle=False (因为 sampler 已经负责了乱序)
    # 2. sampler=sampler (使用自定义采样器)
    # 3. num_workers=opt.num_worker (单卡的工作进程数)
    dataloaders = torch.utils.data.DataLoader(
        image_datasets, 
        batch_size=opt.batchsize,
        sampler=sampler,
        num_workers=opt.num_worker, 
        pin_memory=True,
        collate_fn=train_collate_fn
    )
    
    # 计算数据集大小（这只是个估计值，用于计算 epoch 进度，不影响训练）
    # 这里的逻辑是：所有 ID * sample_num
    dataset_sizes = {x: len(image_datasets) * opt.sample_num for x in ['satellite', 'drone']}
    
    class_names = image_datasets.cls_names
    
    return dataloaders, class_names, dataset_sizes