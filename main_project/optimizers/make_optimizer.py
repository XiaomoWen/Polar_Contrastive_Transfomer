# -*- coding: utf-8 -*-
"""
@File    : make_optimizer.py
@Author  : Xiaomo Wen, NJUST-Automation
@Date    : 2025-11-25
@Purpose : 配置优化器与学习率策略 (DDP 适配版)
@Desc    : 
    本模块实现了"差分学习率" (Differential Learning Rate) 策略。
    由于 Backbone (如 ViT) 已经在大规模数据集(ImageNet)上预训练过，
    而分类头(Classifier)和极坐标变换层(Polar)是随机初始化的，
    因此我们对 Backbone 使用较小的学习率 (0.3x)，对其他部分使用正常学习率 (1.0x)。
"""

import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP

def make_optimizer(model, opt):
    """
    构建优化器和调度器

    Args:
        model: 实例化后的网络模型 (可能是 DDP 包装过的，也可能是原始的)
        opt: 配置参数对象 (包含 lr, steps 等)

    Returns:
        optimizer_ft: 配置好的优化器
        exp_lr_scheduler: 学习率调度器
    """
    
    # ============================================================
    # 0. DDP 适配 (关键修改)
    # ============================================================
    # 如果模型被 DDP 或 DataParallel 包装了，真正的模型在 model.module 里
    if isinstance(model, DDP):
        real_model = model.module
    else:
        real_model = model

    # ============================================================
    # 1. 参数分组 (Parameter Grouping)
    # 目标：找出所有属于 Backbone (预训练部分) 的参数 ID
    # ============================================================
    ignored_params = []
    
    # 获取模型中的 Encoder 列表
    # 注意：这里使用 real_model 来获取属性
    encoders = []
    
    # 注意：请确保这里属性名和你 models/model.py 里定义的一致
    # 你的代码里可能是 model_1/model_2 或者 aerial_encoder/ground_encoder
    # 为了保险，建议你检查一下 models/model.py
    if hasattr(real_model, 'aerial_encoder'):
        encoders.append(real_model.aerial_encoder)
    elif hasattr(real_model, 'model_1'): # 兼容旧命名
        encoders.append(real_model.model_1)
    
    if hasattr(real_model, 'ground_encoder'):
        # 只有 ThreeViewNet 且不共享权重时才有 ground_encoder
        if id(real_model.ground_encoder) != id(real_model.aerial_encoder):
            encoders.append(real_model.ground_encoder)
    elif hasattr(real_model, 'model_2'): # 兼容旧命名
        if id(real_model.model_2) != id(real_model.model_1):
            encoders.append(real_model.model_2)
            
    # 遍历所有 Encoder，提取其中 transformer 部分的参数 ID
    for encoder in encoders:
        # 假设 encoder 内部有名为 'transformer' 的 backbone 属性 (ViTBackbone)
        if hasattr(encoder, 'transformer'):
            ignored_params += list(map(id, encoder.transformer.parameters()))
        elif hasattr(encoder, 'features'): # ResNet 可能是 features
            ignored_params += list(map(id, encoder.features.parameters()))
        else:
            # 这是一个非常重要的调试信息，如果打印出来，说明参数分组失败了
            print("Warning: Encoder has no attribute 'transformer' or 'features'. Differential LR might fail.")

    # ============================================================
    # 2. 过滤参数
    # ============================================================
    # base_params: Backbone 的参数 (预训练过的) -> 用小 LR
    # extra_params: 分类头、Polar层的参数 (新加的) -> 用大 LR
    
    # 使用 real_model.parameters() 确保拿到的是同一个对象的参数
    
    # id(p) not in ignored_params -> 说明是新加的层
    extra_params = filter(lambda p: id(p) not in ignored_params, real_model.parameters())
    
    # id(p) in ignored_params -> 说明是骨干层的参数
    base_params = filter(lambda p: id(p) in ignored_params, real_model.parameters())

    # ============================================================
    # 3. 构建优化器 (SGD)
    # ============================================================
    print(f'=> Optimizer Strategy: SGD with Differential Learning Rates.')
    print(f'   - Backbone LR: {0.3 * opt.lr:.6f} (0.3x)')
    print(f'   - Heads LR:    {opt.lr:.6f} (1.0x)')
    
    optimizer_ft = optim.SGD([
        {'params': base_params, 'lr': 0.3 * opt.lr}, # 核心逻辑：骨干网络学习率打3折
        {'params': extra_params, 'lr': opt.lr}       # 其他部分全速学习
    ], weight_decay=5e-4, momentum=0.9, nesterov=True) # 常用的 SGD 配置

    # ============================================================
    # 4. 构建学习率调度器 (Scheduler)
    # ============================================================
    # 调度器不需要改动，因为它只认 optimizer
    exp_lr_scheduler = lr_scheduler.MultiStepLR(
        optimizer_ft, 
        milestones=opt.steps, 
        gamma=0.1
    )
    
    return optimizer_ft, exp_lr_scheduler