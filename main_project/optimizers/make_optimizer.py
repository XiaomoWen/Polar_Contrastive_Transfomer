# -*- coding: utf-8 -*-
"""
@File    : make_optimizer.py
@Author  : Xiaomo Wen, NJUST-Automation
@Date    : 2025-11-25
@Purpose : 配置优化器与学习率策略
@Desc    : 
    本模块实现了"差分学习率" (Differential Learning Rate) 策略。
    由于 Backbone (如 ViT) 已经在大规模数据集(ImageNet)上预训练过，
    而分类头(Classifier)和极坐标变换层(Polar)是随机初始化的，
    因此我们对 Backbone 使用较小的学习率 (0.3x)，对其他部分使用正常学习率 (1.0x)。
"""

import torch.optim as optim
from torch.optim import lr_scheduler

def make_optimizer(model, opt):
    """
    构建优化器和调度器

    Args:
        model: 实例化后的网络模型 (DualViewNet 或 ThreeViewNet)
        opt: 配置参数对象 (包含 lr, steps 等)

    Returns:
        optimizer_ft: 配置好的优化器
        exp_lr_scheduler: 学习率调度器
    """
    
    # ============================================================
    # 1. 参数分组 (Parameter Grouping)
    # 目标：找出所有属于 Backbone (预训练部分) 的参数 ID
    # ============================================================
    ignored_params = []
    
    # 获取模型中的 Encoder 列表
    # 注意：这里适配了我们在 model.py 中修改后的新命名 (aerial_encoder, ground_encoder)
    # 如果你的 model.py 还没改，请把下面的 aerial_encoder 改回 model_1
    encoders = []
    
    if hasattr(model, 'aerial_encoder'):
        encoders.append(model.aerial_encoder)
    
    if hasattr(model, 'ground_encoder'):
        # 只有 ThreeViewNet 且不共享权重时才有 ground_encoder
        # 如果共享权重，ground_encoder 指向的就是 aerial_encoder，已经在列表里了，不用重复加
        if id(model.ground_encoder) != id(model.aerial_encoder):
            encoders.append(model.ground_encoder)
            
    # 遍历所有 Encoder，提取其中 transformer 部分的参数 ID
    for encoder in encoders:
        # 假设 encoder 内部有名为 'transformer' 的 backbone 属性
        if hasattr(encoder, 'transformer'):
            ignored_params += list(map(id, encoder.transformer.parameters()))
        else:
            print("Warning: Encoder has no attribute 'transformer'. Check backbone definition.")

    # ============================================================
    # 2. 过滤参数
    # ============================================================
    # base_params: Backbone 的参数 (预训练过的) -> 用小 LR
    # extra_params: 分类头、Polar层的参数 (新加的) -> 用大 LR
    
    # id(p) not in ignored_params -> 说明是新加的层
    extra_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    
    # id(p) in ignored_params -> 说明是骨干层的参数
    base_params = filter(lambda p: id(p) in ignored_params, model.parameters())

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
    # MultiStepLR: 在指定的 epoch (milestones) 处让学习率衰减
    # 例如 milestones=[40, 70], gamma=0.1
    # 0-39 epoch: lr = 初始值
    # 40-69 epoch: lr = 初始值 * 0.1
    # 70+ epoch:   lr = 初始值 * 0.01
    exp_lr_scheduler = lr_scheduler.MultiStepLR(
        optimizer_ft, 
        milestones=opt.steps, 
        gamma=0.1
    )
    
    # 备选方案 (已注释):
    # StepLR: 每隔固定步数衰减
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=80, gamma=0.1)
    
    return optimizer_ft, exp_lr_scheduler