# -*- coding: utf-8 -*-
"""
@File    : make_optimizer.py
@Author  : Xiaomo Wen, NJUST-Automation 
@Date    : 2025-11-27
@Purpose : 配置优化器与学习率策略 (支持 PCT + ViT-RoPE)
@Desc    : 
    1. 差分学习率: Backbone (0.3x) vs Heads (1.0x)
    2. Warmup + Cosine Annealing 调度器
"""

import math
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP


class WarmupCosineAnnealingLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_max, warmup_epochs, eta_min=1e-6, last_epoch=-1):
        """
        Args:
            optimizer: 优化器
            T_max: 总 Epoch 数
            warmup_epochs: 预热轮数
            eta_min: 最小学习率
        """
        self.T_max = T_max
        self.warmup_epochs = warmup_epochs
        self.eta_min = eta_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # 1. Warmup 阶段: 线性增长
        if self.last_epoch < self.warmup_epochs:
            scale = (self.last_epoch + 1) / max(self.warmup_epochs, 1)
            return [base_lr * scale for base_lr in self.base_lrs]

        # 2. Cosine Annealing 阶段: 余弦下降
        curr_epoch = self.last_epoch - self.warmup_epochs
        total_cosine_epochs = max(self.T_max - self.warmup_epochs, 1)
        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * curr_epoch / total_cosine_epochs)) / 2
            for base_lr in self.base_lrs
        ]


def make_optimizer(model, opt):
    """
    构建优化器和调度器:
    - AdamW
    - 差分学习率 (Backbone vs Heads)
    - Warmup + Cosine Annealing
    """
    # 0. DDP 适配
    if isinstance(model, DDP):
        real_model = model.module
    else:
        real_model = model

    # 1. 参数分组
    ignored_params = []   # 要单独分组的 backbone 参数
    encoders = []

    if hasattr(real_model, 'aerial_encoder'):
        encoders.append(real_model.aerial_encoder)
    elif hasattr(real_model, 'model_1'):
        encoders.append(real_model.model_1)

    if hasattr(real_model, 'ground_encoder'):
        if id(real_model.ground_encoder) != id(real_model.aerial_encoder):
            encoders.append(real_model.ground_encoder)
    elif hasattr(real_model, 'model_2'):
        if id(real_model.model_2) != id(real_model.model_1):
            encoders.append(real_model.model_2)

    for encoder in encoders:
        if hasattr(encoder, 'transformer'):
            ignored_params += list(map(id, encoder.transformer.parameters()))
        elif hasattr(encoder, 'features'):
            ignored_params += list(map(id, encoder.features.parameters()))
        elif hasattr(encoder, 'backbone'):  # OfficialRoPEViTSmall
            ignored_params += list(map(id, encoder.backbone.parameters()))
        else:
            print("Warning: Encoder backbone grouping failed (no transformer/features/backbone).")

    base_params = filter(lambda p: id(p) in ignored_params, real_model.parameters())
    extra_params = filter(lambda p: id(p) not in ignored_params, real_model.parameters())

    # 2. AdamW + 差分 LR
    base_lr = opt.lr
    backbone_lr = 0.3 * base_lr

    print(f'=> Optimizer Strategy: AdamW with Differential Learning Rates.')
    print(f'   - Backbone LR: {backbone_lr:.6f} (0.3x)')
    print(f'   - Heads LR:    {base_lr:.6f} (1.0x)')

    optimizer_ft = optim.AdamW([
        {'params': base_params, 'lr': backbone_lr},
        {'params': extra_params, 'lr': base_lr},
    ], weight_decay=0.05)

    # 3. Warmup + Cosine
    total_epochs = getattr(opt, 'num_epochs', 160)
    warmup_epochs = getattr(opt, 'warmup_epochs', 10)

    print(f'=> Scheduler Strategy: Warmup ({warmup_epochs} eps) + Cosine Annealing')
    print(f'   - Total Epochs: {total_epochs}')
    print(f'   - Min LR: 1e-6')

    exp_lr_scheduler = WarmupCosineAnnealingLR(
        optimizer_ft,
        T_max=total_epochs,
        warmup_epochs=warmup_epochs,
        eta_min=1e-6
    )

    return optimizer_ft, exp_lr_scheduler