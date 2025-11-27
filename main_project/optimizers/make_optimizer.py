# -*- coding: utf-8 -*-
"""
@File    : make_optimizer.py
@Author  : Xiaomo Wen, NJUST-Automation 
@Date    : 2025-11-27
@Purpose : 配置优化器与学习率策略 (DDP 适配版 + Warmup Cosine)
@Desc    : 
    1. 实现了"差分学习率": Backbone (0.3x) vs Heads (1.0x)
    2. [新增] 实现了 Warmup + Cosine Annealing 策略，
       专为大 BatchSize (BS=256) 设计，解决训练后期收敛停滞问题。
"""

import math
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP

# ============================================================
# 新增：自定义 Warmup + Cosine 退火调度器
# ============================================================
class WarmupCosineAnnealingLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_max, warmup_epochs, eta_min=1e-6, last_epoch=-1):
        """
        Args:
            optimizer: 优化器
            T_max: 总 Epoch 数
            warmup_epochs: 预热轮数 (建议 5-10)
            eta_min: 最小学习率 (建议 1e-6)
        """
        self.T_max = T_max
        self.warmup_epochs = warmup_epochs
        self.eta_min = eta_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # 1. Warmup 阶段: 线性增长
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        
        # 2. Cosine Annealing 阶段: 余弦下降
        else:
            curr_epoch = self.last_epoch - self.warmup_epochs
            total_cosine_epochs = self.T_max - self.warmup_epochs
            return [self.eta_min + (base_lr - self.eta_min) * 
                    (1 + math.cos(math.pi * curr_epoch / total_cosine_epochs)) / 2
                    for base_lr in self.base_lrs]

def make_optimizer(model, opt):
    """
    构建优化器和调度器
    """
    
    # ============================================================
    # 0. DDP 适配
    # ============================================================
    if isinstance(model, DDP):
        real_model = model.module
    else:
        real_model = model

    # ============================================================
    # 1. 参数分组 (保持你原来的逻辑不变)
    # ============================================================
    ignored_params = []
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
        else:
            print("Warning: Encoder grouping failed.")

    extra_params = filter(lambda p: id(p) not in ignored_params, real_model.parameters())
    base_params = filter(lambda p: id(p) in ignored_params, real_model.parameters())

    # ============================================================
    # 2. 构建优化器 (SGD) - 保持不变
    # ============================================================
    print(f'=> Optimizer Strategy: SGD with Differential Learning Rates.')
    print(f'   - Backbone LR: {0.3 * opt.lr:.6f} (0.3x)')
    print(f'   - Heads LR:    {opt.lr:.6f} (1.0x)')
    
    optimizer_ft = optim.SGD([
        {'params': base_params, 'lr': 0.3 * opt.lr}, 
        {'params': extra_params, 'lr': opt.lr}       
    ], weight_decay=5e-4, momentum=0.9, nesterov=True) 

    # ============================================================
    # 3. [核心修改] 构建学习率调度器 (Scheduler)
    # ============================================================
    
    # 获取总 Epoch 数，优先从 opt 获取，如果没有则默认 160 (防止报错)
    total_epochs = getattr(opt, 'num_epochs', 160)
    
    print(f'=> Scheduler Strategy: Warmup ({10} eps) + Cosine Annealing')
    print(f'   - Total Epochs: {total_epochs}')
    print(f'   - Min LR: 1e-6')

    # 使用我们刚刚定义的类替代 MultiStepLR
    exp_lr_scheduler = WarmupCosineAnnealingLR(
        optimizer_ft, 
        T_max=total_epochs,     # 确保这里等于你设置的总训练轮数
        warmup_epochs=opt.warmup_epochs,       # 预热 x 轮，对大 BS 非常重要
        eta_min=1e-6            # 最后降到 0.000001，收得非常干净
    )
    
    # 原来的代码 (已注释掉)
    # exp_lr_scheduler = lr_scheduler.MultiStepLR(
    #     optimizer_ft, 
    #     milestones=opt.steps, 
    #     gamma=0.1
    # )
    
    return optimizer_ft, exp_lr_scheduler