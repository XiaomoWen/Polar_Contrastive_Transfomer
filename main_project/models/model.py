# -*- coding: utf-8 -*-
"""
@File    : model.py
@Author  : Xiaomo Wen, NJUST-Automation
@Date    : 2025-11-25
@Purpose : 定义多视图跨模态检索网络架构 (Multi-View Cross-Modal Network Architecture)
@Model   : PCT-Net (Polar-Contrastive Transformer)
@Desc    : 
    本文件作为模型的总入口，负责根据配置 (opt.views) 组装不同的网络结构。
    主要包含两种架构：
    1. DualViewNet: 用于双视图匹配 (如 Satellite <-> Drone)
    2. ThreeViewNet: 用于三视图匹配 (如 Satellite <-> Street <-> Drone)
"""

import torch.nn as nn

# =====================================================================
# 模块导入
# =====================================================================
# 从 PCT 文件夹导入核心构建函数
# 该函数负责构建基于 ViT + Polar Transform 的特征提取器
from .PCT.make_model import make_transformer_model


class DualViewNet(nn.Module):
    """
    双视图网络 (Dual-View Network)
    
    适用场景:
        - 仅包含两种俯视视角的任务 (如 CVUSA, CVACT 数据集)。
        - 任务类型: Satellite (卫星) <-> Drone (无人机/全景)。
        
    设计理念:
        - 由于两种视图都是“空中俯视”视角，几何特征相似度高。
        - 因此采用【完全权重共享】策略，使用同一个 Encoder 提取特征。
    """
    
    def __init__(self, opt, class_num, block=4, return_f=False):
        """
        初始化双视图网络
        
        Args:
            opt: 配置参数对象
            class_num (int): 分类类别数 (通常对应 ID 数量)
            block (int): Transformer 的层数或 block 索引
            return_f (bool): 是否返回原始特征 (用于 Triplet Loss 计算)
        """
        super(DualViewNet, self).__init__()

        print('=> [Model Strategy] Using PCT (Polar Transform) strictly for Dual-View.')
        
        # 构建核心编码器 (Aerial Encoder)
        # 负责提取空中视角图像的特征
        self.aerial_encoder = make_transformer_model(
            opt, 
            num_class=class_num, 
            block=block, 
            return_f=return_f
        )

    def forward(self, img_satellite, img_drone):
        """
        前向传播逻辑
        
        Args:
            img_satellite (Tensor): 卫星图像 batch
            img_drone (Tensor): 无人机图像 batch
            
        Returns:
            feat_satellite, feat_drone: 提取后的特征向量
        """
        
        # -------------------------------------------------------
        # 分支 1: 处理卫星图像
        # -------------------------------------------------------
        if img_satellite is None:
            feat_satellite = None
        else:
            # 卫星图送入空中编码器
            feat_satellite = self.aerial_encoder(img_satellite)

        # -------------------------------------------------------
        # 分支 2: 处理无人机图像
        # -------------------------------------------------------
        if img_drone is None:
            feat_drone = None
        else:
            # 无人机图也送入同一个空中编码器 (Weight Sharing)
            # 这种共享机制强迫网络学习两种视图共有的语义特征
            feat_drone = self.aerial_encoder(img_drone)
            
        return feat_satellite, feat_drone


class ThreeViewNet(nn.Module):
    """
    三视图网络 (Three-View Network)
    
    适用场景:
        - 包含三种视角的复杂匹配任务 (如 University-1652 数据集)。
        - 任务类型: Satellite (卫星) <-> Street (街景) <-> Drone (无人机)。
        
    设计理念:
        - 非对称结构设计。
        - 空中视角 (Sat/Drone) 使用 aerial_encoder。
        - 地面视角 (Street) 使用 ground_encoder。
        - 支持通过 share_weight 参数控制是否强制共享所有权重。
    """
    
    def __init__(self, opt, class_num, share_weight=False, block=4, return_f=False):
        """
        Args:
            share_weight (bool): 是否强制街景分支也共享空中分支的权重。
                                 默认 False, 因为街景和航拍差异巨大, 通常独立训练效果更好。
        """
        super(ThreeViewNet, self).__init__()
        self.share_weight = share_weight

        print('=> [Model Strategy] Using PCT (Polar Transform) strictly for Three-View.')
        
        # 1. 构建空中视角编码器 (用于 Satellite 和 Drone)
        self.aerial_encoder = make_transformer_model(opt, num_class=class_num, block=block, return_f=return_f)

        # 2. 构建地面视角编码器 (用于 Street)
        if self.share_weight:
            # 策略 A: 强制共享 (所有视角用同一个网络)
            # 优点: 参数量少; 缺点: 街景和航拍特征差异大，可能难以收敛
            print('=> [Note] Weights are shared between Aerial and Ground views.')
            self.ground_encoder = self.aerial_encoder
        else:
            # 策略 B: 独立权重 (地面视角用单独的网络)
            # 优点: 能更好地学习地面特有的特征; 缺点: 参数量增加
            self.ground_encoder = make_transformer_model(opt, num_class=class_num, block=block, return_f=return_f)


    def forward(self, img_satellite, img_street, img_drone, img_extra=None): 
        """
        前向传播逻辑
        注意: 参数顺序必须与 DataLoader 输出顺序严格一致
        """
        
        # -------------------------------------------------------
        # 1. 卫星分支 (空中视角 -> Aerial Encoder)
        # -------------------------------------------------------
        if img_satellite is None:
            feat_satellite = None
        else:
            feat_satellite = self.aerial_encoder(img_satellite)

        # -------------------------------------------------------
        # 2. 街景分支 (地面视角 -> Ground Encoder)
        # -------------------------------------------------------
        if img_street is None:
            feat_street = None
        else:
            # 这里使用了专门的地面编码器 (除非 share_weight=True)
            feat_street = self.ground_encoder(img_street)

        # -------------------------------------------------------
        # 3. 无人机分支 (空中视角 -> Aerial Encoder)
        # -------------------------------------------------------
        if img_drone is None:
            feat_drone = None
        else:
            # 无人机和卫星长得像，复用 aerial_encoder
            feat_drone = self.aerial_encoder(img_drone)

        # -------------------------------------------------------
        # 4. 额外数据分支 (用于扩展实验)
        # -------------------------------------------------------
        if img_extra is None:
            return feat_satellite, feat_street, feat_drone
        else:
            # 假设额外数据也是地面的，使用 ground_encoder
            feat_extra = self.ground_encoder(img_extra)
            return feat_satellite, feat_street, feat_drone, feat_extra


def make_model(opt):
    """
    模型工厂函数 (Factory Method)
    
    功能:
        根据配置文件中的 views 参数，实例化对应的网络类。
    
    Args:
        opt: 全局配置对象
        
    Returns:
        model: 实例化后的 PyTorch 模型对象
    """
    # 打印构建信息，方便日志排查
    print(f"Building PCT Model | Views: {opt.views}, Block: {opt.block}")

    if opt.views == 2:
        # 对应 CVUSA / CVACT
        model = DualViewNet(opt, opt.nclasses, block=opt.block, return_f=opt.triplet_loss)
    elif opt.views == 3:
        # 对应 University-1652
        model = ThreeViewNet(opt, opt.nclasses, share_weight=opt.share, block=opt.block, return_f=opt.triplet_loss)
    else:
        raise ValueError(f"Unsupported number of views: {opt.views}. Only 2 or 3 are supported.")
        
    return model