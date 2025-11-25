# -*- coding: utf-8 -*-
"""
@File    : polar_transform.py
@Author  : Xiaomo Wen, NJUST-Automation
@Date    : 2025-11-25
@Purpose : 实现可微极坐标变换模块 (Differentiable Polar Transform Module)
@Model   : PCT-Net (Polar-Contrastive Transformer)
@Desc    : 
    该模块利用 PyTorch 的 grid_sample 实现从笛卡尔坐标系到极坐标系的图像变换。
    其核心作用是将 "俯视中心对称" 的卫星图像 (Satellite) 转换为 
    "全景条带状" 的图像 (Panorama-like)，以在几何上对齐地面/无人机全景图。

    变换逻辑:
    - Input (Cartesian): [B, C, H, W] -> 通常是正方形卫星图 (B means Batch, C means Channels)
    - Output (Polar):    [B, C, H, W] -> 展开后的条带图
        - Output Height (H) 对应: 半径 (Radius) 方向
        - Output Width  (W) 对应: 角度 (Azimuth) 方向
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PolarTransform(nn.Module):
    """
    极坐标变换层 (Polar Transform Layer)
    
    继承自 nn.Module，可以作为一个标准的神经网络层被插入到任何模型中。
    它包含一个自动缓存机制，如果输入图像尺寸不变，不会重复计算采样网格。
    """
    
    def __init__(self, input_shape=(256, 256)):
        """
        初始化极坐标变换层

        Args:
            input_shape (tuple): 预期的输入图像尺寸 (Height, Width)。
                                 主要用于预计算采样网格。
        """
        super(PolarTransform, self).__init__()
        self.h, self.w = input_shape
        
        # 预计算网格 (Lazy Initialization)
        # 将其注册为 buffer，意味着它不是可训练参数，但会随模型保存/加载
        # 注意：如果不想保存 grid 到 state_dict，可以不注册，直接 self.grid = ...
        self.register_buffer('grid', self._build_grid(self.h, self.w))

    def _build_grid(self, h, w):
        """
        构建采样网格 (Sampling Grid Generator)
        
        原理:
        我们需要知道输出图像上每一个像素 (u, v) 对应原图上的哪个坐标 (x, y)。
        
        Mapping:
        - 输出图像的宽 (W轴) -> 对应极角 theta [-pi, pi]
        - 输出图像的高 (H轴) -> 对应半径 r [0, 1]
        
        Args:
            h (int): 输出图像高度 (半径分辨率)
            w (int): 输出图像宽度 (角度分辨率)

        Returns:
            grid (Tensor): [H, W, 2] 形状的采样坐标场，数值范围 [-1, 1]
        """
        # 1. 创建极坐标系基准
        # theta: 角度场，形状 [H, W]，每一行都是 -pi 到 pi
        theta = torch.linspace(-np.pi, np.pi, w).view(1, -1).repeat(h, 1)
        # r: 半径场，形状 [H, W]，每一列都是 0 到 1
        r = torch.linspace(0, 1, h).view(-1, 1).repeat(1, w)

        # 2. 极坐标 -> 笛卡尔坐标转换
        # 公式: x = r * cos(theta), y = r * sin(theta)
        # 结果范围: 
        # cos/sin 在 [-1, 1], r 在 [0, 1] -> x, y 理论范围 [-1, 1]
        # 这正好符合 F.grid_sample 对 grid 输入的要求 ((-1,-1)是左上角, (1,1)是右下角)
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)

        # 3. 堆叠坐标
        # unsqueeze(2) 将 [H, W] 变为 [H, W, 1]
        # cat 后变为 [H, W, 2]
        grid = torch.cat((x.unsqueeze(2), y.unsqueeze(2)), 2)
        
        return grid

    def forward(self, x):
        """
        前向传播 (Forward Pass)

        Args:
            x (Tensor): 输入图像张量, 形状 [Batch, Channel, Height, Width]

        Returns:
            x_polar (Tensor): 变换后的极坐标图像, 形状 [Batch, Channel, Height, Width]
        """
        # 1. 动态尺寸检查 (Dynamic Shape Check)
        # 防止输入图片尺寸突然改变导致网格不匹配 (例如 inference 时图片变大)
        if x.shape[2] != self.h or x.shape[3] != self.w:
            device = x.device
            self.h, self.w = x.shape[2], x.shape[3]
            # 重新计算网格并移动到对应设备
            new_grid = self._build_grid(self.h, self.w).to(device)
            # 更新 buffer
            self.grid = new_grid
            # 如果使用了 register_buffer，这里重新赋值会自动处理
        
        # 2. 设备对齐
        # 确保网格和输入数据在同一个 GPU 上
        if self.grid.device != x.device:
            self.grid = self.grid.to(x.device)

        # 3. 批次扩展 (Batch Expansion)
        # grid 原始形状: [H, W, 2]
        # 需要扩展为: [Batch, H, W, 2] 以匹配输入数据的 Batch Size
        batch_size = x.shape[0]
        grid = self.grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        # 4. 双线性插值采样 (Bilinear Sampling)
        # F.grid_sample 是这一步的核心，它根据 grid 里的坐标去 x 里“抓”像素
        # padding_mode='zeros': 采样点如果超出原图圆的范围，填 0 (黑色)
        # align_corners=True:  严格对齐角点像素，对于几何变换通常选 True
        x_polar = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        return x_polar