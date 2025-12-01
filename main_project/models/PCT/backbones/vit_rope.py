# models/PCT/backbones/vit_rope.py
# 官方 RoPE-ViT-Small 封装：内部调用 timm 的 vit_small_patch16_rope_224.naver_in1k

import torch
import torch.nn as nn
import timm


class OfficialRoPEViTSmall(nn.Module):
    """
    使用 timm 的 'vit_small_patch16_rope_224.naver_in1k' 作为 backbone。
    - 结构与官方权重 100% 一致
    - forward 返回的是特征（通常是 CLS pooled 特征），由上层 PCT 做投影/分类
    """
    def __init__(self, pretrained_ckpt_path=None):
        super().__init__()

        # 1. 创建 timm 模型（不自动下载预训练，用我们本地 ckpt）
        self.backbone = timm.create_model(
            'vit_small_patch16_rope_224.naver_in1k',
            pretrained=False,    # 不从网络下载，手动 load
            num_classes=1000,     # 与 ImageNet 预训练对齐
            img_size=384         # 修改输入尺寸为 384x384
        )

        # 2. 加载本地预训练权重
        if pretrained_ckpt_path is not None:
            print(f'=> [OfficialRoPEViTSmall] loading pretrain from: {pretrained_ckpt_path}')
            state_dict = torch.load(pretrained_ckpt_path, map_location='cpu')

            # 兼容几种常见格式：{'state_dict': {...}} / {'model': {...}} / 直接 state_dict
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            if isinstance(state_dict, dict) and 'model' in state_dict:
                state_dict = state_dict['model']

            missing, unexpected = self.backbone.load_state_dict(
                state_dict, strict=False
            )
            print(f'=> [OfficialRoPEViTSmall] load_state_dict done. '
                  f'missing={len(missing)}, unexpected={len(unexpected)}')

        # 3. 记录特征维度（ViT-Small 是 384）
        self.embed_dim = self.backbone.num_features  # 通常为 384

    def forward(self, x):
        """
        返回 [B, C] 的图像级特征：
        - timm 的 ViT 模型都有 forward_features，用于拿中间特征（CLS）
        """
        feats = self.backbone.forward_features(x)  # [B, C]
        return feats