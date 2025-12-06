# -*- coding: utf-8 -*-
"""
@File    : make_model.py
@Author  : Xiaomo Wen, NJUST-Automation (with RoPE-ViT support)
@Date    : 2025-11-25
@Purpose : 构建 PCT-Net 的核心网络架构
@Model   : PCT-Net (Polar-Contrastive Transformer)
@Desc    : 
    本文件组装了以下核心组件：
    1. Backbone: ViT-Small (FSRA 版本) / ViT-Small-RoPE-OFFICIAL / VAN
    2. LPN Head: 局部特征分块策略 (Local Part-based Network)
"""

import torch
import os
import torch.nn as nn
import torch.nn.functional as F

# 导入骨干网络
from .backbones.vit_pytorch import vit_small_patch16_224_FSRA
from .backbones.van import van_small
from .backbones.vit_rope import OfficialRoPEViTSmall  # 官方 RoPE-ViT 封装


class Gem_heat(nn.Module):
    """
    广义平均池化层 (Generalized-Mean Pooling)
    用于特征聚合，通过可学习参数 p 动态调整池化策略。
    """
    def __init__(self, dim=768, p=3, eps=1e-6):
        super(Gem_heat, self).__init__()
        # p 是可学习参数，初始化为 3
        self.p = nn.Parameter(torch.ones(dim) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p)

    def gem(self, x, p=3):
        # 类似于 Softmax 的加权操作
        p = F.softmax(p).unsqueeze(-1)
        x = torch.matmul(x, p)
        x = x.view(x.size(0), x.size(1))
        return x


class ClassBlock(nn.Module):
    """
    分类器模块 (Classification Block)
    标准的 [Linear -> BN -> LeakyReLU -> Dropout -> Linear] 结构
    """
    def __init__(self, input_dim, class_num, droprate,
                 relu=False, bnorm=True, num_bottleneck=512,
                 linear=True, return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        
        # 特征降维层 (Bottleneck)
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
            
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
            
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)  # 凯明初始化

        # 最终分类层 (Classifier)
        classifier = [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        # 先经过 Bottleneck 层
        x = self.add_block(x)
        
        if self.training:
            # 训练时：如果需要 Triplet Loss，则返回特征 f 和分类 logits
            if self.return_f:
                f = x
                x = self.classifier(x)
                return x, f
            else:
                # 仅做 CrossEntropy Loss
                x = self.classifier(x)
                return x
        else:
            # 测试时：直接返回特征用于检索 (不需要过分类层)
            return x


def weights_init_kaiming(m):
    """Kaiming 初始化 (适用于 ReLU/LeakyReLU 网络)"""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    """分类层初始化 (通常用正态分布)"""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


class build_transformer(nn.Module):
    """
    PCT-Net 主体网络结构
    
    流程:
    Input -> ViT/VAN Backbone -> Global/Local Features -> Classifiers
    """
    def __init__(self, opt, num_classes, block=4, return_f=False):
        super(build_transformer, self).__init__()
        self.return_f = return_f
        self.block = block

        # 默认 ViT 预训练权重（如果 opt.pretrain_path 没设置就用这个）
        default_vit_pretrain = "/home/fishros/Code/Polar_Contrastive_Transfomer/main_project/pretrain_model/vit_small_p16_224-15ec54c9.pth"
        if not hasattr(opt, "pretrain_path") or opt.pretrain_path in [None, ""]:
            opt.pretrain_path = default_vit_pretrain

        # =====================================================
        # 【Step 1】初始化 Backbone (特征提取)
        # =====================================================
        if opt.backbone == "VIT-S":
            model_path = opt.pretrain_path
            transformer_name = "vit_small_patch16_224_FSRA"
            self.in_planes = 768  # ViT-Small 的输出维度

            print('=> using Transformer_type: ViT-Small (FSRA) as backbone')
            print(f'=> pretrain: {model_path}')

            # 构建 ViT
            self.transformer = vit_small_patch16_224_FSRA(
                img_size=(256, 256), 
                stride_size=[16, 16], 
                drop_path_rate=0.1,
                drop_rate=0.0, 
                attn_drop_rate=0.0
            )
            
            # 加载 ImageNet 预训练权重
            if os.path.exists(model_path):
                self.transformer.load_param(model_path)
            else:
                print(f"Warning: Pretrain model {model_path} not found.")

        elif opt.backbone == "VIT-S-RoPE-OFFICIAL":
            # 使用 timm 官方 RoPE-ViT-Small
            model_path = opt.pretrain_path
            print('=> using Transformer_type: ViT-Small-RoPE-OFFICIAL (timm) as backbone')
            print(f'=> pretrain: {model_path}')

            self.transformer = OfficialRoPEViTSmall(
                pretrained_ckpt_path=model_path
            )
            # 官方 ViT-Small 的特征维度是 384
            self.in_planes = self.transformer.embed_dim

        elif opt.backbone == "VAN-S":
            print('=> using Transformer_type: VAN-Small as backbone')
            self.transformer = van_small()
            checkpoint = torch.load(opt.pretrain_path)["state_dict"]
            self.transformer.load_state_dict(checkpoint)
            self.in_planes = 512  # 根据 VAN-S 定义

        else:
            raise ValueError(f"Unknown backbone type: {opt.backbone}")

        self.num_classes = num_classes

        # =====================================================
        # 【Step 2】初始化分类头 (Classifiers)
        # =====================================================
        
        # 1. 全局特征分类器 (处理 cls_token 或全局 pooled 特征)
        self.classifier1 = ClassBlock(self.in_planes, num_classes, 0.5, return_f=return_f)
        
        # 2. 局部特征分类器 (处理分块后的 tokens)
        for i in range(self.block):
            name = 'classifier_heat' + str(i + 1)
            setattr(self, name, ClassBlock(self.in_planes, num_classes, 0.5, return_f=self.return_f))

    def forward(self, x):
        """
        Args:
            x: [Batch, 3, 256, 256]
        """
        # -----------------------------------------------------
        # 1. 特征提取 (Backbone)
        # -----------------------------------------------------
        # 对 FSRA/VAN：self.transformer(x) -> [B, N_tokens, C]
        # 对 OfficialRoPEViTSmall：self.transformer(x) -> [B, C]
        features = self.transformer(x)

        # 对 RoPE-ViT 官方实现，将 [B, C] 统一扩展成 [B, 1, C]
        if features.ndim == 2:
            features = features.unsqueeze(1)

        # -----------------------------------------------------
        # 2. 处理全局特征 (Global Branch)
        # -----------------------------------------------------
        # 提取 cls_token / 全局特征并过分类器
        tranformer_feature = self.classifier1(features[:, 0])

        # 如果只需要全局特征 (block=1)，直接返回
        if self.block == 1:
            return tranformer_feature

        # -----------------------------------------------------
        # 3. 处理局部特征 (Part-based Branch)
        # -----------------------------------------------------
        part_features = features[:, 1:]  # 扔掉 cls_token

        # 使用热力图池化策略将 patch tokens 分成几个组
        heat_result = self.get_heartmap_pool(part_features)
        
        # 将每一块特征分别送入对应的分类器
        y = self.part_classifier(self.block, heat_result, cls_name='classifier_heat')

        # -----------------------------------------------------
        # 4. 返回结果
        # -----------------------------------------------------
        if self.training:
            # 训练阶段：需要计算所有分支的 Loss
            y = y + [tranformer_feature]
            if self.return_f:
                cls, feats = [], []
                for out in y:
                    cls.append(out[0])   # logits
                    feats.append(out[1]) # features
                return cls, feats
        else:
            # 测试阶段：拼接所有特征向量用于检索 (Retrieval)
            tranformer_feature = tranformer_feature.view(tranformer_feature.size(0), -1, 1)
            y = torch.cat([y, tranformer_feature], dim=2)

        return y

    def get_heartmap_pool(self, part_features, add_global=False, otherbranch=False):
        """
        基于热力图的特征重组策略 (Heatmap-based Pooling)
        """
        # 计算热力图 (沿着 channel 维度求均值) -> [Batch, N_patches]
        heatmap = torch.mean(part_features, dim=-1)
        size = part_features.size(1)  # Patch 数量
        
        # 根据热力值对 Patch 进行降序排序
        arg = torch.argsort(heatmap, dim=1, descending=True)
        
        # 把特征按照热力图的顺序重排
        x_sort = [part_features[i, arg[i], :] for i in range(part_features.size(0))]
        x_sort = torch.stack(x_sort, dim=0)

        # 将重排后的特征均分为 block 份
        split_each = size / self.block
        split_list = [int(split_each) for _ in range(self.block - 1)]
        split_list.append(size - sum(split_list))
        
        # 切分
        split_x = x_sort.split(split_list, dim=1)

        # 对每一份进行池化 (求均值) 得到局部特征
        split_list = [torch.mean(split, dim=1) for split in split_x]
        
        # 堆叠结果 -> [Batch, Dim, Block]
        part_featuers_ = torch.stack(split_list, dim=2)
        
        return part_featuers_

    def part_classifier(self, block, x, cls_name='classifier_lpn'):
        """局部特征分类器集合"""
        part = {}
        predict = {}
        for i in range(block):
            # 取出第 i 个 block 的特征
            part[i] = x[:, :, i].view(x.size(0), -1)
            # 找到对应的分类器 (classifier_heat1, classifier_heat2...)
            name = cls_name + str(i + 1)
            c = getattr(self, name)
            # 前向传播
            predict[i] = c(part[i])
            
        y = [predict[i] for i in range(block)]
            
        if not self.training:
            # 测试时返回堆叠后的特征 -> [Batch, Dim, Block]
            return torch.stack(y, dim=2)
        return y

    def load_param(self, trained_path):
        """加载整个模型的参数"""
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        """加载微调参数"""
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_transformer_model(opt, num_class, block=4, return_f=False):
    print('=========== Building Model: PCT (Polar-Contrastive Transformer) ===========')
    model = build_transformer(opt, num_class, block=block, return_f=return_f)
    return model