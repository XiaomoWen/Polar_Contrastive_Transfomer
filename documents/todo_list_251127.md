# UAV-Satellite Geo-Localization 冲刺计划 (ICIP Version)

## 第一阶段：工程重构与算力释放 (E-Day 0 ~ 3)

### 任务 1：代码升级为 DDP 分布式训练架构
- [ ] **核心修改**: 将单卡训练代码修改为支持 `torch.nn.parallel.DistributedDataParallel`。
- [ ] **SyncBN**: 加入 `torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)` 以保证多卡统计量同步。
- [ ] **LR 调整**: 遵循 Linear Scaling Rule: $LR_{new} = LR_{base} \times \frac{BatchSize_{new}}{BatchSize_{base}}$。
- [ ] **验收**: 在 8x A800 上跑通，Global Batch Size >= 256，GPU 利用率 > 90%。

### 任务 2：构建“双视图”数据加载器 (Dual-View DataLoader)
- [ ] **修改 `__getitem__`**: 每次读取返回 `(img_weak, img_strong)`。
    - `img_weak`: Resize + Norm (用于主 Loss)。
    - `img_strong`: Resize + **RandomErasing (p=0.5)** + **ColorJitter** + Norm (用于对比正则化)。
- [ ] **验收**: 可视化 Batch 图像，确认强增强视图包含遮挡/变色，弱增强视图保持干净。

### 任务 3：生成“鲁棒性测试集” (Robustness Benchmarks)
- [ ] **脚本编写**: 基于 University-1652 `query_drone` 生成副本（**不动 Gallery**）：
    - `query_occ_mild`: 中心随机遮挡 20% 面积。
    - `query_occ_hard`: 中心随机遮挡 50% 面积。
    - `query_crop`: 仅保留中心 60% 视场。
- [ ] **验收**: 生成完毕，文件结构正确。

---

## 第二阶段：核心实验与数据产出 (E-Day 4 ~ 10)

### 实验 A：主战场 SOTA 对比 (Main Results)
*目标：证明在标准数据集上处于第一梯队*

| 实验 ID | 模型配置 | Batch Size | 说明 | 预期 R@1 |
| :--- | :--- | :--- | :--- | :--- |
| **M-1** | Baseline (Swin-Tiny) | 256 (DDP) | 验证大 Batch 对 Transformer 的增益 | ~83% |
| **M-2** | + Polar Transform | 256 (DDP) | 当前状态，解决几何旋转 | ~86% |
| **M-3** | **+ Contrastive Reg (Ours)** | **256 (DDP)** | **完整方案，利用双视图强一致性** | **> 88%** |

### 实验 B：鲁棒性压力测试 (The Selling Point)
*目标：ICIP 核心卖点，证明抗干扰能力强*

| 实验 ID | 测试数据集 | 对比对象 | 关注指标 | 预期结论 |
| :--- | :--- | :--- | :--- | :--- |
| **R-1** | Occlusion (20% -> 50%) | FSRA / LPN | R@1 曲线斜率 | Ours 下降最缓 |
| **R-2** | Field of View (100% -> 60%) | FSRA / LPN | R@1 曲线斜率 | Ours 保持高位 |

### 实验 C：消融实验 (Ablation Study)
*目标：证明每个模块的必要性*

- [ ] **Baseline**: 无 Polar, 无 Reg。
- [ ] **+ Polar**: 证明几何对齐有效。
- [ ] **+ Reg**: 证明特征鲁棒性提升。
- [ ] **Full**: 叠加 Online Hard Mining。

---

## 第三阶段：论文图表制作 (E-Day 11+)

- [ ] **Table 1**: 主实验对比表 (加粗 M-3 结果)。
- [ ] **Figure 3**: 鲁棒性折线图 (X轴干扰强度，Y轴 R@1)。
- [ ] **Figure 4**: Grad-CAM 可视化 (展示模型关注到了建筑结构而非干扰物)。