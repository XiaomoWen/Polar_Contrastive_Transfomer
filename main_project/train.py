# -*- coding: utf-8 -*-

from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.cuda.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn
import time
import warnings

from optimizers.make_optimizer import make_optimizer
from models.model import make_model
from datasets.make_dataloader import make_dataset
from tool.utils_server import save_network, copyfiles2checkpoints
from losses.triplet_loss import Tripletloss
from losses.cal_loss import cal_kl_loss, cal_loss, cal_triplet_loss

warnings.filterwarnings("ignore")
version = torch.__version__


def get_parse():
    parser = argparse.ArgumentParser(description='Training PCT-Net')

    # 基础配置
    parser.add_argument('--gpu_ids', default='0', type=str)
    parser.add_argument('--name', default='test', type=str)
    parser.add_argument('--data_dir', default='University-Release/train', type=str)
    parser.add_argument('--train_all', action='store_true')
    parser.add_argument('--color_jitter', action='store_true')
    parser.add_argument('--num_worker', default=6, type=int)
    parser.add_argument('--batchsize', default=8, type=int)
    parser.add_argument('--pad', default=0, type=int)
    parser.add_argument('--h', default=224, type=int, help='image height (force to 224 for timm RoPE-ViT)')
    parser.add_argument('--w', default=224, type=int, help='image width (force to 224 for timm RoPE-ViT)')
    parser.add_argument('--views', default=2, type=int)
    parser.add_argument('--erasing_p', default=0, type=float)
    parser.add_argument('--DA', action='store_true')

    # 模型配置
    parser.add_argument('--backbone', default="VIT-S-RoPE-OFFICIAL", type=str,
                        help='VIT-S / VIT-S-RoPE-OFFICIAL / VAN-S')
    parser.add_argument('--pretrain_path', default="", type=str)
    parser.add_argument('--share', action='store_true', default=True)
    parser.add_argument('--block', default=3, type=int)

    # [新增/修改] 关键正则化参数接口 (为了鲁棒性)
    # 注意：默认值给得比较保守，实际训练通过shell脚本传入 aggressive 的值
    parser.add_argument('--drop_path_rate', default=0.1, type=float, help='Drop Path Rate for ViT')

    # 优化参数
    parser.add_argument('--num_epochs', default=120, type=int)
    parser.add_argument('--steps', default=[70, 110], type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--warm_epoch', default=0, type=int)
    parser.add_argument('--warmup_epochs', default=10, type=int)
    parser.add_argument('--moving_avg', default=1.0, type=float)
    
    # AMP
    # 默认设为 False
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--autocast', action='store_true', default=False)

    # 损失项
    parser.add_argument('--kl_loss', action='store_true', default=False)
    parser.add_argument('--triplet_loss', default=0.3, type=float)
    parser.add_argument('--triplet_weight', default=4.5, type=float)
    parser.add_argument('--sample_num', default=1, type=float)

    opt = parser.parse_args()
    return opt


def train_model(model, opt, optimizer, scheduler, dataloaders, dataset_sizes):
    use_gpu = opt.use_gpu
    num_epochs = opt.num_epochs
    since = time.time()

    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()
    loss_kl = nn.KLDivLoss(reduction='batchmean')
    triplet_loss_fn = Tripletloss(margin=opt.triplet_loss)

    print(f"Training Strategy Confirmed:")
    print(f"  >>> DropPath Rate: {opt.drop_path_rate} (High for Robustness)")
    print(f"  >>> Weight Decay : {opt.weight_decay} (High for Regularization)")
    print(f"  >>> Autocast     : {opt.autocast}")

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # ----------------- 训练模式 -----------------
        model.train(True)

        # 统计量
        running_cls_loss = 0.0
        running_triplet = 0.0
        running_kl_loss = 0.0
        running_loss = 0.0
        total_samples = 0

        running_corrects = 0  # 用于计算 train_acc（top‑1）

        # 使用 enumerate 以便获取 batch 索引
        for batch_idx, (data, data2, data3) in enumerate(dataloaders):
            loss = 0.0
            inputs, labels = data
            inputs2, labels2 = data2
            inputs3, labels3 = data3

            now_batch_size = inputs.shape[0]
            if now_batch_size < opt.batchsize:
                continue
            total_samples += now_batch_size

            if use_gpu:
                inputs = inputs.cuda()
                inputs2 = inputs2.cuda()
                inputs3 = inputs3.cuda()
                labels = labels.cuda()
                labels2 = labels2.cuda()
                labels3 = labels3.cuda()

            optimizer.zero_grad()

            # ---------- 前向 ----------
            if opt.views == 2:
                with autocast(enabled=opt.autocast):
                    outputs, outputs2 = model(inputs, inputs3)
            elif opt.views == 3:
                with autocast(enabled=opt.autocast):
                    outputs, outputs2, outputs3 = model(inputs, inputs2, inputs3)
            else:
                raise ValueError(f"views={opt.views} not supported")

            # Triplet loss（outputs: (logits, feat)）
            f_triplet_loss = torch.tensor(0., device=inputs.device)
            if opt.triplet_loss > 0:
                features = outputs[1]
                features2 = outputs2[1]
                split_num = opt.batchsize // opt.sample_num
                f_triplet_loss = cal_triplet_loss(
                    features, features2, labels, triplet_loss_fn, split_num
                )
                # 此时 outputs 变成了 tuple(logits, feat)，我们需要把 logits 取出来覆盖 outputs
                outputs, outputs2 = outputs[0], outputs2[0]
            else:
                # 如果没有 triplet loss，模型输出依然是 tuple，也需要取出 logits
                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]
                if isinstance(outputs2, (list, tuple)):
                    outputs2 = outputs2[0]
                if opt.views == 3 and isinstance(outputs3, (list, tuple)):
                    outputs3 = outputs3[0]

            # ---------- 分类损失 + KL ----------
            # 注意：经过上面的处理，outputs 已经是 logits Tensor 了
            if opt.views == 2:
                cls_loss = cal_loss(outputs, labels, criterion) + \
                           cal_loss(outputs2, labels3, criterion)
                if opt.kl_loss:
                    kl_loss = cal_kl_loss(outputs, outputs2, loss_kl)
                else:
                    kl_loss = torch.tensor(0., device=inputs.device)
            else:   # views == 3
                cls_loss = cal_loss(outputs, labels, criterion) + \
                           cal_loss(outputs2, labels2, criterion) + \
                           cal_loss(outputs3, labels3, criterion)
                kl_loss = torch.tensor(0., device=inputs.device)

            loss = kl_loss + cls_loss + opt.triplet_weight * f_triplet_loss

            # ---------- 反向传播 ----------
            with autocast(enabled=opt.autocast):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # ---------- 累计统计 ----------
            running_loss += loss.item() * now_batch_size
            running_cls_loss += cls_loss.item() * now_batch_size
            running_triplet += f_triplet_loss.item() * now_batch_size
            running_kl_loss += kl_loss.item() * now_batch_size

            # ---------- 计算准确率 (修复了 TypeError) ----------
            with torch.no_grad():
                # 再次做安全检查，确保 score 是 Tensor
                if isinstance(outputs, (list, tuple)):
                    score = outputs[0]
                else:
                    score = outputs
                
                _, preds = torch.max(score, 1)
                batch_correct = torch.sum(preds == labels).item()
                running_corrects += batch_correct

            # ---------- [新增] 实时打印 (每20个batch打印一次) ----------
            if (batch_idx + 1) % 20 == 0:
                print("  Step[{}/{}] Loss: {:.4f} (Cls: {:.3f} Tri: {:.3f}) Acc: {:.2f}%".format(
                    batch_idx + 1, len(dataloaders),
                    loss.item(),
                    cls_loss.item(),
                    f_triplet_loss.item(),
                    (batch_correct / now_batch_size) * 100
                ))

        # ----------------- 每个 Epoch 的汇总打印 -----------------
        if total_samples == 0:
            print("  [Warning] No batch reached batchsize={}, skip epoch stats.".format(opt.batchsize))
            scheduler.step()
            continue

        epoch_loss = running_loss / total_samples
        epoch_cls_loss = running_cls_loss / total_samples
        epoch_triplet_loss = running_triplet / total_samples
        epoch_kl = running_kl_loss / total_samples
        epoch_acc = running_corrects / float(total_samples)

        epoch_time = time.time() - epoch_start

        print("Epoch {} summary:".format(epoch))
        print("  total_loss   : {:.4f}".format(epoch_loss))
        print("  cls_loss     : {:.4f}".format(epoch_cls_loss))
        print("  triplet_loss : {:.4f}".format(epoch_triplet_loss))
        print("  kl_loss      : {:.4f}".format(epoch_kl))
        print("  train_acc    : {:.4f}".format(epoch_acc))
        print("  epoch_time   : {:.0f}m {:.0f}s".format(epoch_time // 60, epoch_time % 60))
        print('-' * 30)

        # 学习率调度 & 保存
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            save_network(model, opt.name, epoch)

        # 累计训练时长提示
        time_elapsed = time.time() - since
        print("Training elapsed: {:.0f}m {:.0f}s\n".format(
            time_elapsed // 60, time_elapsed % 60
        ))


if __name__ == '__main__':
    opt = get_parse()

    gpu_ids = [int(i) for i in opt.gpu_ids.split(',') if int(i) >= 0]
    use_gpu = torch.cuda.is_available()
    opt.use_gpu = use_gpu
    if use_gpu and gpu_ids:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True

    dataloaders, class_names, dataset_sizes = make_dataset(opt)
    opt.nclasses = len(class_names)

    model = make_model(opt).cuda()
    optimizer_ft, exp_lr_scheduler = make_optimizer(model, opt)
    copyfiles2checkpoints(opt)

    print('---------- Start Training -------------')
    train_model(model, opt, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes)
