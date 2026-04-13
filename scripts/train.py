from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
from torchvision import transforms

# 你自己的模块
from engine.trainer_cls import fit
from utils.config import load_config

# 如果你决定继续沿用旧 iter，就从旧文件里导入
# 假设你把它整理成 dataiter.py
from dataio.dataiter import DataIter


def build_basic_transforms(cfg):
    """
    当前任务的最基础 ImageNet 风格预处理

    为什么这里先不用你旧的复杂 preprocess？
    --------------------------------
    因为这次任务的目标是先做一个标准、干净、可控的 baseline。
    先用最经典的 torchvision transforms，有两个好处：

    1. 变量少，便于控制实验
    2. 后面如果你想把旧 preprocess 接进来，很容易做对照实验

    返回
    ----
    train_transform, val_transform
    """
    image_size = cfg.data.image_size
    resize_size = int(image_size / 0.875)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.data.mean, std=cfg.data.std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.data.mean, std=cfg.data.std),
    ])

    return train_transform, val_transform


def build_datasets(cfg):
    """
    使用 ImageFolder 构建分类数据集

    目录结构应为：
    data_root/
        train/
            class_a/
                xxx.jpg
            class_b/
                yyy.jpg
        val/
            class_a/
            class_b/

    说明
    ----
    这一步故意走最稳的 torchvision 标准接口。
    因为你当前的目标不是折腾数据格式，而是尽快进入 ResNet 实验。
    """
    train_transform, val_transform = build_basic_transforms(cfg)

    train_set = datasets.ImageFolder(
        root=cfg.data.train_dir,
        transform=train_transform
    )
    val_set = datasets.ImageFolder(
        root=cfg.data.val_dir,
        transform=val_transform
    )

    return train_set, val_set


def dataset_to_inmemory_arrays(dataset):
    """
    把 torchvision Dataset 预先全部转成内存数组，供你旧 iter 的 inmemory 模式使用。

    为什么这样做？
    -------------
    因为你当前想继续沿用旧 iter，而旧 iter 的 InMemoryDataManager
    更自然地吃的是 ndarray / tensor。

    这里先走最简单版本：
    - 预先把整个 train / val 数据集变成内存 tensor
    - 再交给 DataIter(inmemory)

    这样做的优点：
    -------------
    1. 最少改你的旧系统
    2. trainer 和 iter 能快速接起来
    3. 适合先做 baseline

    缺点：
    ----
    1. 数据集太大时会占内存
    2. 不适合完整 ImageNet-1K

    但对 ImageNet-lite / ImageNet-100 的第一版实验来说，通常是可接受的。

    返回
    ----
    images_tensor : torch.Tensor [N, C, H, W]
    labels_tensor : torch.Tensor [N]
    """
    images = []
    labels = []

    for img, label in dataset:
        # img 已经是 transform 后的 tensor [C, H, W]
        images.append(img)
        labels.append(label)

    images_tensor = torch.stack(images, dim=0)              # [N, C, H, W]
    labels_tensor = torch.tensor(labels, dtype=torch.long)  # [N]

    return images_tensor, labels_tensor


def build_iters_with_old_dataiter(cfg, device):
    """
    用你旧的 DataIter 构建 train_iter / val_iter

    当前策略
    --------
    - 单模态输入
    - manager='inmemory'
    - loop=False
    - output_backend='torch'
    - 不开复杂 lazy/backend 切换
    - 只保留最基本 batch / shuffle / accum 能力

    这就是我们之前讨论的：
    “旧 iter 可以继续用，但先把复杂能力关掉，作为一个稳定熟悉的 batch provider。”
    """
    train_set, val_set = build_datasets(cfg)

    train_x, train_y = dataset_to_inmemory_arrays(train_set)
    val_x, val_y = dataset_to_inmemory_arrays(val_set)

    # 注意：你的旧 DataIter 设计里 X 是多模态 list
    # 所以这里即使只有一个模态，也要写成 [train_x]
    train_iter = DataIter(
        X=[train_x],
        y=train_y,
        num_samples=len(train_y),
        batch_size=cfg.train.batch_size,
        shuffle=True,
        accum_size=cfg.train.accum_size,
        input_backend="numpy",      # inmemory 模式下这两个其实不会真正用到
        output_backend="torch",
        manager="inmemory",
        device=device,
        loop=False,
        dtype="float32",
    )

    val_iter = DataIter(
        X=[val_x],
        y=val_y,
        num_samples=len(val_y),
        batch_size=cfg.train.batch_size,
        shuffle=False,
        accum_size=cfg.train.batch_size,  # 验证时通常不需要累积
        input_backend="numpy",
        output_backend="torch",
        manager="inmemory",
        device=device,
        loop=False,
        dtype="float32",
    )

    num_classes = len(train_set.classes)
    return train_iter, val_iter, num_classes


def build_model(cfg, num_classes: int):
    """
    先做最稳的第一版：
    直接用 torchvision 的 ResNet-18 / ResNet-34

    为什么先这么做？
    ---------------
    因为当前第一目标是：
    先把 baseline 跑通，再做 plain net 对照。

    现在如果连 baseline 都还没跑通，就先别急着把所有变量一起引入。
    """
    model_type = cfg.model.type.lower()
    depth = cfg.model.depth

    if model_type == "resnet":
        if depth == 18:
            model = torchvision.models.resnet18(weights=None)
        elif depth == 34:
            model = torchvision.models.resnet34(weights=None)
        else:
            raise ValueError(f"Unsupported resnet depth: {depth}")

        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    elif model_type == "plainnet":
        # 这里先占位。等 baseline 跑通后再补 plainnet。
        raise NotImplementedError("PlainNet will be added after ResNet baseline is confirmed.")

    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")


def build_optimizer(cfg, model):
    """
    当前先用最经典的 SGD 配置，贴近 ResNet 论文风格。
    """
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.train.lr,
        momentum=cfg.train.momentum,
        weight_decay=cfg.train.weight_decay,
    )
    return optimizer


def prepare_batch_from_old_iter(batch, device):
    """
    这是接你旧 iter 的唯一薄接口。

    为什么这里仍然保留这个函数？
    ---------------------------
    不是因为必须造“适配层”，而是因为：
    你的旧 iter 当前返回形式是 (X, y)，其中 X 是 list。
    即使只有一个模态，依然会是：
        X = [images]

    所以 trainer 最自然的做法就是：
    把这种“单模态 list”解包成分类任务真正想要的 images tensor。

    这不是额外复杂度，而是把“旧系统格式”和“当前任务格式”的连接点
    明确固定在一个地方，避免以后 trainer 主体里到处写 X[0]。

    输入
    ----
    batch = (X, y)
        X: list, 长度为1，X[0]是 [B, C, H, W]
        y: [B]

    返回
    ----
    images, targets
    """
    X, y = batch
    images = X[0].to(device, non_blocking=True)
    targets = y.to(device, non_blocking=True).long()
    return images, targets


def main():
    parser = argparse.ArgumentParser(description="Train ResNet on ImageNet-lite with old iter + new trainer.")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # -------------------------
    # 设备选择
    # -------------------------
    if torch.cuda.is_available() and "cuda" in cfg.train.device:
        device = torch.device(cfg.train.device)
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # -------------------------
    # 构建数据
    # -------------------------
    print("Building iterators...")
    train_iter, val_iter, num_classes = build_iters_with_old_dataiter(cfg, device)

    # -------------------------
    # 构建模型
    # -------------------------
    print("Building model...")
    model = build_model(cfg, num_classes=num_classes).to(device)

    # -------------------------
    # 损失函数
    # -------------------------
    # 分类任务第一版最推荐的就是 CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()

    # -------------------------
    # 优化器
    # -------------------------
    optimizer = build_optimizer(cfg, model)

    # -------------------------
    # 输出目录
    # -------------------------
    save_dir = Path(cfg.train.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # 开始训练
    # -------------------------
    print("Start training...")
    history = fit(
        model=model,
        train_iter=train_iter,
        val_iter=val_iter,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=cfg.train.epochs,
        save_dir=save_dir,
        prepare_batch_fn=prepare_batch_from_old_iter,
        scheduler=None,
        use_iter_accum=cfg.train.use_iter_accum,
        accum_steps=cfg.train.accum_steps,
        log_top5=True,
        early_stop_patience=cfg.train.early_stop_patience,
        monitor=cfg.train.monitor,
        maximize_monitor=cfg.train.maximize_monitor,
    )

    print("Training finished.")
    print("History keys:", history.keys())


if __name__ == "__main__":
    main()