from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Any

import copy
import torch
from tqdm import tqdm


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    计算 top-1 accuracy

    参数
    ----
    logits : [B, num_classes]
        模型未归一化输出
    targets : [B]
        类别标签

    返回
    ----
    float
        当前 batch 的平均准确率
    """
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def accuracy_topk(logits: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    """
    计算 top-k accuracy

    说明
    ----
    对于 ImageNet 风格分类任务，top-5 很常见。
    如果类别数不足 k，会自动截断到 num_classes。

    参数
    ----
    logits : [B, num_classes]
    targets : [B]
    k : int

    返回
    ----
    float
    """
    k = min(k, logits.shape[1])
    topk_idx = logits.topk(k, dim=1).indices          # [B, k]
    correct = topk_idx.eq(targets.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()


def default_prepare_batch(batch: Any, device: torch.device):
    """
    默认的 batch 预处理函数。

    设计意图
    --------
    你之前的 trainer / iter 系统里，真正的通用性很大程度来自：
    - 数据层按约定输出
    - trainer 只依赖统一格式

    所以这里单独留一个 prepare_batch，而不是把 batch 解包写死在 train loop 里。

    这样以后如果：
    - 你的旧 iter 返回 (X, y)
    - 且 X 是 list，里面只有一个模态
    - 或者以后想接多输入
    你只要换这个函数，而不用重写 trainer 本体。

    当前默认约定
    ------------
    batch = (images, targets)
    images : Tensor [B, C, H, W]
    targets: Tensor [B]

    返回
    ----
    images, targets : 都已经被搬到 device 上
    """
    images, targets = batch
    images = images.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    return images, targets


def train_one_epoch(
    model: torch.nn.Module,
    train_iter,
    optimizer: torch.optim.Optimizer,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    prepare_batch_fn: Callable[[Any, torch.device], tuple[torch.Tensor, torch.Tensor]] = default_prepare_batch,
    scheduler: Optional[Any] = None,
    use_iter_accum: bool = False,
    accum_steps: int = 1,
    log_top5: bool = True,
    epoch_index: Optional[int] = None,
) -> Dict[str, float]:
    """
    训练一个 epoch

    核心设计思路
    ------------
    这是从你 notebook 里抽出来的“训练主骨架”。
    我保留了你原来很重要的一点：支持 accumulation。
    但我把它分成两种模式：

    1) use_iter_accum = True
       表示沿用你旧 iter 的 accum 完成信号：
       train_iter.is_accum_complete()

       这种模式最贴近你原来的系统，不强行改你的习惯。

    2) use_iter_accum = False
       表示使用普通的固定 accum_steps 逻辑：
       每 accum_steps 个 batch 做一次 optimizer.step()

       这种模式更接近 PyTorch 常见写法。

    为什么这样做
    ------------
    因为你现在最大的价值不是“立刻换掉旧 iter”，
    而是先把 trainer 整理成稳定函数。
    所以这里允许两套 accumulation 策略并存，后面你可以再慢慢统一。

    参数
    ----
    model : 分类模型
    train_iter : 你的旧 iter 或任意可迭代 batch 对象
    optimizer : 优化器
    criterion : 损失函数，如 CrossEntropyLoss
    device : 设备
    prepare_batch_fn : batch 预处理函数
    scheduler : 若需要按 step 调度可在内部调用，当前默认不在 batch 内 step
    use_iter_accum : 是否使用旧 iter 的 accum 完成逻辑
    accum_steps : 非 iter 模式下的累积步数
    log_top5 : 是否记录 top-5
    epoch_index : 仅用于日志显示

    返回
    ----
    dict
        包含平均 train loss / acc1 / acc5
    """
    model.train()

    running_loss = 0.0
    running_acc1 = 0.0
    running_acc5 = 0.0
    num_batches = 0

    optimizer.zero_grad(set_to_none=True)

    desc = f"train" if epoch_index is None else f"train@{epoch_index}"
    pbar = tqdm(train_iter, desc=desc, leave=False)

    for step_idx, batch in enumerate(pbar, start=1):
        images, targets = prepare_batch_fn(batch, device)

        logits = model(images)
        loss = criterion(logits, targets)

        # -------------------------
        # 梯度累积逻辑
        # -------------------------
        # 两种方式二选一：
        #
        # A. 沿用旧 iter 的“一个 accum 块何时结束”的逻辑
        # B. 使用标准的 fixed accum_steps
        #
        # 这样整理的好处是：
        # trainer 主体稳定，accum 行为可配置
        # 而不是把它写死在 notebook 流程里。
        # -------------------------
        if use_iter_accum:
            # 旧 iter 模式下，通常你一个 accum block 由 iter 自己决定
            # 为了让梯度尺度稳定，这里仍建议对 loss 做归一化
            #
            # 注意：
            # 你旧系统里若有更特殊的 accum 长度逻辑，
            # 可以在这里继续细化。
            effective_accum = getattr(train_iter, "accum_size", None)
            batch_size = images.shape[0]
            if effective_accum is not None and batch_size > 0:
                # 例如 accum_size=128, batch_size=32，则每 4 个 batch 更新一次
                # 这里按近似比例做 loss 缩放
                # 若你后面想做更精确控制，可以再改成你 notebook 里的方式
                current_accum_steps = max(1, effective_accum // batch_size)
            else:
                current_accum_steps = 1

            scaled_loss = loss / current_accum_steps
            scaled_loss.backward()

            if train_iter.is_accum_complete():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        else:
            # 标准 accumulation 写法
            scaled_loss = loss / accum_steps
            scaled_loss.backward()

            if step_idx % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        # -------------------------
        # 统计指标
        # -------------------------
        batch_loss = loss.item()
        batch_acc1 = accuracy_top1(logits.detach(), targets)
        batch_acc5 = accuracy_topk(logits.detach(), targets, k=5) if log_top5 else 0.0

        running_loss += batch_loss
        running_acc1 += batch_acc1
        running_acc5 += batch_acc5
        num_batches += 1

        postfix = {
            "loss": f"{batch_loss:.4f}",
            "acc1": f"{batch_acc1:.4f}",
        }
        if log_top5:
            postfix["acc5"] = f"{batch_acc5:.4f}"
        pbar.set_postfix(postfix)

    # 如果最后一个 epoch 恰好没有触发 step（标准 accum 模式下可能发生）
    if not use_iter_accum and num_batches > 0 and (num_batches % accum_steps != 0):
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    if scheduler is not None and hasattr(scheduler, "step"):
        # 当前先按 epoch 级 step
        # 若你以后想按 batch 调度，再移到循环内部即可
        scheduler.step()

    if num_batches == 0:
        return {"loss": 0.0, "acc1": 0.0, "acc5": 0.0}

    return {
        "loss": running_loss / num_batches,
        "acc1": running_acc1 / num_batches,
        "acc5": running_acc5 / num_batches if log_top5 else 0.0,
    }


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    val_iter,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    prepare_batch_fn: Callable[[Any, torch.device], tuple[torch.Tensor, torch.Tensor]] = default_prepare_batch,
    log_top5: bool = True,
    epoch_index: Optional[int] = None,
) -> Dict[str, float]:
    """
    验证一个 epoch

    核心思路
    --------
    结构上尽量和 train_one_epoch 对称：
    - 这样阅读成本低
    - 以后改 metric / batch 接口时，train/val 可一起改

    当前分类任务里：
    - 不需要 segmentation 的 dice
    - 不需要复杂后处理
    - 只保留 loss / acc1 / acc5
    """
    model.eval()

    running_loss = 0.0
    running_acc1 = 0.0
    running_acc5 = 0.0
    num_batches = 0

    desc = f"val" if epoch_index is None else f"val@{epoch_index}"
    pbar = tqdm(val_iter, desc=desc, leave=False)

    for batch in pbar:
        images, targets = prepare_batch_fn(batch, device)

        logits = model(images)
        loss = criterion(logits, targets)

        batch_loss = loss.item()
        batch_acc1 = accuracy_top1(logits, targets)
        batch_acc5 = accuracy_topk(logits, targets, k=5) if log_top5 else 0.0

        running_loss += batch_loss
        running_acc1 += batch_acc1
        running_acc5 += batch_acc5
        num_batches += 1

        postfix = {
            "loss": f"{batch_loss:.4f}",
            "acc1": f"{batch_acc1:.4f}",
        }
        if log_top5:
            postfix["acc5"] = f"{batch_acc5:.4f}"
        pbar.set_postfix(postfix)

    if num_batches == 0:
        return {"loss": 0.0, "acc1": 0.0, "acc5": 0.0}

    return {
        "loss": running_loss / num_batches,
        "acc1": running_acc1 / num_batches,
        "acc5": running_acc5 / num_batches if log_top5 else 0.0,
    }


def fit(
    model: torch.nn.Module,
    train_iter,
    val_iter,
    optimizer: torch.optim.Optimizer,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    epochs: int,
    save_dir: str | Path,
    prepare_batch_fn: Callable[[Any, torch.device], tuple[torch.Tensor, torch.Tensor]] = default_prepare_batch,
    scheduler: Optional[Any] = None,
    use_iter_accum: bool = False,
    accum_steps: int = 1,
    log_top5: bool = True,
    early_stop_patience: Optional[int] = None,
    monitor: str = "val_acc1",
    maximize_monitor: bool = True,
) -> Dict[str, list]:
    """
    总训练入口

    设计思路
    --------
    这就是把你 notebook 里的 epoch 外层整理成一个真正的 fit 函数。

    它负责：
    - 组织 epoch 循环
    - 调 train / val
    - 保存 last / best 模型
    - early stopping
    - 返回 history 供你后续画图分析

    这里故意只保留最核心能力，不先追求“万能 trainer”，
    因为这次任务的目标是尽快跑通 ResNet 消融实验。

    monitor 设计
    ------------
    以前你 notebook 里是以 val_dice 作为最佳模型判据。
    现在分类任务自然切到：
    - val_acc1
    或者以后也可以改成：
    - val_loss

    这就是为什么我单独保留 monitor 字段：
    让 trainer 骨架不再绑死某一个任务指标。
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "train_acc1": [],
        "train_acc5": [],
        "val_loss": [],
        "val_acc1": [],
        "val_acc5": [],
    }

    best_state = None
    best_score = None
    wait = 0

    for epoch in range(1, epochs + 1):
        print(f"\n========== Epoch [{epoch}/{epochs}] ==========")

        train_stats = train_one_epoch(
            model=model,
            train_iter=train_iter,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            prepare_batch_fn=prepare_batch_fn,
            scheduler=scheduler,
            use_iter_accum=use_iter_accum,
            accum_steps=accum_steps,
            log_top5=log_top5,
            epoch_index=epoch,
        )

        val_stats = validate_one_epoch(
            model=model,
            val_iter=val_iter,
            criterion=criterion,
            device=device,
            prepare_batch_fn=prepare_batch_fn,
            log_top5=log_top5,
            epoch_index=epoch,
        )

        history["train_loss"].append(train_stats["loss"])
        history["train_acc1"].append(train_stats["acc1"])
        history["train_acc5"].append(train_stats["acc5"])
        history["val_loss"].append(val_stats["loss"])
        history["val_acc1"].append(val_stats["acc1"])
        history["val_acc5"].append(val_stats["acc5"])

        print(
            f"train loss={train_stats['loss']:.4f}, "
            f"train acc1={train_stats['acc1']:.4f}, "
            f"val loss={val_stats['loss']:.4f}, "
            f"val acc1={val_stats['acc1']:.4f}"
        )

        # 始终保存最近一次模型，方便中断恢复
        torch.save(model.state_dict(), save_dir / "last.pt")

        # -------------------------
        # 监控指标统一管理
        # -------------------------
        if monitor == "val_acc1":
            current_score = val_stats["acc1"]
        elif monitor == "val_acc5":
            current_score = val_stats["acc5"]
        elif monitor == "val_loss":
            current_score = val_stats["loss"]
        else:
            raise ValueError(f"Unknown monitor: {monitor}")

        is_better = False
        if best_score is None:
            is_better = True
        else:
            if maximize_monitor:
                is_better = current_score > best_score
            else:
                is_better = current_score < best_score

        if is_better:
            best_score = current_score
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, save_dir / "best.pt")
            wait = 0
            print(f"[best] {monitor} -> {best_score:.6f}")
        else:
            wait += 1
            print(f"[wait] early-stop counter = {wait}")

        if early_stop_patience is not None and wait >= early_stop_patience:
            print("Early stopping triggered.")
            break

    return history