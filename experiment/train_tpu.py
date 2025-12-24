"""
TPU (PyTorch XLA) 适配版训练脚本。

使用方式：
export XLA_USE_BF16=1  # 开启 BFloat16 加速（推荐）
python experiment/train_tpu.py --config configs/train_baseline.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import time
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# TPU Imports
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.utils.utils as xu
except ImportError:
    print("警告: 未检测到 torch_xla，请确保已安装 TPU 环境。")

try:
    from .datasets import GridMaskDataset, SampleItem, discover_samples
    from .models_factory import ModelSpec, build_model, iter_model_specs
except ImportError:
    import sys
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
    from experiment.datasets import GridMaskDataset, SampleItem, discover_samples
    from experiment.models_factory import ModelSpec, build_model, iter_model_specs


@dataclass
class ExperimentConfig:
    name: str
    seed: int
    output_dir: Path


@dataclass
class DataConfig:
    train_dirs: tuple[Path, ...]
    val_dirs: tuple[Path, ...]
    val_split: float | None
    batch_size: int
    num_workers: int
    augment: bool
    in_channels: int
    num_classes: int


@dataclass
class OptimConfig:
    lr: float
    weight_decay: float
    type: str


@dataclass
class TrainingConfig:
    epochs: int
    log_interval: int
    amp: bool  # TPU 上通常通过 XLA_USE_BF16=1 环境变量控制，代码中不再使用 GradScaler
    grad_clip: float | None


@dataclass
class LossConfig:
    primary: str
    dice_weight: float


@dataclass
class ModelConfig:
    spec: ModelSpec
    encoder_weights: str | None
    in_channels: int
    classes: int
    model_kwargs: Dict[str, Any]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练网格分割模型 (TPU)")
    parser.add_argument("--config", type=Path, required=True, help="YAML 配置文件路径")
    return parser.parse_args(argv)


def load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("配置文件格式错误")
    return data


def to_experiment_config(cfg: dict[str, Any]) -> ExperimentConfig:
    experiment = cfg.get("experiment", {})
    name = experiment.get("name", "grid_experiment")
    seed = int(experiment.get("seed", 2025))
    output_dir = Path(experiment.get("output_dir", "./runs")) / name
    return ExperimentConfig(name=name, seed=seed, output_dir=output_dir)


def to_data_config(cfg: dict[str, Any], exp_cfg: ExperimentConfig, *, config_path: Path) -> DataConfig:
    data = cfg.get("data", {})
    base_dir = config_path.parent.resolve()

    def _resolve_path(value: str) -> Path:
        p = Path(value)
        if p.is_absolute():
            return p
        return (base_dir / p).resolve()

    def _resolve_path_list(value: Any, default: list[str]) -> tuple[Path, ...]:
        if value is None:
            candidates = default
        elif isinstance(value, (str, Path)):
            candidates = [str(value)]
        elif isinstance(value, Sequence):
            candidates = [str(v) for v in value]
        else:
            raise TypeError("路径配置必须是字符串或字符串列表")
        return tuple(_resolve_path(v) for v in candidates)

    train_dirs = _resolve_path_list(
        data.get("train_dirs", data.get("train_dir")),
        default=["./synthetic_grid_3000"],
    )
    
    val_dirs_raw = data.get("val_dirs", data.get("val_dir"))
    if val_dirs_raw is not None:
        val_dirs = _resolve_path_list(val_dirs_raw, default=[])
    else:
        val_dirs = tuple()
    
    val_split = data.get("val_split", 0.1 if not val_dirs else None)
    batch_size = int(data.get("batch_size", 8))
    num_workers = int(data.get("num_workers", 4))
    augment = bool(data.get("augment", True))
    in_channels = int(data.get("in_channels", 3))
    num_classes = int(data.get("classes", 1))
    
    return DataConfig(
        train_dirs=train_dirs,
        val_dirs=val_dirs,
        val_split=val_split,
        batch_size=batch_size,
        num_workers=num_workers,
        augment=augment,
        in_channels=in_channels,
        num_classes=num_classes,
    )


def to_optim_config(cfg: dict[str, Any]) -> OptimConfig:
    optim = cfg.get("optim", {})
    return OptimConfig(
        lr=float(optim.get("lr", 1e-3)),
        weight_decay=float(optim.get("weight_decay", 1e-4)),
        type=str(optim.get("type", "adamw")).lower(),
    )


def to_training_config(cfg: dict[str, Any]) -> TrainingConfig:
    training = cfg.get("training", {})
    grad_clip = training.get("grad_clip")
    return TrainingConfig(
        epochs=int(training.get("epochs", 20)),
        log_interval=int(training.get("log_interval", 20)),
        amp=bool(training.get("amp", True)),
        grad_clip=float(grad_clip) if grad_clip is not None else None,
    )


def to_loss_config(cfg: dict[str, Any]) -> LossConfig:
    loss = cfg.get("loss", {})
    primary = str(loss.get("type", "bce")).lower()
    dice_weight = float(loss.get("dice_weight", 0.5 if primary == "bce" else 0.0))
    return LossConfig(primary=primary, dice_weight=dice_weight)


def resolve_model_configs(cfg: dict[str, Any], *, default_in_channels: int, default_classes: int) -> List[ModelConfig]:
    spec_map = {spec.key: spec for spec in iter_model_specs()}
    models_cfg = cfg.get("models")
    if not models_cfg:
        raise ValueError("配置需包含 models 列表")

    resolved: list[ModelConfig] = []
    for entry in models_cfg:
        key = entry.get("key")
        spec = spec_map.get(key)
        if spec is None:
            continue # Skip unknown models or handle error
        
        resolved.append(ModelConfig(
            spec=spec,
            encoder_weights=entry.get("encoder_weights", "imagenet"),
            in_channels=int(entry.get("in_channels", default_in_channels)),
            classes=int(entry.get("classes", default_classes)),
            model_kwargs=entry.get("model_kwargs", {}),
        ))
    return resolved


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets, dims)
        denominator = torch.sum(probs + targets, dims)
        dice = (2 * intersection + self.eps) / (denominator + self.eps)
        return 1 - dice.mean()


def dice_metric(logits: torch.Tensor, targets: torch.Tensor) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    intersection = torch.sum(preds * targets)
    denominator = torch.sum(preds + targets)
    if denominator == 0:
        return 1.0
    return float((2 * intersection + 1e-6) / (denominator + 1e-6))


def create_optimizer(params: Iterable[torch.Tensor], cfg: OptimConfig) -> torch.optim.Optimizer:
    if cfg.type == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.type == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.type == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay, nesterov=True)
    raise ValueError(f"不支持的优化器类型: {cfg.type}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    loss_config: LossConfig,
    device: torch.device,
    num_classes: int,
    grad_clip: float | None,
    log_interval: int,
) -> dict[str, float]:
    model.train()
    dice_loss_fn = DiceLoss()
    primary_loss_fn = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()

    total_loss = 0.0
    total_batches = 0
    tic = time.time()
    
    # TPU Parallel Loader
    para_loader = pl.ParallelLoader(loader, [device])
    
    for step, batch in enumerate(para_loader.per_device_loader(device), start=1):
        images = batch["image"]
        masks = batch["mask"]

        optimizer.zero_grad()
        outputs = model(images)
        
        if num_classes == 1:
            primary_loss = primary_loss_fn(outputs, masks)
            if loss_config.dice_weight > 0:
                dice_loss = dice_loss_fn(outputs, masks)
                loss = (1 - loss_config.dice_weight) * primary_loss + loss_config.dice_weight * dice_loss
            else:
                loss = primary_loss
        else:
            masks_int = masks.long()
            loss = primary_loss_fn(outputs, masks_int)

        loss.backward()
        
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
        xm.optimizer_step(optimizer)

        # loss.item() 在 TPU 上会导致同步，影响性能，仅在 logging 时获取
        if step % log_interval == 0:
            loss_val = loss.item()
            total_loss += loss_val
            total_batches += 1
            elapsed = time.time() - tic
            xm.master_print(f"  step {step:04d} | loss {loss_val:.4f} | {elapsed:.1f}s")
            tic = time.time()
        else:
            # 简单累加用于后续平均（虽不完全精确因为没有每步同步，但在此场景下可接受）
            pass

    # 注意：这里的 total_loss 仅是采样点的平均，为了性能不做全同步
    return {"train_loss": total_loss / max(total_batches, 1)}


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    num_classes: int,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    batches = 0
    primary_loss_fn = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
    
    para_loader = pl.ParallelLoader(loader, [device])

    with torch.no_grad():
        for batch in para_loader.per_device_loader(device):
            images = batch["image"]
            masks = batch["mask"]
            outputs = model(images)
            
            if num_classes == 1:
                primary_loss = primary_loss_fn(outputs, masks)
                total_loss += primary_loss.item()
                total_dice += dice_metric(outputs, masks)
            else:
                masks_int = masks.long()
                primary_loss = primary_loss_fn(outputs, masks_int)
                total_loss += primary_loss.item()
                
            batches += 1

    # Reduce metrics across cores
    metrics_tensor = torch.tensor([total_loss, total_dice, batches], device=device)
    reduced = xm.all_reduce('sum', metrics_tensor)
    
    final_loss = reduced[0].item() / max(reduced[2].item(), 1)
    final_dice = reduced[1].item() / max(reduced[2].item(), 1)

    metrics = {"val_loss": final_loss}
    if num_classes == 1:
        metrics["val_dice"] = final_dice
    return metrics


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    model_config: ModelConfig,
) -> None:
    # 仅在 Master 节点保存
    if not xm.is_master_ordinal():
        return
        
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "model_config": asdict(model_config),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # 使用 xm.save 确保正确保存（它在 CPU 上保存）
    xm.save(state, str(path))


def prepare_dataloaders(
    data_cfg: DataConfig,
    *,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    # 收集样本 (仅在 global master 做，或者轻量级重复做)
    # 为了简单，所有 rank 都跑一遍文件扫描（几千个文件很快）
    
    def _collect_samples(directories: Sequence[Path]) -> list[SampleItem]:
        items: list[SampleItem] = []
        for directory in directories:
            try:
                items.extend(discover_samples(directory))
            except FileNotFoundError:
                pass # 允许部分目录不存在
        return items

    train_items = _collect_samples(data_cfg.train_dirs)
    if not train_items:
        # 如果还没下载数据，可能为空，这里抛出更友好的错误
        if xm.is_master_ordinal():
             print(f"错误: 在 {data_cfg.train_dirs} 未找到训练数据。请检查是否已运行 gsutil cp 下载数据。")
        exit(1)

    if data_cfg.val_dirs:
        val_items = _collect_samples(data_cfg.val_dirs)
    else:
        indices = np.arange(len(train_items))
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        val_count = max(1, int(len(indices) * float(data_cfg.val_split or 0.1)))
        val_idx = indices[:val_count]
        train_idx = indices[val_count:]
        val_items = [train_items[i] for i in val_idx]
        train_items = [train_items[i] for i in train_idx]

    train_ds = GridMaskDataset(
        train_items,
        augment=data_cfg.augment,
        num_classes=data_cfg.num_classes,
    )
    val_ds = GridMaskDataset(
        val_items,
        augment=False,
        num_classes=data_cfg.num_classes,
    )

    # DistributedSampler 对 TPU 多核训练至关重要
    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True,
    )
    val_sampler = DistributedSampler(
        val_ds,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg.batch_size,
        sampler=train_sampler,
        num_workers=data_cfg.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg.batch_size,
        sampler=val_sampler,
        num_workers=data_cfg.num_workers,
        drop_last=False,
    )
    
    return train_loader, val_loader


def run_training_process(rank: int, args: argparse.Namespace) -> None:
    # 设置随机种子
    torch.manual_seed(2025)
    
    # 获取 TPU 设备
    device = xm.xla_device()
    
    raw_cfg = load_yaml_config(args.config)
    exp_cfg = to_experiment_config(raw_cfg)
    data_cfg = to_data_config(raw_cfg, exp_cfg, config_path=args.config.resolve())
    optim_cfg = to_optim_config(raw_cfg)
    training_cfg = to_training_config(raw_cfg)
    loss_cfg = to_loss_config(raw_cfg)
    model_cfgs = resolve_model_configs(
        raw_cfg,
        default_in_channels=data_cfg.in_channels,
        default_classes=data_cfg.num_classes,
    )

    if xm.is_master_ordinal():
        exp_cfg.output_dir.mkdir(parents=True, exist_ok=True)
        with (exp_cfg.output_dir / "config_tpu.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(raw_cfg, f, allow_unicode=True)

    # 准备数据Loader
    train_loader, val_loader = prepare_dataloaders(data_cfg, seed=exp_cfg.seed)

    for model_cfg in model_cfgs:
        xm.master_print(f"开始训练: {model_cfg.spec.key}")
        
        # 实例化模型
        model = build_model(
            model_cfg.spec,
            in_channels=model_cfg.in_channels,
            classes=model_cfg.classes,
            encoder_weights=model_cfg.encoder_weights,
            **model_cfg.model_kwargs,
        ).to(device)

        optimizer = create_optimizer(model.parameters(), optim_cfg)
        
        model_dir = exp_cfg.output_dir / model_cfg.spec.key
        if xm.is_master_ordinal():
            model_dir.mkdir(parents=True, exist_ok=True)

        best_metric = -float("inf")
        history = []

        for epoch in range(1, training_cfg.epochs + 1):
            xm.master_print(f"Epoch {epoch}/{training_cfg.epochs}")
            
            # 训练
            train_loader.sampler.set_epoch(epoch)
            train_stats = train_one_epoch(
                model,
                train_loader,
                optimizer=optimizer,
                loss_config=loss_cfg,
                device=device,
                num_classes=model_cfg.classes,
                grad_clip=training_cfg.grad_clip,
                log_interval=training_cfg.log_interval,
            )
            
            # 验证
            val_stats = evaluate(
                model,
                val_loader,
                device=device,
                num_classes=model_cfg.classes,
            )
            
            epoch_stats = {"epoch": epoch, **train_stats, **val_stats}
            history.append(epoch_stats)
            
            msg = " | ".join(f"{k}: {v:.4f}" for k, v in epoch_stats.items() if isinstance(v, float))
            xm.master_print(f"  {msg}")

            score = val_stats.get("val_dice", -val_stats["val_loss"])
            if score > best_metric:
                best_metric = score
                save_checkpoint(
                    model_dir / "best.ckpt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=val_stats,
                    model_config=model_cfg,
                )
            
            # 定期保存
            if epoch % 5 == 0:
                save_checkpoint(
                    model_dir / f"epoch_{epoch:03d}.ckpt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=val_stats,
                    model_config=model_cfg,
                )

        if xm.is_master_ordinal():
             with (model_dir / "history.json").open("w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    # 启动多进程训练 (默认利用所有 TPU 核心)
    xmp.spawn(run_training_process, args=(args,), start_method='fork')


if __name__ == "__main__":
    main()
