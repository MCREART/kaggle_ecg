"""
基于 YAML 配置的多模型训练脚本。

使用方式：
python experiment/train.py --config configs/train_baseline.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn
from torch.cuda import amp
from torch.utils.data import DataLoader

try:
    from .datasets import GridMaskDataset, SampleItem, discover_samples
    from .models_factory import ModelSpec, build_model, iter_model_specs
except ImportError:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
    from experiment.datasets import GridMaskDataset, SampleItem, discover_samples  # type: ignore
    from experiment.models_factory import (  # type: ignore
        ModelSpec,
        build_model,
        iter_model_specs,
    )


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
    amp: bool
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
    parser = argparse.ArgumentParser(description="训练网格分割模型")
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


def to_data_config(
    cfg: dict[str, Any],
    exp_cfg: ExperimentConfig,
    *,
    config_path: Path,
) -> DataConfig:
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
    if not train_dirs:
        raise ValueError("data.train_dirs 不能为空")

    val_dirs_raw = data.get("val_dirs", data.get("val_dir"))
    if val_dirs_raw is not None:
        val_dirs = _resolve_path_list(val_dirs_raw, default=[])
    else:
        val_dirs = tuple()
    val_split = data.get("val_split", 0.1 if not val_dirs else None)
    if val_dirs and val_split is not None:
        raise ValueError("同时指定 val_dirs 与 val_split 不受支持")
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
    grad_clip_value = float(grad_clip) if grad_clip is not None else None
    return TrainingConfig(
        epochs=int(training.get("epochs", 20)),
        log_interval=int(training.get("log_interval", 20)),
        amp=bool(training.get("amp", True)),
        grad_clip=grad_clip_value,
    )


def to_loss_config(cfg: dict[str, Any]) -> LossConfig:
    loss = cfg.get("loss", {})
    primary = str(loss.get("type", "bce")).lower()
    dice_weight = float(loss.get("dice_weight", 0.5 if primary == "bce" else 0.0))
    return LossConfig(primary=primary, dice_weight=dice_weight)


def resolve_model_configs(
    cfg: dict[str, Any],
    *,
    default_in_channels: int,
    default_classes: int,
) -> List[ModelConfig]:
    spec_map = {spec.key: spec for spec in iter_model_specs()}
    models_cfg = cfg.get("models")
    if not models_cfg:
        raise ValueError("配置需包含 models 列表")

    resolved: list[ModelConfig] = []
    for entry in models_cfg:
        if not isinstance(entry, dict):
            raise ValueError("models 条目应为字典")
        key = entry.get("key")
        if key is None:
            raise ValueError("models 条目缺少 key")
        spec = spec_map.get(key)
        if spec is None:
            raise KeyError(f"未知模型 key: {key}")
        encoder_weights = entry.get("encoder_weights", "imagenet")
        in_channels = int(entry.get("in_channels", default_in_channels))
        classes = int(entry.get("classes", default_classes))
        model_kwargs = entry.get("model_kwargs", {})
        resolved.append(
            ModelConfig(
                spec=spec,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                model_kwargs=model_kwargs,
            )
        )
    return resolved


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def prepare_dataloaders(
    data_cfg: DataConfig,
    *,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    def _collect_samples(directories: Sequence[Path]) -> list[SampleItem]:
        items: list[SampleItem] = []
        for directory in directories:
            items.extend(discover_samples(directory))
        if not items:
            raise RuntimeError(f"{', '.join(str(p) for p in directories)} 下未发现样本")
        return items

    train_items = _collect_samples(data_cfg.train_dirs)

    if data_cfg.val_dirs:
        val_items = _collect_samples(data_cfg.val_dirs)
    else:
        indices = np.arange(len(train_items))
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        if data_cfg.val_split is None:
            raise ValueError("val_split 未指定")
        val_count = max(1, int(len(indices) * float(data_cfg.val_split)))
        val_idx = indices[:val_count]
        train_idx = indices[val_count:]
        val_items = [train_items[i] for i in val_idx]
        train_items = [train_items[i] for i in train_idx]

    train_ds = GridMaskDataset(
        train_items,
        augment=data_cfg.augment,
        num_classes=data_cfg.num_classes,
        normalize_mean=(0.485, 0.456, 0.406),
        normalize_std=(0.229, 0.224, 0.225),
    )
    val_ds = GridMaskDataset(
        val_items,
        augment=False,
        num_classes=data_cfg.num_classes,
        normalize_mean=(0.485, 0.456, 0.406),
        normalize_std=(0.229, 0.224, 0.225),
    )

    def _worker_init(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    drop_last_train = len(train_ds) >= data_cfg.batch_size

    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg.batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
        drop_last=drop_last_train,
        worker_init_fn=_worker_init if data_cfg.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=_worker_init if data_cfg.num_workers > 0 else None,
    )
    return train_loader, val_loader


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets, dims)
        denominator = torch.sum(probs + targets, dims)
        dice = (2 * intersection + self.eps) / (denominator + self.eps)
        return 1 - dice.mean()


def dice_metric(logits: Tensor, targets: Tensor) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    intersection = torch.sum(preds * targets)
    denominator = torch.sum(preds + targets)
    if denominator == 0:
        return 1.0
    return float((2 * intersection + 1e-6) / (denominator + 1e-6))


def create_optimizer(params: Iterable[Tensor], cfg: OptimConfig) -> torch.optim.Optimizer:
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
    scaler: amp.GradScaler | None,
    num_classes: int,
    grad_clip: float | None,
    log_interval: int,
) -> dict[str, float]:
    model.train()
    dice_loss_fn = DiceLoss()
    primary_loss_fn: nn.Module
    if num_classes == 1:
        primary_loss_fn = nn.BCEWithLogitsLoss()
    else:
        primary_loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_batches = 0
    tic = time.time()

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with amp.autocast(enabled=scaler is not None):
            outputs = model(images)
            if num_classes == 1:
                primary_loss = primary_loss_fn(outputs, masks)
                if loss_config.dice_weight > 0:
                    dice_loss = dice_loss_fn(outputs, masks)
                    loss = (
                        (1 - loss_config.dice_weight) * primary_loss
                        + loss_config.dice_weight * dice_loss
                    )
                else:
                    loss = primary_loss
            else:
                masks_int = masks.long()
                loss = primary_loss_fn(outputs, masks_int)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += float(loss.detach())
        total_batches += 1

        if step % log_interval == 0:
            elapsed = time.time() - tic
            print(
                f"  step {step:04d}/{len(loader):04d} | loss {loss.item():.4f} | {elapsed:.1f}s"
            )
            tic = time.time()

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
    dice_loss_fn = DiceLoss()
    primary_loss_fn = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            outputs = model(images)
            if num_classes == 1:
                primary_loss = primary_loss_fn(outputs, masks)
                total_loss += float(primary_loss)
                total_dice += dice_metric(outputs, masks)
            else:
                masks_int = masks.long()
                primary_loss = primary_loss_fn(outputs, masks_int)
                total_loss += float(primary_loss)
                # 多分类可选计算 soft dice，这里仅返回 0 以提示
            batches += 1

    metrics = {
        "val_loss": total_loss / max(batches, 1),
    }
    if num_classes == 1:
        metrics["val_dice"] = total_dice / max(batches, 1)
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
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "model_config": asdict(model_config),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def train_model(
    model_cfg: ModelConfig,
    *,
    exp_cfg: ExperimentConfig,
    data_cfg: DataConfig,
    optim_cfg: OptimConfig,
    training_cfg: TrainingConfig,
    loss_cfg: LossConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    print(f"开始训练: {model_cfg.spec.key} ({model_cfg.spec.architecture}/{model_cfg.spec.backbone})")
    model = build_model(
        model_cfg.spec,
        in_channels=model_cfg.in_channels,
        classes=model_cfg.classes,
        encoder_weights=model_cfg.encoder_weights,
        **model_cfg.model_kwargs,
    ).to(device)

    optimizer = create_optimizer(model.parameters(), optim_cfg)
    scaler = amp.GradScaler() if training_cfg.amp and device.type == "cuda" else None

    best_metric = -float("inf")
    history: list[dict[str, float]] = []
    model_dir = exp_cfg.output_dir / model_cfg.spec.key
    (model_dir).mkdir(parents=True, exist_ok=True)
    def _serialize(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_serialize(v) for v in obj]
        return obj

    with (model_dir / "config.json").open("w", encoding="utf-8") as f:
        payload = {
            "experiment": _serialize(asdict(exp_cfg)),
            "model": _serialize(asdict(model_cfg)),
            "data": _serialize(asdict(data_cfg)),
            "optim": _serialize(asdict(optim_cfg)),
            "training": _serialize(asdict(training_cfg)),
            "loss": _serialize(asdict(loss_cfg)),
        }
        json.dump(payload, f, ensure_ascii=False, indent=2)

    for epoch in range(1, training_cfg.epochs + 1):
        print(f"Epoch {epoch}/{training_cfg.epochs}")
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            loss_config=loss_cfg,
            device=device,
            scaler=scaler,
            num_classes=model_cfg.classes,
            grad_clip=training_cfg.grad_clip,
            log_interval=training_cfg.log_interval,
        )
        val_stats = evaluate(
            model,
            val_loader,
            device=device,
            num_classes=model_cfg.classes,
        )
        epoch_stats = {"epoch": epoch, **train_stats, **val_stats}
        history.append(epoch_stats)

        msg = " | ".join(f"{k}: {v:.4f}" for k, v in epoch_stats.items() if isinstance(v, float))
        print(f"  {msg}")

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

        if epoch % 5 == 0:
            save_checkpoint(
                model_dir / f"epoch_{epoch:03d}.ckpt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_stats,
                model_config=model_cfg,
            )

    with (model_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    return {"history": history, "best_metric": best_metric}


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
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

    exp_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with (exp_cfg.output_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(raw_cfg, f, allow_unicode=True)

    set_global_seed(exp_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = prepare_dataloaders(data_cfg, seed=exp_cfg.seed)

    summary: dict[str, Any] = {}
    for model_cfg in model_cfgs:
        result = train_model(
            model_cfg,
            exp_cfg=exp_cfg,
            data_cfg=data_cfg,
            optim_cfg=optim_cfg,
            training_cfg=training_cfg,
            loss_cfg=loss_cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
        )
        summary[model_cfg.spec.key] = result

    with (exp_cfg.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
