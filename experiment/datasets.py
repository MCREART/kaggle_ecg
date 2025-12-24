"""
数据加载工具：读取合成网格图像与掩码。

默认假设目录结构为：
root/
  images/grid_xxxxx.png
  masks/grid_xxxxx_mask.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


def _load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask


def _random_augment(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """简单的镜像增强，保持图像与掩码对齐。"""

    if np.random.rand() < 0.5:
        image = np.flip(image, axis=1)
        mask = np.flip(mask, axis=1)
    if np.random.rand() < 0.5:
        image = np.flip(image, axis=0)
        mask = np.flip(mask, axis=0)
    if np.random.rand() < 0.3:
        angle = np.random.choice([0, 90, 180, 270])
        if angle:
            k = angle // 90
            image = np.rot90(image, k)
            mask = np.rot90(mask, k)
    return image.copy(), mask.copy()


@dataclass(frozen=True)
class SampleItem:
    image_path: Path
    mask_path: Path


class GridMaskDataset(Dataset):
    def __init__(
        self,
        items: Sequence[SampleItem],
        *,
        augment: bool,
        num_classes: int,
        normalize_mean: Iterable[float] = (0.485, 0.456, 0.406),
        normalize_std: Iterable[float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.items = list(items)
        self.augment = augment
        self.num_classes = num_classes
        self.normalize_mean = np.array(list(normalize_mean), dtype=np.float32)
        self.normalize_std = np.array(list(normalize_std), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        # 简单的重试机制：如果当前样本坏了，就试下一个
        try:
            item = self.items[index]
            image = _load_image(item.image_path)
            mask = _load_mask(item.mask_path)
        except Exception as e:
            # 随机换一个索引重试 (避免死循环，最多试5次)
            print(f"Warning: Failed to load {self.items[index].image_path}: {e}. Retrying...")
            for _ in range(5):
                try:
                    alt_idx = np.random.randint(len(self.items))
                    item = self.items[alt_idx]
                    image = _load_image(item.image_path)
                    mask = _load_mask(item.mask_path)
                    break
                except Exception:
                    continue
            else:
                # 实在不行就抛异常
                raise e

        if self.augment:
            image, mask = _random_augment(image, mask)

        image = image.astype(np.float32) / 255.0
        # 增加 epsilon 防止除 0
        image = (image - self.normalize_mean) / (self.normalize_std + 1e-6)
        
        # 检查 NaN
        if np.isnan(image).any():
             image = np.nan_to_num(image)
             
        image = torch.from_numpy(image.transpose(2, 0, 1))

        if self.num_classes == 1:
            mask_tensor = torch.from_numpy((mask > 127).astype(np.float32))[None, ...]
        else:
            mask_tensor = torch.from_numpy(mask.astype(np.int64))

        return {"image": image, "mask": mask_tensor}


def discover_samples(root: Path) -> List[SampleItem]:
    """扫描目录下的图像/掩码对。"""

    images_dir = root / "images"
    masks_dir = root / "masks"

    if not images_dir.is_dir() or not masks_dir.is_dir():
        raise FileNotFoundError(f"{root} 应包含 images/ 与 masks/ 子目录")

    items: list[SampleItem] = []
    for image_path in sorted(images_dir.glob("*.png")):
        stem = image_path.stem
        mask_path = masks_dir / f"{stem}_mask.png"
        if not mask_path.exists():
            # 尝试无后缀形式（以防用户手动命名）
            alt_mask = masks_dir / f"{stem}.png"
            if alt_mask.exists():
                mask_path = alt_mask
            else:
                raise FileNotFoundError(f"未找到 {mask_path}")
        items.append(SampleItem(image_path=image_path, mask_path=mask_path))

    if not items:
        raise RuntimeError(f"{root} 下未发现图像文件")

    return items
