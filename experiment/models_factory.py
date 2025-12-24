"""
Utility helpers to instantiate segmentation models for the grid-line experiments.

The recommended architectures and backbones come from ``experiment/readme``.
All models are instantiated via ``segmentation_models_pytorch`` so they stay
consistent with the rest of the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Mapping

import segmentation_models_pytorch as smp
from torch import nn


@dataclass(frozen=True)
class ModelSpec:
    """Immutable description of one experiment variant."""

    key: str
    architecture: str
    backbone: str
    notes: str


ARCHITECTURE_BUILDERS: Mapping[str, type[nn.Module]] = {
    "unet": smp.Unet,
    "unet++": smp.UnetPlusPlus,
    "fpn": smp.FPN,
    "deeplabv3+": smp.DeepLabV3Plus,
    "pspnet": smp.PSPNet,
}


RECOMMENDED_SPECS: Iterable[ModelSpec] = (
    ModelSpec(
        key="unet_resnet34",
        architecture="unet",
        backbone="resnet34",
        notes="Good default; small parameter count keeps training stable.",
    ),
    ModelSpec(
        key="unet_efficientnetb0",
        architecture="unet",
        backbone="efficientnet-b0",
        notes="Stronger encoder for fine grids; pair with dropout if overfitting.",
    ),
    ModelSpec(
        key="unet++_resnet50",
        architecture="unet++",
        backbone="resnet50",
        notes="Dense skip connections highlight thin edges; tune deep-supervision.",
    ),
    ModelSpec(
        key="unet++_mobilenetv3s",
        architecture="unet++",
        backbone="timm-mobilenetv3_small_100",
        notes="Lightweight variant when memory is limited.",
    ),
    ModelSpec(
        key="fpn_resnet50",
        architecture="fpn",
        backbone="resnet50",
        notes="Top-down fusion for noisy backgrounds; combine Dice + Focal loss.",
    ),
    ModelSpec(
        key="fpn_swint",
        architecture="fpn",
        backbone="tu-swin_tiny_patch4_window7_224",
        notes="Transformer encoder captures long-range context on sparse grids.",
    ),
    ModelSpec(
        key="deeplabv3+_resnet101",
        architecture="deeplabv3+",
        backbone="resnet101",
        notes="ASPP enlarges receptive field; use sliding window for high-res.",
    ),
    ModelSpec(
        key="deeplabv3+_mitb3",
        architecture="deeplabv3+",
        backbone="mit_b3",
        notes="SegFormer backbone balances receptive field and efficiency.",
    ),
    ModelSpec(
        key="pspnet_resnet101",
        architecture="pspnet",
        backbone="resnet101",
        notes="Pyramid pooling supplies global context; smooth but may need CRF.",
    ),
    ModelSpec(
        key="pspnet_convnextt",
        architecture="pspnet",
        backbone="tu-convnext_tiny",
        notes="Improves regularity detection thanks to ConvNeXt feature maps.",
    ),
)


def build_model(
    spec: ModelSpec,
    *,
    in_channels: int = 3,
    classes: int = 1,
    encoder_weights: str | None = "imagenet",
    **kwargs,
) -> nn.Module:
    """
    Instantiate one segmentation model described by ``spec``.

    ``kwargs`` are forwarded to the architecture constructor, making it easy to
    override settings such as ``decoder_attention_type`` or loss-friendly flags.
    """

    builder = ARCHITECTURE_BUILDERS[spec.architecture]
    return builder(
        encoder_name=spec.backbone,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        **kwargs,
    )


def build_all_models(
    *,
    in_channels: int = 3,
    classes: int = 1,
    encoder_weights: str | None = "imagenet",
    **kwargs,
) -> Dict[str, nn.Module]:
    """
    Instantiate every recommended experiment variant.

    Returns a dict keyed by the ``ModelSpec.key`` so callers can pick whichever
    subset they need for a training loop.
    """

    return {
        spec.key: build_model(
            spec,
            in_channels=in_channels,
            classes=classes,
            encoder_weights=encoder_weights,
            **kwargs,
        )
        for spec in RECOMMENDED_SPECS
    }


def iter_model_specs() -> Iterator[ModelSpec]:
    """
    Iterate over all recommended configurations.

    Helpful when wiring the models into experiment trackers or CLI choices.
    """

    yield from RECOMMENDED_SPECS


__all__ = [
    "ModelSpec",
    "build_model",
    "build_all_models",
    "iter_model_specs",
    "RECOMMENDED_SPECS",
]
