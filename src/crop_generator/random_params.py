from __future__ import annotations

import random
from dataclasses import replace
from typing import List

from .config import (
    BlurParams,
    GrayParams,
    MoireParams,
    NoiseParams,
    OcclusionParams,
    StainParams,
    TransformParams,
    WrinkleParams,
)


def random_transform(rng: random.Random) -> TransformParams:
    return TransformParams(
        mode="perspective",
        max_rotate=rng.uniform(4.0, 10.0),
        max_shift=rng.uniform(0.05, 0.12),
        max_scale=rng.uniform(0.05, 0.12),
        perspective_jitter=rng.uniform(0.08, 0.18),
    )


def random_wrinkle(rng: random.Random) -> WrinkleParams:
    if rng.random() < 0.55:
        max_count = rng.randint(2, 4)
        min_count = rng.randint(1, max(1, max_count - 1))
        amp_low = rng.uniform(0.012, 0.025)
        amp_high = rng.uniform(0.03, 0.045)
        sigma_low = rng.uniform(0.015, 0.035)
        sigma_high = rng.uniform(0.04, 0.085)
        wave_low = rng.uniform(0.06, 0.12)
        wave_high = rng.uniform(0.12, 0.22)
        return WrinkleParams(
            enabled=True,
            count_range=(min_count, max_count),
            amplitude_range=(amp_low, amp_high),
            sigma_range=(sigma_low, sigma_high),
            wavelength_range=(wave_low, wave_high),
        )
    return WrinkleParams(enabled=False)


def random_blur(rng: random.Random, *, force: bool = False) -> BlurParams:
    if not force and rng.random() >= 0.5:
        return BlurParams(enabled=False)

    kernel_low = rng.choice([3, 5])
    kernel_high = rng.choice([7, 9])
    sigma_low = rng.uniform(0.4, 1.0)
    sigma_high = rng.uniform(1.0, 2.2)
    if sigma_low > sigma_high:
        sigma_low, sigma_high = sigma_high, sigma_low
    return BlurParams(
        enabled=True,
        kernel_range=(kernel_low, kernel_high),
        sigma_range=(sigma_low, sigma_high),
    )


def random_noise(rng: random.Random) -> NoiseParams:
    if rng.random() < 0.7:
        sigma_low = rng.uniform(6.0, 14.0)
        sigma_high = rng.uniform(14.0, 24.0)
        if sigma_low > sigma_high:
            sigma_low, sigma_high = sigma_high, sigma_low
        return NoiseParams(enabled=True, sigma_range=(sigma_low, sigma_high))
    return NoiseParams(enabled=False)


def random_occlusion(rng: random.Random) -> OcclusionParams:
    if rng.random() < 0.6:
        count_high = rng.randint(1, 3)
        count_low = rng.randint(1, count_high)
        size_low = rng.uniform(0.1, 0.18)
        size_high = rng.uniform(0.2, 0.32)
        intensity_high = rng.uniform(50, 90)
        return OcclusionParams(
            enabled=True,
            count_range=(count_low, count_high),
            size_range=(size_low, size_high),
            intensity_range=(0, int(intensity_high)),
        )
    return OcclusionParams(enabled=False)


def random_stain(rng: random.Random) -> StainParams:
    if rng.random() < 0.55:
        count_high = rng.randint(2, 4)
        count_low = rng.randint(1, max(1, count_high - 1))
        size_low = rng.uniform(0.15, 0.25)
        size_high = rng.uniform(0.28, 0.4)
        intensity_low = rng.uniform(160, 200)
        intensity_high = rng.uniform(200, 255)
        softness_low = rng.uniform(0.18, 0.3)
        softness_high = rng.uniform(0.3, 0.5)
        tint_strength_low = rng.uniform(0.4, 0.7)
        tint_strength_high = rng.uniform(tint_strength_low, 0.95)
        texture_low = rng.uniform(0.6, 0.9)
        texture_high = rng.uniform(0.9, 1.2)
        scale_low = rng.uniform(0.06, 0.12)
        scale_high = rng.uniform(0.12, 0.28)
        return StainParams(
            enabled=True,
            count_range=(count_low, count_high),
            size_range=(size_low, size_high),
            intensity_range=(intensity_low, intensity_high),
            softness_range=(softness_low, softness_high),
            tint_color=(160, 110, 50),
            tint_strength_range=(tint_strength_low, tint_strength_high),
            texture_strength_range=(texture_low, texture_high),
            texture_scale_range=(scale_low, scale_high),
        )
    return StainParams(enabled=False)


def random_moire(rng: random.Random, *, force: bool = False) -> MoireParams:
    if not force and rng.random() >= 0.45:
        return MoireParams(enabled=False)

    period_low = rng.uniform(4.5, 9.0)
    period_high = rng.uniform(9.0, 15.0)
    beat_low = rng.uniform(1.04, 1.1)
    beat_high = rng.uniform(1.1, 1.22)
    angle_low = rng.uniform(-30.0, -10.0)
    angle_high = rng.uniform(10.0, 30.0)
    amplitude_low = rng.uniform(0.16, 0.22)
    amplitude_high = rng.uniform(0.22, 0.32)
    warp_low = rng.uniform(0.02, 0.08)
    warp_high = rng.uniform(0.08, 0.28)
    warp_freq_low = rng.uniform(0.25, 0.55)
    warp_freq_high = rng.uniform(0.6, 1.4)
    texture_low = rng.uniform(0.04, 0.12)
    texture_high = rng.uniform(0.12, 0.26)
    texture_sigma_low = rng.uniform(0.04, 0.08)
    texture_sigma_high = rng.uniform(0.08, 0.16)
    blend_low = rng.uniform(0.72, 0.85)
    blend_high = rng.uniform(max(blend_low + 0.03, 0.78), 0.97)
    contrast_low = rng.uniform(1.4, 2.0)
    contrast_high = rng.uniform(max(contrast_low + 0.2, 1.8), 3.4)
    secondary_prob = rng.uniform(0.28, 0.55)
    secondary_angle_low = rng.uniform(5.0, 10.0)
    secondary_angle_high = rng.uniform(max(secondary_angle_low + 3.0, 9.0), 24.0)
    secondary_amp_low = rng.uniform(0.3, 0.5)
    secondary_amp_high = rng.uniform(max(secondary_amp_low + 0.1, 0.55), 0.9)
    color_prob = rng.uniform(0.45, 0.85)
    color_strength_low = rng.uniform(0.08, 0.18)
    color_strength_high = rng.uniform(max(color_strength_low + 0.04, 0.18), 0.34)
    tint_strength_low = rng.uniform(0.05, 0.15)
    tint_strength_high = rng.uniform(max(tint_strength_low + 0.05, 0.2), 0.4)
    return MoireParams(
        enabled=True,
        period_range=(period_low, period_high),
        beat_ratio_range=(beat_low, beat_high),
        angle_range=(angle_low, angle_high),
        amplitude_range=(amplitude_low, amplitude_high),
        warp_strength_range=(warp_low, warp_high),
        warp_frequency_range=(warp_freq_low, warp_freq_high),
        texture_strength_range=(texture_low, texture_high),
        texture_sigma_range=(texture_sigma_low, texture_sigma_high),
        blend_range=(blend_low, blend_high),
        contrast_gain_range=(contrast_low, contrast_high),
        secondary_prob=secondary_prob,
        secondary_angle_offset_range=(secondary_angle_low, secondary_angle_high),
        secondary_amplitude_scale_range=(secondary_amp_low, secondary_amp_high),
        color_shift_prob=color_prob,
        color_strength_range=(color_strength_low, color_strength_high),
        tint_strength_range=(tint_strength_low, tint_strength_high),
    )


def random_gray(rng: random.Random) -> GrayParams:
    if rng.random() < 0.4:
        single_channel = rng.random() < 0.5
        return GrayParams(enabled=True, preserve_channels=not single_channel)
    return GrayParams(enabled=False)


def ensure_any_module(modules: List[object], rng: random.Random) -> None:
    if any(getattr(m, "enabled", False) for m in modules):
        return
    idx = rng.randrange(len(modules))
    module = modules[idx]
    if isinstance(module, WrinkleParams):
        modules[idx] = replace(random_wrinkle(rng), enabled=True)
    elif isinstance(module, BlurParams):
        modules[idx] = random_blur(rng, force=True)
    elif isinstance(module, NoiseParams):
        modules[idx] = replace(random_noise(rng), enabled=True)
    elif isinstance(module, OcclusionParams):
        modules[idx] = replace(random_occlusion(rng), enabled=True)
    elif isinstance(module, StainParams):
        modules[idx] = replace(random_stain(rng), enabled=True)
    elif isinstance(module, MoireParams):
        modules[idx] = replace(random_moire(rng), enabled=True)
    elif isinstance(module, GrayParams):
        modules[idx] = replace(random_gray(rng), enabled=True)


__all__ = [
    "random_transform",
    "random_wrinkle",
    "random_blur",
    "random_noise",
    "random_occlusion",
    "random_stain",
    "random_moire",
    "random_gray",
    "ensure_any_module",
]
