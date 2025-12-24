from __future__ import annotations

import math
import random

import cv2
import numpy as np

from .config import MoireParams


def _choose_range(range_tuple):
    low, high = range_tuple
    if low > high:
        low, high = high, low
    return low, high


def apply_moire(image: np.ndarray, params: MoireParams) -> np.ndarray:
    if not params.enabled:
        return image

    height, width = image.shape[:2]
    short_side = max(1, min(height, width))

    period_low, period_high = _choose_range(params.period_range)
    period = max(2.0, random.uniform(period_low, period_high))
    beat_low, beat_high = _choose_range(params.beat_ratio_range)
    ratio = max(1.0 + 1e-4, random.uniform(beat_low, beat_high))
    angle_low, angle_high = _choose_range(params.angle_range)
    angle = math.radians(random.uniform(angle_low, angle_high))
    amp_low, amp_high = _choose_range(params.amplitude_range)
    amplitude = max(0.0, random.uniform(amp_low, amp_high))
    phase_low, phase_high = _choose_range(params.phase_range)
    phase1 = random.uniform(phase_low, phase_high)
    phase2 = random.uniform(phase_low, phase_high)
    warp_low, warp_high = _choose_range(params.warp_strength_range)
    warp_strength = max(0.0, random.uniform(warp_low, warp_high))
    warp_freq_low, warp_freq_high = _choose_range(params.warp_frequency_range)
    warp_frequency = max(0.0, random.uniform(warp_freq_low, warp_freq_high))
    texture_low, texture_high = _choose_range(params.texture_strength_range)
    texture_strength = max(0.0, random.uniform(texture_low, texture_high))
    tex_sigma_low, tex_sigma_high = _choose_range(params.texture_sigma_range)
    texture_sigma = max(0.0, random.uniform(tex_sigma_low, tex_sigma_high) * short_side)
    blend_low, blend_high = _choose_range(params.blend_range)
    blend = max(0.0, min(1.0, random.uniform(blend_low, blend_high)))
    cg_low, cg_high = _choose_range(params.contrast_gain_range)
    contrast_gain = max(1.0, random.uniform(cg_low, cg_high))
    secondary_prob = max(0.0, min(1.0, params.secondary_prob))
    sec_angle_low, sec_angle_high = _choose_range(params.secondary_angle_offset_range)
    sec_amp_low, sec_amp_high = _choose_range(params.secondary_amplitude_scale_range)

    freq1 = (2.0 * math.pi) / period
    freq2 = freq1 * ratio

    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    rotated = xx * math.cos(angle) + yy * math.sin(angle)
    orth = -xx * math.sin(angle) + yy * math.cos(angle)

    if warp_strength > 0.0 and warp_frequency > 0.0:
        warp_phase = random.uniform(0.0, 2.0 * math.pi)
        warp_component = warp_strength * short_side * np.sin(
            orth / short_side * 2.0 * math.pi * warp_frequency + warp_phase
        )
        rotated = rotated + warp_component

    pattern = np.sin(freq1 * rotated + phase1) + np.sin(freq2 * rotated + phase2)

    if random.random() < secondary_prob:
        sign = -1.0 if random.random() < 0.5 else 1.0
        offset_deg = random.uniform(sec_angle_low, sec_angle_high)
        secondary_angle = angle + sign * math.radians(offset_deg)
        rotated2 = xx * math.cos(secondary_angle) + yy * math.sin(secondary_angle)
        orth2 = -xx * math.sin(secondary_angle) + yy * math.cos(secondary_angle)
        if warp_strength > 0.0:
            warp_phase2 = random.uniform(0.0, 2.0 * math.pi)
            rotated2 = rotated2 + (warp_strength * 0.7) * short_side * np.sin(
                orth2 / short_side * 2.0 * math.pi * warp_frequency + warp_phase2
            )
        phase3 = random.uniform(phase_low, phase_high)
        phase4 = random.uniform(phase_low, phase_high)
        pattern2 = np.sin(freq1 * rotated2 + phase3) + np.sin(freq2 * rotated2 + phase4)
        amp_scale = max(0.0, random.uniform(sec_amp_low, sec_amp_high))
        pattern = (pattern + amp_scale * pattern2) / (1.0 + amp_scale)

    pattern = pattern / 2.0
    if contrast_gain > 1.0:
        pattern = np.tanh(pattern * contrast_gain)

    if texture_strength > 0.0:
        noise = np.random.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
        if texture_sigma > 0.0:
            noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=texture_sigma, sigmaY=texture_sigma, borderType=cv2.BORDER_REFLECT101)
        noise = noise / (np.max(np.abs(noise)) + 1e-6)
        pattern *= 1.0 + texture_strength * noise

    pattern = np.clip(pattern, -1.0, 1.0)
    modulation = 1.0 + amplitude * pattern

    output = image.astype(np.float32)
    if output.ndim == 2:
        modulated = output * modulation
    else:
        luminance = (
            0.299 * output[..., 0]
            + 0.587 * output[..., 1]
            + 0.114 * output[..., 2]
        )
        mod_luminance = luminance * modulation
        luminance_delta = (mod_luminance - luminance)[..., None]
        modulated = output + luminance_delta

    if modulated.ndim == 3:
        shift_prob = max(0.0, min(1.0, params.color_shift_prob))
        if shift_prob > 0.0 and params.color_strength_range[1] > 0.0 and random.random() < shift_prob:
            strength_low, strength_high = _choose_range(params.color_strength_range)
            strength = max(0.0, random.uniform(strength_low, strength_high))
            if strength > 0.0:
                direction = np.random.normal(0.0, 1.0, size=3).astype(np.float32)
                norm = float(np.linalg.norm(direction))
                if norm < 1e-6:
                    direction = np.array([1.0, -0.5, 0.5], dtype=np.float32)
                    norm = float(np.linalg.norm(direction))
                direction /= norm
                color_mod = 1.0 + strength * pattern[..., None] * direction
                np.clip(color_mod, 0.25, 2.5, out=color_mod)
                modulated = modulated * color_mod
        if params.tint_strength_range[1] > 0.0:
            tint_low, tint_high = _choose_range(params.tint_strength_range)
            tint_strength = max(0.0, random.uniform(tint_low, tint_high))
            if tint_strength > 0.0:
                tint_color = np.random.uniform(0.0, 1.0, size=3).astype(np.float32)
                tint_direction = (tint_color - 0.5) * 2.0  # [-1, 1] per channel
                tint = tint_strength * pattern[..., None] * tint_direction * 255.0
                modulated = modulated + tint

    if blend < 1.0:
        output = output * (1.0 - blend) + modulated * blend
    else:
        output = modulated

    np.clip(output, 0, 255, out=output)
    return output.astype(image.dtype)
