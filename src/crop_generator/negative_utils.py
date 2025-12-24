from __future__ import annotations

import random
import string
from pathlib import Path
import math

import cv2
import numpy as np
from numpy.random import Generator as NpGenerator
from PIL import Image

from .config import NegativeSampleParams


def _random_color(rng: random.Random, base: int | None = None, jitter: int = 50) -> tuple[int, int, int]:
    if base is None:
        base = rng.randint(0, 255)
    return tuple(int(np.clip(base + rng.randint(-jitter, jitter), 0, 255)) for _ in range(3))


def _apply_texture(image: np.ndarray, np_rng: NpGenerator, rng: random.Random, sigma: float) -> np.ndarray:
    height, width, _ = image.shape
    # generate channel-wise smooth noise
    noise_offset = rng.normalvariate(0.0, 1.0)
    texture = np_rng.normal(loc=noise_offset, scale=1.0, size=(height, width, 3)).astype(np.float32)
    ksize = max(3, int(sigma * 3) | 1)  # ensure odd
    texture = cv2.GaussianBlur(texture, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT101)
    texture *= rng.uniform(8.0, 18.0)
    mixed = image.astype(np.float32) + texture
    return np.clip(mixed, 0, 255).astype(np.uint8)


def _draw_random_shapes(image: np.ndarray, params: NegativeSampleParams, rng: random.Random) -> None:
    h, w, _ = image.shape
    count = rng.randint(*params.shape_count_range)
    for _ in range(count):
        shape_type = rng.choice(["ellipse", "rectangle", "polygon"])
        center = (rng.randint(-w // 4, w + w // 4), rng.randint(-h // 4, h + h // 4))
        color = _random_color(rng, base=rng.randint(60, 180), jitter=90)
        thickness = -1 if rng.random() < 0.4 else rng.randint(1, 3)
        if shape_type == "ellipse":
            axes = (rng.randint(20, 140), rng.randint(20, 140))
            angle = rng.uniform(0, 180)
            start_angle = 0
            end_angle = 360
            cv2.ellipse(image, center, axes, angle, start_angle, end_angle, color, thickness, lineType=cv2.LINE_AA)
        elif shape_type == "rectangle":
            width = rng.randint(60, 180)
            height = rng.randint(40, 160)
            top_left = (center[0] - width // 2, center[1] - height // 2)
            bottom_right = (top_left[0] + width, top_left[1] + height)
            cv2.rectangle(image, top_left, bottom_right, color, thickness, lineType=cv2.LINE_AA)
        else:  # polygon
            sides = rng.randint(3, 6)
            radius = rng.randint(30, 130)
            points = []
            for i in range(sides):
                angle = 2 * np.pi * i / sides + rng.uniform(-0.3, 0.3)
                px = int(center[0] + radius * np.cos(angle))
                py = int(center[1] + radius * np.sin(angle))
                points.append([px, py])
            pts = np.array(points, dtype=np.int32)
            if thickness < 0:
                cv2.fillPoly(image, [pts], color, lineType=cv2.LINE_AA)
            else:
                cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def _draw_random_lines(image: np.ndarray, params: NegativeSampleParams, rng: random.Random) -> None:
    h, w, _ = image.shape
    count = rng.randint(*params.line_count_range)
    for _ in range(count):
        pt1 = (rng.randint(-w // 3, w + w // 3), rng.randint(-h // 3, h + h // 3))
        pt2 = (rng.randint(-w // 3, w + w // 3), rng.randint(-h // 3, h + h // 3))
        color = _random_color(rng, base=rng.randint(40, 160), jitter=110)
        thickness = rng.randint(*params.line_thickness_range)
        cv2.line(image, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)


def _draw_random_text(image: np.ndarray, params: NegativeSampleParams, rng: random.Random) -> None:
    h, w, _ = image.shape
    count = rng.randint(*params.text_count_range)
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
    ]
    charset = string.ascii_uppercase + string.digits
    for _ in range(count):
        length = rng.randint(1, 4)
        text = "".join(rng.choice(charset) for _ in range(length))
        font = rng.choice(fonts)
        font_scale = rng.uniform(0.6, 1.8)
        thickness = rng.randint(1, 3)
        color = _random_color(rng, base=rng.randint(80, 200), jitter=80)
        x = rng.randint(-w // 8, w - 1)
        y = rng.randint(int(h * 0.2), int(h * 0.95))
        cv2.putText(
            image,
            text,
            (x, y),
            fontFace=font,
            fontScale=font_scale,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )


def _apply_periodic_pattern(image: np.ndarray, params: NegativeSampleParams, rng: random.Random) -> np.ndarray:
    height, width, _ = image.shape
    period = rng.uniform(*params.pattern_period_range)
    period = max(4.0, period)
    base_angle = rng.choice([0.0, 45.0, 90.0, 135.0])
    angle_offset = rng.uniform(*params.pattern_angle_jitter)
    angle = math.radians(base_angle + angle_offset)
    freq = 2.0 * math.pi / period
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    rotated = xx * math.cos(angle) + yy * math.sin(angle)
    phase = rng.uniform(0.0, 2.0 * math.pi)
    pattern = np.sin(freq * rotated + phase)
    if rng.random() < 0.4:
        harmonic = rng.uniform(1.8, 2.6)
        pattern += 0.35 * np.sin(freq * harmonic * rotated + rng.uniform(0.0, 2.0 * math.pi))
    pattern = np.clip(pattern, -1.0, 1.0)

    amplitude = rng.uniform(*params.pattern_amplitude_range)
    blend = rng.uniform(*params.pattern_blend_range)

    base = image.astype(np.float32)
    delta = amplitude * pattern

    if rng.random() < 0.5:
        luminance = 0.299 * base[..., 0] + 0.587 * base[..., 1] + 0.114 * base[..., 2]
        mod_luminance = luminance + delta * blend
        diff = (mod_luminance - luminance)[..., None]
        base += diff
    else:
        channel_weights = np.array([rng.uniform(0.7, 1.3) for _ in range(3)], dtype=np.float32)
        channel_weights /= channel_weights.max() + 1e-5
        base += (delta * blend)[..., None] * channel_weights

    np.clip(base, 0, 255, out=base)
    return base.astype(np.uint8)


def synthesize_negative_image(params: NegativeSampleParams, rng: random.Random) -> np.ndarray:
    size = params.image_size
    np_rng = np.random.default_rng(rng.randrange(1 << 32))
    base_intensity = rng.randint(*params.background_intensity_range)
    background = np.zeros((size, size, 3), dtype=np.uint8)
    base_color = np.array(
        [
            int(np.clip(base_intensity + rng.randint(-params.background_jitter, params.background_jitter), 0, 255))
            for _ in range(3)
        ],
        dtype=np.uint8,
    )
    background[:] = base_color

    if rng.random() < params.texture_probability:
        sigma = rng.uniform(3.0, 9.0)
        background = _apply_texture(background, np_rng, rng, sigma)

    if rng.random() < params.pattern_probability:
        background = _apply_periodic_pattern(background, params, rng)

    _draw_random_shapes(background, params, rng)
    _draw_random_lines(background, params, rng)

    if rng.random() < params.text_probability:
        _draw_random_text(background, params, rng)

    sigma_noise = rng.uniform(*params.noise_sigma_range)
    noise = np_rng.normal(0, sigma_noise, size=background.shape).astype(np.float32)
    blended = np.clip(background.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if rng.random() < params.pattern_probability * 0.5:
        blended = _apply_periodic_pattern(blended, params, rng)

    kernel = rng.choice(params.blur_kernel_choices)
    if kernel and kernel > 1:
        if kernel % 2 == 0:
            kernel += 1
        blended = cv2.GaussianBlur(blended, (kernel, kernel), sigmaX=0.0, borderType=cv2.BORDER_REFLECT101)

    return blended


def generate_negative_sample(
    output_dir: Path,
    params: NegativeSampleParams,
    rng: random.Random,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image = synthesize_negative_image(params, rng)
    mask = np.zeros((params.image_size, params.image_size), dtype=np.uint8)

    Image.fromarray(image).save(output_dir / "image.png")
    Image.fromarray(mask).save(output_dir / "mask.png")
