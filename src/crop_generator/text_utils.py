from __future__ import annotations

import random
import string
from typing import Tuple

import cv2
import numpy as np

from .config import TextOverlayParams

def _random_color(rng: random.Random, base: int | None = None, jitter: int = 50) -> tuple[int, int, int]:
    if base is None:
        base = rng.randint(0, 100) # Darker text by default
    return tuple(int(np.clip(base + rng.randint(-jitter, jitter), 0, 255)) for _ in range(3))

def apply_text_overlay(
    image: np.ndarray, 
    mask: np.ndarray, 
    params: TextOverlayParams, 
    rng: random.Random
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Overlays random text on the image and updates the mask.
    If params.clear_mask is True, the mask values under the text are set to 0 (Background).
    """
    if not params.enabled:
        return image, mask

    h, w = image.shape[:2]
    # Create a separate mask for where text is drawn
    text_mask = np.zeros((h, w), dtype=np.uint8)
    
    count = rng.randint(*params.count_range)
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
    ]
    charset = string.ascii_uppercase + string.digits + " ./-"
    
    for _ in range(count):
        length = rng.randint(2, 8)
        text = "".join(rng.choice(list(charset)) for _ in range(length))
        font = rng.choice(fonts)
        font_scale = rng.uniform(*params.font_scale_range)
        thickness = rng.randint(*params.thickness_range)
        
        # Determine size to center or place randomly
        (t_w, t_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Place randomly but fully within bounds if possible
        x_max = max(0, w - t_w)
        y_max = max(t_h, h - baseline) # y is bottom-left of text
        if x_max == 0 or y_max == t_h:
            continue
            
        x = rng.randint(0, x_max)
        y = rng.randint(t_h, y_max)
        
        # Draw on text_mask (white text on black bg)
        cv2.putText(
            text_mask,
            text,
            (x, y),
            fontFace=font,
            fontScale=font_scale,
            color=255, # Binary mask
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
        
        # Determine color for this text instance
        # Standard printed text is usually dark
        color = _random_color(rng, base=rng.randint(0, 60), jitter=20)
        
        # Draw on image carefully if we want alpha blending or just overwrite
        # For simplicity and "ink" look, overwrite is often fine, but we can do simple blend
        # where text_mask > 0.
        
        # To handle multicolor text (if needed) or just applying the color:
        # We can do it per word or apply after loop. 
        # Doing it inside loop allows different colors per word.
        
        # Mask for this specific word
        word_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.putText(word_mask, text, (x, y), font, font_scale, 255, thickness, cv2.LINE_AA)
        
        opacity = rng.uniform(*params.opacity_range)
        
        # Apply to image
        if image.ndim == 3:
            # Create colored text layer
            color_layer = np.full_like(image, color, dtype=np.uint8)
            alpha = (word_mask > 0).astype(np.float32)[:, :, None] * opacity
            
            # Blend: image * (1-alpha) + color * alpha
            image = (image.astype(np.float32) * (1 - alpha) + color_layer.astype(np.float32) * alpha).astype(np.uint8)
        else:
            # Grayscale image
            color_val = sum(color) // 3
            color_layer = np.full_like(image, color_val, dtype=np.uint8)
            alpha = (word_mask > 0).astype(np.float32) * opacity
            image = (image.astype(np.float32) * (1 - alpha) + color_layer.astype(np.float32) * alpha).astype(np.uint8)
            
    # Modify the ground truth mask
    if params.clear_mask:
        # text_mask contains all text pixels
        # 0: Background
        # 1: Grid
        # 2: Wave
        # We want to set text pixels to 0 (Background)
        # This explicitly tells model: "This black stuff is NOT wave"
        mask[text_mask > 0] = 0

    return image, mask
