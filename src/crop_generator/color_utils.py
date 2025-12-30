
from __future__ import annotations

import random
import cv2
import numpy as np

from .config import ColorAugParams

def _choose_range(range_tuple):
    low, high = range_tuple
    if low > high:
        low, high = high, low
    return low, high

def apply_color_augmentations(image: np.ndarray, params: ColorAugParams) -> np.ndarray:
    """
    Apply global color augmentations: Brightness, Contrast, Saturation, Hue, White Balance.
    Image is expected to be BGR (uint8).
    
    If params.target_red_only is True, mask out non-red pixels and apply augmentations only to the red grid.
    """
    if not params.enabled:
        return image

    output = image.astype(np.float32)
    original_bgr = output.copy()
    
    # --- Mask Creation (if we only want to target red grid) ---
    # Assuming 'red grid' has high Red channel relative to Blue/Green
    # A simpleheuristic: R > 100 and R > B + 30 and R > G + 30
    # Or convert to HSV and pick Red range.
    # Let's use a soft mask or binary mask. For sharpness, binary is cleaner but soft blends better.
    # Given the requirements "only for red pixels", binary mask is safer to leave white background untouched.
    
    # Detect red pixels
    # BGR
    b, g, r = output[:,:,0], output[:,:,1], output[:,:,2]
    # Red Condition: Red is dominant and bright enough
    is_red = (r > 120) & (r > g + 20) & (r > b + 20)
    # Be more robust: also include pinkish/lighter reds if they are part of the grid? 
    # The grid is usually anti-aliased, so edges are pink.
    # Let's use a slightly relaxed condition to catch anti-aliasing, but avoid white background.
    # White background: R,G,B all high. e.g. > 200.
    is_not_white = (r < 250) | (g < 250) | (b < 250)
    
    target_mask = (is_red & is_not_white).astype(np.float32)
    # Expand dims to (H,W,1) for broadcasting
    target_mask = target_mask[:, :, None]

    # --- 1. White Balance (Warmth) ---
    warn_min, warn_max = _choose_range(params.warmth_range)
    warmth = random.uniform(warn_min, warn_max)
    
    if abs(warmth - 1.0) > 0.01:
        shift = warmth - 1.0
        r_scale = 1.0 + (shift * 0.5)
        b_scale = 1.0 - (shift * 0.5)
        
        output[:, :, 0] *= b_scale # B
        output[:, :, 2] *= r_scale # R
        np.clip(output, 0, 255, out=output)

    # --- 2. Brightness & Contrast ---
    bri_min, bri_max = _choose_range(params.brightness_range)
    brightness_factor = random.uniform(bri_min, bri_max)
    
    con_min, con_max = _choose_range(params.contrast_range)
    contrast_factor = random.uniform(con_min, con_max)
    
    if abs(brightness_factor - 1.0) > 0.01 or abs(contrast_factor - 1.0) > 0.01:
        output *= brightness_factor
        gray_mean = np.mean(cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_BGR2GRAY))
        output = (output - gray_mean) * contrast_factor + gray_mean
        np.clip(output, 0, 255, out=output)

    # --- 3. Hsv Adjustments (Saturation & Hue) ---
    sat_min, sat_max = _choose_range(params.saturation_range)
    saturation_factor = random.uniform(sat_min, sat_max)
    
    hue_min, hue_max = _choose_range(params.hue_range)
    hue_factor = random.uniform(hue_min, hue_max)
    
    if abs(saturation_factor - 1.0) > 0.01 or abs(hue_factor) > 0.001:
        hsv = cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= saturation_factor
        h5_shift = hue_factor * 180.0
        hsv[:, :, 0] += h5_shift
        hsv[:, :, 0] = np.mod(hsv[:, :, 0], 180.0)
        hsv = np.clip(hsv, 0, 255)
        output = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    output = np.clip(output, 0, 255).astype(np.uint8)

    # --- Blend back based on mask ---
    # Result = Mask * Augmented + (1-Mask) * Original
    # However we operated in float, so output is uint8 now. Use float blending.
    
    final_output = output.astype(np.float32) * target_mask + original_bgr * (1.0 - target_mask)
    
    return np.clip(final_output, 0, 255).astype(np.uint8)
