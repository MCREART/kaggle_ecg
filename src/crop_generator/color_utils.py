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
    """
    if not params.enabled:
        return image

    output = image.astype(np.float32)
    
    # 1. White Balance (Warmth)
    # Apply in RGB/BGR space.
    # Warmth factor > 1.0 => Warm (Boost Red, Reduce Blue)
    # Warmth factor < 1.0 => Cool (Boost Blue, Reduce Red)
    warn_min, warn_max = _choose_range(params.warmth_range)
    warmth = random.uniform(warn_min, warn_max)
    
    if abs(warmth - 1.0) > 0.01:
        # BGR order: Blue is channel 0, Red is channel 2
        # If warmth=1.2: Red * 1.1, Blue * 0.9 (approx)
        # To keep overall brightness somewhat constant, we balance cuts/boosts
        shift = warmth - 1.0 # e.g. +0.2 or -0.2
        
        # Red scale: 1 + shift/2 (e.g. 1.1)
        # Blue scale: 1 - shift/2 (e.g. 0.9)
        r_scale = 1.0 + (shift * 0.5)
        b_scale = 1.0 - (shift * 0.5)
        
        # Channel 0: Blue
        output[:, :, 0] *= b_scale
        # Channel 2: Red
        output[:, :, 2] *= r_scale
        
        np.clip(output, 0, 255, out=output)

    # 2. Brightness & Contrast
    # Brightness: Multiplier or Additive? Standard ColorJitter is often additive or multiplicative.
    # Contrast: Scale around mean.
    
    bri_min, bri_max = _choose_range(params.brightness_range)
    brightness_factor = random.uniform(bri_min, bri_max)
    
    con_min, con_max = _choose_range(params.contrast_range)
    contrast_factor = random.uniform(con_min, con_max)
    
    if abs(brightness_factor - 1.0) > 0.01 or abs(contrast_factor - 1.0) > 0.01:
        # Apply standard linear transform: dest = alpha * src + beta
        # Here we do it as: dest = (src - 128) * contrast + 128 * brightness
        # Or simply: dest = src * brightness_factor
        
        # Implementation similar to torchvision ColorJitter:
        # Brightness
        output *= brightness_factor
        
        # Contrast
        # Mean of the whole image (or per channel?) usually just 128 or gray mean. 
        # Torchvision uses gray conversion mean. OpenCV simple approach:
        gray_mean = np.mean(cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_BGR2GRAY))
        output = (output - gray_mean) * contrast_factor + gray_mean
        
        np.clip(output, 0, 255, out=output)

    # 3. Hsv Adjustments (Saturation & Hue)
    sat_min, sat_max = _choose_range(params.saturation_range)
    saturation_factor = random.uniform(sat_min, sat_max)
    
    hue_min, hue_max = _choose_range(params.hue_range)
    hue_factor = random.uniform(hue_min, hue_max)
    
    if abs(saturation_factor - 1.0) > 0.01 or abs(hue_factor) > 0.001:
        # Convert to HSV. Note: OpenCV HSV H is 0-179, S,V are 0-255
        hsv = cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Saturation (Channel 1)
        hsv[:, :, 1] *= saturation_factor
        
        # Hue (Channel 0) - additive shift
        # Hue wrap around 180
        h5_shift = hue_factor * 180.0
        hsv[:, :, 0] += h5_shift
        # Handle wrap
        hsv[:, :, 0] = np.mod(hsv[:, :, 0], 180.0)
        
        hsv = np.clip(hsv, 0, 255)
        output = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    return np.clip(output, 0, 255).astype(np.uint8)
