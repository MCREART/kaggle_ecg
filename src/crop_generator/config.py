from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Tuple

TransformMode = Literal["none", "affine", "perspective"]


@dataclass(slots=True)
class TransformParams:
    """Parameters controlling the geometric transform."""

    mode: TransformMode = "perspective"
    max_rotate: float = 8.0
    max_shift: float = 0.08
    max_scale: float = 0.10
    perspective_jitter: float = 0.12


@dataclass(slots=True)
class CropParams:
    """Parameters controlling random crop generation."""

    count: int = 10
    min_size: int = 256
    max_size: int = 768
    out_size: int = 512
    allow_out_of_bounds: bool = False
    edge_focus_prob: float = 0.35
    edge_focus_span_ratio: float = 0.2  # fraction of selectable span treated as edge zone
    corner_focus_prob: float = 0.25
    random_border_fill: bool = True
    border_noise_sigma: float = 8.0
    corner_override: Tuple[bool, bool] | None = None
    min_mask_coverage: float = 0.0
    coverage_dilation_kernel: int = 15
    coverage_max_attempts: int = 20
    min_horizontal_density: float = 0.2
    min_horizontal_rows: int = 2
    uniform_std_threshold: float = 1.0
    uniform_retry_attempts: int = 20
    uniform_range_threshold: float = 12.0
    global_retry_attempts: int = 3
    write_metadata: bool = False
    mask_guided_prob: float = 0.0
    mask_guided_min_area_ratio: float = 0.0005
    mask_guided_expand_ratio: float = 0.3
    mask_guided_jitter_ratio: float = 0.15


@dataclass(slots=True)
class WrinkleParams:
    """Parameters controlling wrinkle-style displacement augmentation."""

    enabled: bool = False
    count_range: Tuple[int, int] = (1, 3)
    amplitude_range: Tuple[float, float] = (0.01, 0.03)  # fraction of min(height, width)
    sigma_range: Tuple[float, float] = (0.02, 0.08)  # fraction of min(height, width)
    wavelength_range: Tuple[float, float] = (0.08, 0.18)  # fraction of min(height, width)


@dataclass(slots=True)
class BlurParams:
    """Parameters controlling post-wrinkle blur augmentation."""

    enabled: bool = False
    kernel_range: Tuple[int, int] = (3, 7)  # inclusive, odd values enforced at runtime
    sigma_range: Tuple[float, float] = (0.3, 1.5)


@dataclass(slots=True)
class GrayParams:
    """Parameters controlling grayscale conversion."""

    enabled: bool = False
    preserve_channels: bool = True  # when True, replicate grayscale to 3 channels


@dataclass(slots=True)
class NoiseParams:
    """Parameters controlling additive Gaussian noise."""

    enabled: bool = False
    sigma_range: Tuple[float, float] = (3.0, 12.0)  # noise std in intensity units


@dataclass(slots=True)
class OcclusionParams:
    """Parameters for random rectangular occlusions."""

    enabled: bool = False
    count_range: Tuple[int, int] = (1, 3)
    size_range: Tuple[float, float] = (0.12, 0.25)  # fraction of shorter side
    intensity_range: Tuple[int, int] = (0, 80)  # occluder color intensity


@dataclass(slots=True)
class StainParams:
    """Parameters for blotchy stain augmentation (dark smudges)."""

    enabled: bool = False
    count_range: Tuple[int, int] = (1, 3)
    size_range: Tuple[float, float] = (0.05, 0.15)  # fraction of shorter side
    intensity_range: Tuple[float, float] = (35.0, 120.0)  # amount to darken (0-255)
    softness_range: Tuple[float, float] = (0.6, 1.2)  # controls Gaussian softness
    tint_color: Tuple[int, int, int] | None = None  # RGB tint applied when provided
    tint_strength_range: Tuple[float, float] = (0.2, 0.6)  # blend factor for tint
    texture_strength_range: Tuple[float, float] = (0.3, 0.8)
    texture_scale_range: Tuple[float, float] = (0.05, 0.25)  # relative to radius


@dataclass(slots=True)
class MoireParams:
    """Parameters for moiré pattern simulation."""

    enabled: bool = False
    period_range: Tuple[float, float] = (5.0, 16.0)  # base stripe period in pixels
    beat_ratio_range: Tuple[float, float] = (1.02, 1.15)  # second wave frequency ratio
    angle_range: Tuple[float, float] = (-25.0, 25.0)  # degrees
    amplitude_range: Tuple[float, float] = (0.12, 0.28)  # modulation strength
    phase_range: Tuple[float, float] = (0.0, 2.0 * 3.1415926)
    warp_strength_range: Tuple[float, float] = (0.0, 0.2)
    warp_frequency_range: Tuple[float, float] = (0.3, 1.2)
    texture_strength_range: Tuple[float, float] = (0.05, 0.22)
    texture_sigma_range: Tuple[float, float] = (0.04, 0.14)  # fraction of short side
    blend_range: Tuple[float, float] = (0.7, 0.95)
    contrast_gain_range: Tuple[float, float] = (1.4, 3.0)
    secondary_prob: float = 0.4
    secondary_angle_offset_range: Tuple[float, float] = (6.0, 18.0)
    secondary_amplitude_scale_range: Tuple[float, float] = (0.35, 0.75)
    color_shift_prob: float = 0.0  # probability to apply channel-wise modulation
    color_strength_range: Tuple[float, float] = (0.0, 0.0)
    tint_strength_range: Tuple[float, float] = (0.0, 0.0)  # additive tint intensity (0-1 scale)


@dataclass(slots=True)
class NegativeSampleParams:
    """Parameters for synthesising negative (no-mask) samples."""

    image_size: int = 512
    background_intensity_range: Tuple[int, int] = (170, 235)
    background_jitter: int = 12  # +- jitter per channel
    noise_sigma_range: Tuple[float, float] = (6.0, 18.0)
    line_count_range: Tuple[int, int] = (2, 6)
    line_thickness_range: Tuple[int, int] = (1, 4)
    shape_count_range: Tuple[int, int] = (1, 4)
    texture_probability: float = 0.65
    text_probability: float = 0.55
    text_count_range: Tuple[int, int] = (1, 3)
    blur_kernel_choices: Tuple[int, ...] = (0, 3, 5)  # 0 means skip blur
    pattern_probability: float = 0.6
    pattern_period_range: Tuple[float, float] = (12.0, 48.0)
    pattern_amplitude_range: Tuple[float, float] = (18.0, 45.0)
    pattern_blend_range: Tuple[float, float] = (0.25, 0.55)
    pattern_angle_jitter: Tuple[float, float] = (-35.0, 35.0)


@dataclass(slots=True)
class ColorAugParams:
    """Parameters for color jitter and white balance."""

    enabled: bool = False
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    saturation_range: Tuple[float, float] = (0.5, 1.5)
    hue_range: Tuple[float, float] = (-0.05, 0.05)
    # White balance temperature (warm/cool) simulated by channel scaling
    # > 1.0 means boost Red (warm), < 1.0 means boost Blue (cool)
    warmth_range: Tuple[float, float] = (0.8, 1.2)


@dataclass(slots=True)
class TextOverlayParams:
    """Parameters for adding random text distractors."""

    enabled: bool = False
    count_range: Tuple[int, int] = (1, 5)
    font_scale_range: Tuple[float, float] = (0.5, 1.5)
    thickness_range: Tuple[int, int] = (1, 2)
    color_jitter: int = 50
    opacity_range: Tuple[float, float] = (0.7, 1.0) # Text isn't always perfectly black
    clear_mask: bool = True # If True, text pixels in mask become Background (0)


@dataclass(slots=True)
class PipelineConfig:
    """Top-level configuration for the crop generation pipeline."""

    image_path: Path
    mask_path: Path
    output_dir: Path
    coverage_mask_path: Path | None = None
    seed: Optional[int] = None
    transform: TransformParams = field(default_factory=TransformParams)
    crop: CropParams = field(default_factory=CropParams)
    wrinkle: WrinkleParams = field(default_factory=WrinkleParams)
    blur: BlurParams = field(default_factory=BlurParams)
    gray: GrayParams = field(default_factory=GrayParams)
    noise: NoiseParams = field(default_factory=NoiseParams)
    occlusion: OcclusionParams = field(default_factory=OcclusionParams)
    stain: StainParams = field(default_factory=StainParams)
    moire: MoireParams = field(default_factory=MoireParams)
    color_aug: ColorAugParams = field(default_factory=ColorAugParams)
    text_overlay: TextOverlayParams = field(default_factory=TextOverlayParams)
