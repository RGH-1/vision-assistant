import cv2
import numpy as np
import pytest

from server.ingest.selection import FrameSelectionConfig, FrameSelector, load_frame_selection_config
from dataclasses import replace

@pytest.fixture
def config() -> FrameSelectionConfig:
    return FrameSelectionConfig(
        analysis_width=640,
        sharpness_min=60,
        sharpness_median_factor=0.6,
        sharpness_soft_ratio=0.6,
        sharpness_history_size=50,
        brightness_dark=60,
        brightness_bright=220,
        contrast_min=25,
        unsharp_sigma=1.0,
        unsharp_amount=1.0,
    )


def checkerboard(low: int = 0, high: int = 255) -> np.ndarray:
    rows, columns = np.indices((128, 128))
    pattern = (rows // 8 + columns // 8) % 2
    image = np.where(pattern[..., None] == 0, low, high).astype(np.uint8)
    return np.repeat(image, 3, axis=2)


def test_sharp_sample_is_accepted_without_changes(config: FrameSelectionConfig) -> None:
    result = FrameSelector(config).select(checkerboard())

    assert result.accepted
    assert not result.unsharp_applied
    assert not result.needs_enhancement
    assert result.metrics.sharpness >= result.metrics.adaptive_sharpness_threshold


def test_blurred_sample_is_dropped(config: FrameSelectionConfig) -> None:
    sharp_edge = np.zeros((128, 128, 3), dtype=np.uint8)
    sharp_edge[:, 64:] = 255
    blurred = cv2.GaussianBlur(sharp_edge, (0, 0), 5)
    result = FrameSelector(config).select(blurred)

    assert not result.accepted
    assert result.drop_reason == "blurred"


def test_dark_sample_is_sent_to_enhancement(config: FrameSelectionConfig) -> None:
    result = FrameSelector(config).select(checkerboard(10, 50))

    assert result.accepted
    assert result.needs_enhancement
    assert result.enhancement_reasons == ("dark", "low_contrast")


def test_low_contrast_sample_is_sent_to_enhancement(config: FrameSelectionConfig) -> None:
    result = FrameSelector(config).select(checkerboard(110, 130))

    assert result.accepted
    assert result.needs_enhancement
    assert result.enhancement_reasons == ("low_contrast",)


def test_overexposed_sample_is_dropped(config: FrameSelectionConfig) -> None:
    result = FrameSelector(config).select(np.full((128, 128, 3), 230, dtype=np.uint8))

    assert not result.accepted
    assert result.drop_reason == "overexposed"


def test_soft_band_applies_unsharp_mask(config: FrameSelectionConfig) -> None:
    selector = FrameSelector(config)
    selector.select(checkerboard())
    softened = cv2.GaussianBlur(checkerboard(), (0, 0), 0.5)
    result = selector.select(softened)

    assert result.accepted
    assert result.unsharp_applied
    assert not np.array_equal(result.image, softened)


def test_adaptive_threshold_uses_previous_fifty_frames(config: FrameSelectionConfig) -> None:
    selector = FrameSelector(config)
    for _ in range(55):
        selector.select(checkerboard())

    result = selector.select(checkerboard())

    assert result.metrics.adaptive_sharpness_threshold > config.sharpness_min


def test_config_is_loaded_from_yaml() -> None:
    config = load_frame_selection_config()

    assert config.analysis_width == 640
    assert config.sharpness_min == 60


@pytest.mark.parametrize("field", ["analysis_width", "sharpness_history_size"])
def test_invalid_size_is_rejected(config: FrameSelectionConfig, field: str) -> None:
    invalid = config.__class__(**{**config.__dict__, field: 0})

    with pytest.raises(ValueError):
        FrameSelector(invalid)

def test_severe_blur_threshold_is_configurable() -> None:
    config = load_frame_selection_config()

    assert config.severe_blur_ratio == 0.01

    rows, cols = np.indices((128, 128))
    pattern = (rows // 8 + cols // 8) % 2

    dark = np.repeat(
        np.where(pattern[..., None] == 0, 10, 40).astype(np.uint8),
        3,
        axis=2,
    )

    blurred = cv2.GaussianBlur(dark, (0, 0), 8)

    strict_result = FrameSelector(config).select(blurred)

    permissive_config = replace(config, severe_blur_ratio=0.00001)
    permissive_result = FrameSelector(permissive_config).select(blurred)

    assert not strict_result.accepted
    assert strict_result.drop_reason == "blurred"

    assert permissive_result.accepted
    assert permissive_result.needs_enhancement