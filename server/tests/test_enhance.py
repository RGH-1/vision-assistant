import cv2
import numpy as np
import pytest

from server.ingest.enhance import (
    FrameEnhancementConfig,
    FrameEnhancer,
    load_frame_enhancement_config,
)


@pytest.fixture
def enhancer() -> FrameEnhancer:
    return FrameEnhancer(
        FrameEnhancementConfig(
            dark_brightness=60,
            gamma_min=0.6,
            gamma_max=0.8,
            clahe_clip_limit=2.0,
            clahe_tile_size=8,
            unsharp_sigma=1.0,
            unsharp_amount=1.0,
        )
    )


def checkerboard(low: int, high: int) -> np.ndarray:
    rows, columns = np.indices((128, 128))
    pattern = (rows // 8 + columns // 8) % 2
    image = np.where(pattern[..., None] == 0, low, high).astype(np.uint8)
    return np.repeat(image, 3, axis=2)


def test_dark_frame_uses_adaptive_gamma_and_clahe(enhancer: FrameEnhancer) -> None:
    frame = checkerboard(10, 40)
    result = enhancer.enhance(frame, brightness=float(frame.mean()))

    assert result.enhanced
    assert result.enhance_ops == ("gamma", "clahe")
    assert result.original is not None
    assert result.image.mean() > frame.mean()
    assert np.array_equal(result.original, frame)


def test_low_contrast_frame_uses_clahe_only(enhancer: FrameEnhancer) -> None:
    frame = checkerboard(110, 130)
    result = enhancer.enhance(frame, brightness=float(frame.mean()), low_contrast=True)

    assert result.enhanced
    assert result.enhance_ops == ("clahe",)
    assert result.original is not None
    assert not np.array_equal(result.image, frame)


def test_slightly_soft_frame_uses_unsharp_mask(enhancer: FrameEnhancer) -> None:
    frame = cv2.GaussianBlur(checkerboard(0, 255), (0, 0), 1.5)
    result = enhancer.enhance(frame, brightness=float(frame.mean()), slightly_soft=True)

    assert result.enhance_ops == ("unsharp",)
    assert not np.array_equal(result.image, frame)


def test_strong_motion_blur_is_dropped_without_deblurring(enhancer: FrameEnhancer) -> None:
    frame = cv2.GaussianBlur(checkerboard(0, 255), (0, 0), 8)
    result = enhancer.enhance(
        frame,
        brightness=float(frame.mean()),
        strong_motion_blur=True,
    )

    assert result.dropped
    assert not result.enhanced
    assert result.original is None
    assert np.array_equal(result.image, frame)


def test_config_is_loaded_from_yaml() -> None:
    config = load_frame_enhancement_config()

    assert config.gamma_min == 0.6
    assert config.gamma_max == 0.8
    assert config.clahe_clip_limit == 2.0
    assert config.clahe_tile_size == 8


def test_invalid_gamma_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="gamma range"):
        FrameEnhancer(
            FrameEnhancementConfig(
                dark_brightness=60,
                gamma_min=0.9,
                gamma_max=0.8,
                clahe_clip_limit=2.0,
                clahe_tile_size=8,
                unsharp_sigma=1.0,
                unsharp_amount=1.0,
            )
        )
