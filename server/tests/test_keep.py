from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from server.ingest.keep import (
    FrameKeepCandidate,
    FrameKeepConfig,
    FrameKeepDecider,
    load_frame_keep_config,
)


@pytest.fixture
def config() -> FrameKeepConfig:
    return FrameKeepConfig(
        detection_confidence_min=0.35,
        edge_density_min=0.04,
        canny_low=100,
        canny_high=200,
        window_seconds=1,
    )


def flat_image() -> np.ndarray:
    return np.zeros((64, 64, 3), dtype=np.uint8)


def checkerboard() -> np.ndarray:
    rows, columns = np.indices((64, 64))
    image = ((rows // 4 + columns // 4) % 2 * 255).astype(np.uint8)
    return np.repeat(image[..., None], 3, axis=2)


def candidate(
    captured_at: float,
    sharpness: float,
    *,
    image: np.ndarray | None = None,
    detections: list[dict[str, float | str]] | None = None,
    keep_reason: str = "novel",
) -> FrameKeepCandidate:
    return FrameKeepCandidate(
        image=flat_image() if image is None else image,
        captured_at=captured_at,
        sharpness=sharpness,
        detections=() if detections is None else detections,
        keep_reason=keep_reason,  # type: ignore[arg-type]
    )


def test_detection_or_sign_or_edges_make_a_candidate_informative(config: FrameKeepConfig) -> None:
    decider = FrameKeepDecider(config)
    detections = [{"label": "door", "confidence": 0.35}]
    sign = [{"label": "exit sign", "confidence": 0.01}]

    decider.consider(candidate(0, 10, detections=detections), 0)
    emitted = decider.consider(candidate(1, 20, detections=sign), 1)
    decider.consider(candidate(2, 30, image=checkerboard()), 2)
    edge_frame = decider.flush()

    assert emitted[0].candidate.detections == detections
    assert emitted[0].candidate.keep_reason == "novel"
    assert edge_frame[0].edge_density >= config.edge_density_min


def test_flat_frame_without_a_detection_is_not_informative(config: FrameKeepConfig) -> None:
    decider = FrameKeepDecider(config)

    assert decider.consider(candidate(0, 10), 0) == ()
    assert decider.flush() == ()


def test_only_the_sharpest_in_a_one_second_window_is_kept(config: FrameKeepConfig) -> None:
    decider = FrameKeepDecider(config)
    detection = [{"label": "door", "confidence": 0.9}]
    decider.consider(candidate(0.0, 10, detections=detection), 0.0)
    decider.consider(candidate(0.2, 30, detections=detection, keep_reason="new_objects"), 0.2)
    decider.consider(candidate(0.8, 20, detections=detection), 0.8)

    emitted = decider.consider(candidate(1.0, 15, detections=detection), 1.0)

    assert len(emitted) == 1
    assert emitted[0].candidate.sharpness == 30
    assert emitted[0].candidate.keep_reason == "new_objects"


def test_counts_sampled_and_kept_frames_with_a_ten_percent_ratio(config: FrameKeepConfig) -> None:
    decider = FrameKeepDecider(config)
    detection = [{"label": "door", "confidence": 0.9}]

    for tick in range(100):
        timestamp = tick / 10
        passing = candidate(timestamp, 10, detections=detection) if tick % 10 == 0 else None
        decider.consider(passing, timestamp)
    decider.flush()

    assert decider.stats.sampled_frames == 100
    assert decider.stats.kept_frames == 10
    assert decider.stats.kept_ratio == pytest.approx(0.10)


def test_invalid_config_and_out_of_order_samples_are_rejected(config: FrameKeepConfig) -> None:
    with pytest.raises(ValueError, match="confidence"):
        FrameKeepDecider(replace(config, detection_confidence_min=0))
    decider = FrameKeepDecider(config)
    decider.consider(None, 2)
    with pytest.raises(ValueError, match="ordered"):
        decider.consider(None, 1)


def test_config_matches_the_design_defaults() -> None:
    config = load_frame_keep_config()

    assert config == FrameKeepConfig(0.35, 0.04, 100, 200, 1)


def test_heartbeat_bypasses_informative_filter(
    config: FrameKeepConfig,
) -> None:
    decider = FrameKeepDecider(config)

    decider.consider(
        candidate(
            0.0,
            10,
            keep_reason="heartbeat",
        ),
        0.0,
    )

    emitted = decider.consider(None, 1.0)

    assert len(emitted) == 1
    assert emitted[0].candidate.keep_reason == "heartbeat"
    assert emitted[0].edge_density == 0.0
    assert decider.stats.kept_frames == 1
    assert decider.stats.sampled_frames == 2