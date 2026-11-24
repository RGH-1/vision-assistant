from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from server.ingest.novelty import (
    FrameNoveltyConfig,
    FrameNoveltySelector,
    dhash,
    hamming_distance,
    load_frame_novelty_config,
)


@pytest.fixture
def config() -> FrameNoveltyConfig:
    return FrameNoveltyConfig(new_distance=12, duplicate_distance=5, heartbeat_seconds=10)


def test_dhash_is_signed_64_bit_and_hamming_handles_the_sign_bit() -> None:
    increasing = np.tile(np.arange(9, dtype=np.uint8), (8, 1))
    decreasing = np.fliplr(increasing)

    first = dhash(increasing)
    second = dhash(decreasing)

    assert -(2**63) <= first < 2**63
    assert -(2**63) <= second < 2**63
    assert hamming_distance(first, second) == 64


def test_novel_and_duplicate_distances_follow_configuration(config: FrameNoveltyConfig) -> None:
    selector = FrameNoveltySelector(config)

    assert selector.select_hash(0, (), 0).keep_reason == "novel"
    selector.confirm_saved(0, (), 0)
    duplicate = selector.select_hash((1 << 5) - 1, (), 1)
    novel = selector.select_hash((1 << 12) - 1, (), 2)

    assert (duplicate.keep, duplicate.distance, duplicate.keep_reason) == (False, 5, None)
    assert (novel.keep, novel.distance, novel.keep_reason) == (True, 12, "novel")


def test_new_object_label_keeps_anotherwise_duplicate_frame(config: FrameNoveltyConfig) -> None:
    selector = FrameNoveltySelector(config)
    selector.select_hash(0, ("door",), 0)
    selector.confirm_saved(0, ("door",), 0)

    decision = selector.select_hash(0, ("door", "stairs"), 1)

    assert decision.keep
    assert decision.keep_reason == "new_objects"
    assert decision.distance == 0



def test_standing_still_for_one_minute_saves_six_heartbeat_frames(
    config: FrameNoveltyConfig,
) -> None:
    selector = FrameNoveltySelector(config)

    decisions = []

    for second in range(60):
        timestamp = float(second)

        decision = selector.select_hash(0, (), timestamp)
        decisions.append(decision)

        if decision.keep:
            selector.confirm_saved(
                decision.dhash,
                (),
                timestamp,
            )

    kept = [decision for decision in decisions if decision.keep]

    assert len(kept) == 6
    assert [decision.keep_reason for decision in kept] == [
        "novel",
        "heartbeat",
        "heartbeat",
        "heartbeat",
        "heartbeat",
        "heartbeat",
    ]



def test_invalid_configuration_is_rejected(config: FrameNoveltyConfig) -> None:
    with pytest.raises(ValueError, match="new_distance"):
        FrameNoveltySelector(replace(config, new_distance=5))
    with pytest.raises(ValueError, match="heartbeat"):
        FrameNoveltySelector(replace(config, heartbeat_seconds=0))


def test_config_matches_default_thresholds() -> None:
    config = load_frame_novelty_config()

    assert config == FrameNoveltyConfig(new_distance=12, duplicate_distance=5, heartbeat_seconds=10)
def test_rejected_novel_frames_do_not_prevent_heartbeat(
    config: FrameNoveltyConfig,
) -> None:
    selector = FrameNoveltySelector(config)

    first = selector.select_hash(0, (), 0.0)

    for tick in range(1, 50):
        selector.select_hash(
            (1 << 64) - 1 if tick % 2 else 0,
            (),
            tick / 5.0,
        )

    heartbeat = selector.select_hash(0, (), 10.0)

    assert first.keep_reason == "novel"
    assert heartbeat.keep
    assert heartbeat.keep_reason == "heartbeat"


def test_rejected_new_objects_do_not_prevent_heartbeat(
    config: FrameNoveltyConfig,
) -> None:
    selector = FrameNoveltySelector(config)

    first = selector.select_hash(0, ("door",), 0.0)
    assert first.keep

    selector.confirm_saved(first.dhash, ("door",), 0.0)

    for tick in range(1, 50):
        decision = selector.select_hash(
            0,
            ("door", "stairs"),
            tick / 5.0,
        )
        assert decision.keep_reason == "new_objects"

    heartbeat = selector.select_hash(
        0,
        ("door", "stairs"),
        10.0,
    )

    assert heartbeat.keep
    assert heartbeat.keep_reason == "heartbeat"
