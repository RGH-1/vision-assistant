"""VA-70 checks: ranking and near-duplicate removal on made-up candidates (design §6.5.3)."""

import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from server.assistant.retrieve import CONFIG, rank, scaled
from server.common.schemas import QuestionEvidenceParts

ASKED_AT = datetime(2026, 11, 15, 12, 0, tzinfo=UTC)


def candidate(
    frame_id: int,
    seconds_ago: float,
    dhash: int,
    clip: float = 0.2,
    text: float = 0.0,
    obj: float = 0.0,
    sharpness: float = 100.0,
) -> dict[str, Any]:
    """One row as M2's hybrid SQL returns it."""
    return {
        "id": frame_id,
        "captured_at": ASKED_AT - timedelta(seconds=seconds_ago),
        "dhash": dhash,
        "sharpness": sharpness,
        "s_clip": clip,
        "s_text": text,
        "s_obj": obj,
        "seconds_ago": Decimal(seconds_ago),  # what Postgres's EXTRACT returns
    }


def test_each_part_is_min_max_scaled_then_weighted_for_the_intent() -> None:
    rows = [
        candidate(1, seconds_ago=60, dhash=0, clip=0.30, text=0.0, obj=0.9, sharpness=200),
        candidate(2, seconds_ago=0, dhash=-1, clip=0.15, text=0.6, obj=0.0, sharpness=100),
    ]

    [first, second] = rank(rows, "FIND_OBJECT")

    assert first.frame_id == 1
    assert first.parts == QuestionEvidenceParts(clip=1, text=0, obj=1, rec=0, q=1)
    assert second.parts == QuestionEvidenceParts(clip=0, text=1, obj=0, rec=1, q=0)
    weights = CONFIG.weights["FIND_OBJECT"]
    assert first.score == pytest.approx(weights.clip + weights.obj + weights.q)
    assert second.score == pytest.approx(weights.text + weights.rec)
    assert first.seconds_ago == 60


def test_the_intent_decides_which_frame_wins() -> None:
    rows = [
        candidate(1, seconds_ago=30, dhash=0, text=0.8),  # a sign was read here
        candidate(2, seconds_ago=90, dhash=-1, obj=0.9),  # the asked-about object was seen here
    ]

    assert rank(rows, "READ_TEXT_PAST")[0].frame_id == 1
    assert rank(rows, "FIND_OBJECT")[0].frame_id == 2


def test_near_duplicates_of_a_better_frame_are_dropped_and_three_are_kept() -> None:
    rows = [
        candidate(1, seconds_ago=10, dhash=0b0000, clip=0.35),
        candidate(2, seconds_ago=11, dhash=-1, clip=0.34),  # 1 s after the best one
        candidate(3, seconds_ago=40, dhash=0b1111_1111, clip=0.33),  # dHash 8 bits away
        candidate(4, seconds_ago=70, dhash=-1, clip=0.32),
        candidate(5, seconds_ago=100, dhash=-(2**40), clip=0.31),
        candidate(6, seconds_ago=130, dhash=2**20 - 1, clip=0.30),
    ]

    assert [frame.frame_id for frame in rank(rows, "MEMORY")] == [1, 4, 5]


def test_signed_dhash_values_are_compared_on_all_64_bits() -> None:
    # Postgres stores dHash as a signed BIGINT: -1 is all 64 bits set, so it is far from 0.
    rows = [
        candidate(1, seconds_ago=10, dhash=0, clip=0.35),
        candidate(2, seconds_ago=60, dhash=-1, clip=0.30),
    ]

    assert [frame.frame_id for frame in rank(rows, "MEMORY")] == [1, 2]


def test_a_lone_candidate_keeps_its_raw_scores() -> None:
    [only] = rank([candidate(1, seconds_ago=60, dhash=0, clip=0.25, obj=0.9)], "FIND_OBJECT")

    assert only.parts.model_dump() == pytest.approx(
        {"clip": 0.25, "text": 0, "obj": 0.9, "rec": math.exp(-1), "q": 1}
    )


def test_no_candidates_means_no_evidence() -> None:
    assert rank([], "MEMORY") == []


def test_scaled_spreads_values_over_0_to_1() -> None:
    assert scaled([0.15, 0.25, 0.35]) == pytest.approx([0, 0.5, 1])


def test_every_memory_intent_has_weights_that_add_up_to_1() -> None:
    assert set(CONFIG.weights) == {"READ_TEXT_PAST", "MEMORY", "FIND_OBJECT"}
    for weights in CONFIG.weights.values():
        assert sum(weights.model_dump().values()) == pytest.approx(1)
