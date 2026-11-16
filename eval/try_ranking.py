"""VA-70 playground: watch rank() score six made-up frames for each memory intent.

No database, models or keys: the rows are what M2's hybrid SQL would return.
Run from the repo root:  uv run python -m eval.try_ranking
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from server.assistant.retrieve import CONFIG, is_near_duplicate, rank

NOW = datetime.now(UTC)

# name: (seconds_ago, dhash, s_clip, s_text, s_obj, sharpness)
FRAMES = {
    "keys on the table": (50, 0b0000_0001, 0.29, 0.0, 0.82, 180),
    "same table, 1 s later": (51, 0x0F0F_0F0F, 0.29, 0.0, 0.80, 120),
    "same table, 2 min ago": (120, 0b0000_0111, 0.28, 0.0, 0.60, 90),
    "keys in the hallway": (240, 0x7F00_FF00_0000_0000, 0.27, 0.0, 0.55, 150),
    "EXIT sign": (20, -(2**40), 0.21, 0.70, 0.0, 200),
    "empty corridor": (5, 0x00FF_00FF_00FF_00FF, 0.22, 0.0, 0.0, 160),
}


def rows() -> list[dict]:
    return [
        {
            "id": i,
            "captured_at": NOW - timedelta(seconds=ago),
            "dhash": dhash,
            "s_clip": clip,
            "s_text": text,
            "s_obj": obj,
            "sharpness": sharp,
            "seconds_ago": Decimal(ago),
        }
        for i, (ago, dhash, clip, text, obj, sharp) in enumerate(FRAMES.values())
    ]


def show(intent: str) -> None:
    names = list(FRAMES)
    weights = CONFIG.weights[intent]
    # Score every frame (no duplicate removal) to see the full order, then the real result.
    everyone = rank(
        rows(),
        intent,
        CONFIG.model_copy(update={"keep": 99, "duplicate_s": 0, "duplicate_dhash_max": -1}),
    )
    kept = rank(rows(), intent)
    kept_ids = [frame.frame_id for frame in kept]

    print(f"\n=== {intent}   weights {weights.model_dump()}")
    print(
        f"{'frame':24} {'clip':>5} {'text':>5} {'obj':>5} {'rec':>5} {'q':>5}  {'score':>5}  result"
    )
    for frame in everyone:
        p = frame.parts
        if frame.frame_id in kept_ids:
            result = f"KEPT #{kept_ids.index(frame.frame_id) + 1}"
        else:
            better = next((k for k in kept if is_near_duplicate(frame, k, CONFIG)), None)
            result = (
                f"dropped: near-duplicate of '{names[better.frame_id]}'"
                if better
                else (f"dropped: only the top {CONFIG.keep} are kept")
            )
        print(
            f"{names[frame.frame_id]:24} {p.clip:5.2f} {p.text:5.2f} {p.obj:5.2f} {p.rec:5.2f}"
            f" {p.q:5.2f}  {frame.score:5.2f}  {result}"
        )


for intent in CONFIG.weights:
    show(intent)
