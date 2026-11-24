"""Runnable VA-40 semantic check using the supplied stairs image."""

from pathlib import Path

import cv2

from server.common.embeddings import ClipTextEncoder
from server.ingest.embeddings import ClipImageEncoder

STAIRS_IMAGE = Path("data/spike_frames/stairs_01/frame_0002s.jpg")


def main() -> None:
    frame = cv2.imread(str(STAIRS_IMAGE))
    if frame is None:
        raise FileNotFoundError(STAIRS_IMAGE)

    image_embedding = ClipImageEncoder().encode(frame, {})
    text_encoder = ClipTextEncoder()
    stairs_score = float(image_embedding @ text_encoder.encode("stairs"))
    arabic_stairs_score = float(image_embedding @ text_encoder.encode("درج"))
    car_score = float(image_embedding @ text_encoder.encode("car"))
    print(
        f"stairs={stairs_score:.4f}, arabic_stairs={arabic_stairs_score:.4f}, car={car_score:.4f}"
    )
    assert stairs_score > car_score
    assert arabic_stairs_score > car_score


if __name__ == "__main__":
    main()
