from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from server.common.db import open_ingest_connection
from server.ingest.enhance import FrameEnhancer, load_frame_enhancement_config
from server.ingest.faceblur import FaceBlurResult
from server.ingest.selection import FrameSelector, load_frame_selection_config
from server.ingest.store import FrameStore, ProcessedFrame


class FakeFaceBlurrer:
    def blur(self, image: np.ndarray, _: dict[str, float] | None = None) -> FaceBlurResult:
        return FaceBlurResult(image.copy(), 0)


def test_dark_frame_selection_enhancement_and_storage(tmp_path: Path) -> None:
    rows, cols = np.indices((128, 128))
    pattern = (rows // 8 + cols // 8) % 2
    frame = np.repeat(
        np.where(pattern[..., None] == 0, 10, 40).astype(np.uint8),
        3,
        axis=2,
    )

    selection = FrameSelector(load_frame_selection_config()).select(frame)

    assert selection.accepted
    assert selection.needs_enhancement

    enhancement = FrameEnhancer(load_frame_enhancement_config()).enhance(
        selection.image,
        brightness=selection.metrics.brightness,
        low_contrast="low_contrast" in selection.enhancement_reasons,
    )

    assert enhancement.enhanced
    assert enhancement.enhance_ops == ("gamma", "clahe")
    assert enhancement.original is not None
    assert np.array_equal(enhancement.original, frame)
    assert enhancement.image.mean() > frame.mean()

    connection = open_ingest_connection()
    session_id = None

    try:
        session_id = connection.execute(
            """
            INSERT INTO sessions (app_lang, consent_at, consent_ver)
            VALUES ('en', now(), 'enhancement-integration-test')
            RETURNING id
            """
        ).fetchone()["id"]
        connection.commit()

        store = FrameStore(tmp_path, connection, FakeFaceBlurrer())
        frame_id = store.store(
            ProcessedFrame(
                session_id=session_id,
                captured_at=datetime.now(UTC),
                image=enhancement.image,
                original=enhancement.original,
                enhanced=enhancement.enhanced,
                enhance_ops=enhancement.enhance_ops,
                sharpness=selection.metrics.sharpness,
                brightness=selection.metrics.brightness,
                contrast=selection.metrics.contrast,
                edge_density=0.0,
                dhash=0,
                keep_reason="heartbeat",
                clip_emb=np.eye(1, 512, dtype=np.float32)[0],
                timings_ms={},
            )
        )

        row = connection.execute(
            "SELECT * FROM frames WHERE id = %s",
            (frame_id,),
        ).fetchone()

        assert row["enhanced"] is True
        assert row["enhance_ops"] == ["gamma", "clahe"]
        assert row["original_path"] is not None

        saved_image = cv2.imread(str(tmp_path / row["image_path"]))
        saved_original = cv2.imread(str(tmp_path / row["original_path"]))

        assert saved_image is not None
        assert saved_original is not None
        assert (tmp_path / row["thumb_path"]).is_file()
        assert saved_image.mean() > saved_original.mean()

    finally:
        connection.rollback()
        if session_id is not None:
            connection.execute(
                "DELETE FROM sessions WHERE id = %s",
                (session_id,),
            )
            connection.commit()
        connection.close()


def test_dark_frame_with_extreme_blur_is_rejected() -> None:
    rows, cols = np.indices((128, 128))
    pattern = (rows // 8 + cols // 8) % 2

    dark = np.repeat(
        np.where(pattern[..., None] == 0, 10, 40).astype(np.uint8),
        3,
        axis=2,
    )

    blurred = cv2.GaussianBlur(dark, (0, 0), 8)

    selector = FrameSelector(load_frame_selection_config())

    dark_result = selector.select(dark)
    blurred_result = selector.select(blurred)

    assert dark_result.accepted
    assert dark_result.needs_enhancement

    assert not blurred_result.accepted
    assert blurred_result.drop_reason == "blurred"
