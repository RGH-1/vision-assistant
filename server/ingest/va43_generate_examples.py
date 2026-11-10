"""Generate the five real dark-frame before/after pairs required by VA-43."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from server.ingest.enhance import FrameEnhancer, load_frame_enhancement_config
from server.settings import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create VA-43 slide evidence from a recorded video"
    )
    parser.add_argument("--video", type=Path, default=ROOT / "data" / "videos" / "arabic_sign.mp4")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "slides" / "enhancement")
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()
    if args.count <= 0:
        raise ValueError("count must be positive")

    candidates = _dark_candidates(args.video, args.count)
    if len(candidates) < args.count:
        raise RuntimeError(f"only found {len(candidates)} dark frames in {args.video}")

    args.output.mkdir(parents=True, exist_ok=True)
    enhancer = FrameEnhancer(load_frame_enhancement_config())
    rows = []
    for number, (timestamp_s, brightness, frame) in enumerate(candidates, start=1):
        result = enhancer.enhance(frame, brightness=brightness)
        before_name = f"dark_{number:02}_before.jpg"
        after_name = f"dark_{number:02}_after.jpg"
        if not cv2.imwrite(str(args.output / before_name), frame):
            raise RuntimeError(f"could not write {before_name}")
        if not cv2.imwrite(str(args.output / after_name), result.image):
            raise RuntimeError(f"could not write {after_name}")
        rows.append((number, timestamp_s, brightness, before_name, after_name, result.enhance_ops))

    _write_readme(args.output / "README.md", args.video, rows)


def _dark_candidates(video: Path, count: int) -> list[tuple[float, float, np.ndarray]]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(video)
    fps = capture.get(cv2.CAP_PROP_FPS) or 1.0
    interval = max(round(fps), 1)
    candidates: list[tuple[float, float, np.ndarray]] = []
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % interval == 0:
                brightness = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
                if brightness < 60:
                    candidates.append((index / fps, brightness, frame.copy()))
            index += 1
    finally:
        capture.release()
    return sorted(candidates, key=lambda candidate: candidate[1])[:count]


def _write_readme(
    path: Path,
    video: Path,
    rows: list[tuple[int, float, float, str, str, tuple[str, ...]]],
) -> None:
    lines = [
        "# VA-43 image enhancement evidence",
        "",
        f"Source: `{video.as_posix()}`. Each pair is a real frame with mean brightness below 60.",
        "",
        "| Frame | Time (s) | Mean brightness | Before | After | Operations |",
        "|---|---:|---:|---|---|---|",
    ]
    for number, timestamp_s, brightness, before_name, after_name, operations in rows:
        lines.append(
            f"| {number} | {timestamp_s:.1f} | {brightness:.1f} | "
            f"![before]({before_name}) | ![after]({after_name}) | {', '.join(operations)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
