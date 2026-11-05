
from pathlib import Path

import cv2


VIDEOS = [
    "walk_01.mp4",
    "stairs_01.mp4",
    "arabic_sign.mp4",
]

INTERVAL_SECONDS = 2


def extract_frames(filename: str) -> None:
    video_path = Path("data/videos") / filename
    output_dir = Path("data/spike_frames") / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid FPS: {video_path}")

    duration = total_frames / fps
    saved = 0

    print(f"\nVideo: {filename}")
    print(f"FPS: {fps:.2f}")
    print(f"Duration: {duration:.2f} seconds")

    for second in range(0, int(duration) + 1, INTERVAL_SECONDS):
        frame_number = int(second * fps)
        if frame_number >= total_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = cap.read()

        if not success:
            continue

        height, width = frame.shape[:2]

        if max(width, height) > 1280:
            scale = 1280 / max(width, height)
            frame = cv2.resize(
                frame,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )

        output_path = output_dir / f"frame_{second:04d}s.jpg"

        if not cv2.imwrite(str(output_path), frame):
            raise RuntimeError(f"Cannot save: {output_path}")

        saved += 1

    cap.release()
    print(f"Frames saved: {saved}")
    print(f"Folder: {output_dir}")


if __name__ == "__main__":
    for video in VIDEOS:
        extract_frames(video)
