
from pathlib import Path

import cv2
import torch
from ultralytics import YOLOE


IMAGE_PATH = Path("data/spike_frames/stairs_01/frame_0002s.jpg")
OUTPUT_DIR = Path("data/spike_results/yoloe")

CLASSES = [
    "stairs",
    "staircase",
    "steps",
    "wooden stairs",
    "wooden staircase",
    "wooden steps",
    "flight of stairs",
]

CONFIDENCE = 0.30


def main() -> None:
    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(IMAGE_PATH)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("GPU:", torch.cuda.get_device_name(0))
    print("Loading YOLOE model...")

    model = YOLOE("yoloe-26s-seg.pt")
    model.set_classes(CLASSES)

    print("Running object detection...")

    results = model.predict(
        source=str(IMAGE_PATH),
        conf=CONFIDENCE,
        imgsz=640,
        device=0,
        half=True,
        verbose=False,
    )

    result = results[0]

    print("\nDetected objects:")

    if len(result.boxes) == 0:
        print(f"No objects found at confidence >= {CONFIDENCE:.2f}")

    for box in result.boxes:
        class_id = int(box.cls.item())
        label = result.names[class_id]
        confidence = float(box.conf.item())

        print(f"{label}: {confidence:.2f}")

    output_path = OUTPUT_DIR / "stairs_final.jpg"

    annotated_image = result.plot()
    if not cv2.imwrite(str(output_path), annotated_image):
        raise RuntimeError("Could not save annotated image")

    print("\nSaved result to:", output_path)


if __name__ == "__main__":
    main()
