
from pathlib import Path
from time import perf_counter

import easyocr
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from ultralytics import YOLOE


CORRIDOR = Path("data/spike_frames/walk_01/frame_0002s.jpg")
SIGN = Path("data/ocr_samples/push_sign.jpg")

MIB = 1024 ** 2


def gpu_memory(stage: str) -> None:
    free, total = torch.cuda.mem_get_info()

    allocated = torch.cuda.memory_allocated() / MIB
    reserved = torch.cuda.memory_reserved() / MIB
    used_total = (total - free) / MIB

    print(f"\n--- GPU memory: {stage} ---")
    print(f"PyTorch allocated: {allocated:.2f} MiB")
    print(f"PyTorch reserved: {reserved:.2f} MiB")
    print(f"Total GPU memory in use: {used_total:.2f} MiB")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    for path in [CORRIDOR, SIGN]:
        if not path.is_file():
            raise FileNotFoundError(path)

    print("GPU:", torch.cuda.get_device_name(0))

    gpu_memory("Before loading models")

    # 1. YOLOE
    print("\nLoading YOLOE...")

    yoloe = YOLOE("yoloe-26s-seg.pt")
    yoloe.set_classes([
        "door",
        "wooden door",
        "open door",
        "closed door",
        "stairs",
        "wooden stairs",
        "staircase",
    ])

    start = perf_counter()

    detections = yoloe.predict(
        source=str(CORRIDOR),
        conf=0.30,
        imgsz=640,
        device=0,
        half=True,
        verbose=False,
    )

    torch.cuda.synchronize()

    print(f"YOLOE time: {(perf_counter() - start) * 1000:.2f} ms")
    print(f"YOLOE detections: {len(detections[0].boxes)}")

    gpu_memory("After YOLOE")

    # 2. EasyOCR
    print("\nLoading EasyOCR...")

    reader = easyocr.Reader(["ar", "en"], gpu=True)

    start = perf_counter()

    ocr_results = reader.readtext(str(SIGN))

    torch.cuda.synchronize()

    print(f"EasyOCR time: {(perf_counter() - start) * 1000:.2f} ms")

    for _, text, confidence in ocr_results:
        print(f"OCR: {text} ({confidence:.2f})")

    gpu_memory("After YOLOE + EasyOCR")

    # 3. CLIP
    print("\nLoading CLIP...")

    clip_model = SentenceTransformer(
        "sentence-transformers/clip-ViT-B-32",
        device="cuda",
    )

    with Image.open(CORRIDOR) as image:
        clip_image = image.convert("RGB")

    start = perf_counter()

    embedding = clip_model.encode(
        clip_image,
        normalize_embeddings=True,
    )

    torch.cuda.synchronize()

    print(f"CLIP time: {(perf_counter() - start) * 1000:.2f} ms")
    print(f"Embedding dimensions: {len(embedding)}")

    gpu_memory("All three models loaded")
    print("\n--- Repeated inference test ---")

    for i in range(5):
        # YOLOE
        start = perf_counter()

        yoloe.predict(
            source=str(CORRIDOR),
            conf=0.30,
            imgsz=640,
            device=0,
            half=True,
            verbose=False,
        )

        torch.cuda.synchronize()
        yoloe_ms = (perf_counter() - start) * 1000

        # EasyOCR
        start = perf_counter()

        reader.readtext(str(SIGN))

        torch.cuda.synchronize()
        ocr_ms = (perf_counter() - start) * 1000

        # CLIP
        start = perf_counter()

        clip_model.encode(
            clip_image,
            normalize_embeddings=True,
        )

        torch.cuda.synchronize()
        clip_ms = (perf_counter() - start) * 1000

        total_ms = yoloe_ms + ocr_ms + clip_ms

        print(
            f"Pass {i + 1}: "
            f"YOLOE={yoloe_ms:.1f} ms | "
            f"OCR={ocr_ms:.1f} ms | "
            f"CLIP={clip_ms:.1f} ms | "
            f"Total={total_ms:.1f} ms"
        )

    gpu_memory("After five repeated passes")


    print("\nPeak allocated GPU memory:",
          f"{torch.cuda.max_memory_allocated() / MIB:.2f} MiB")


if __name__ == "__main__":
    main()
