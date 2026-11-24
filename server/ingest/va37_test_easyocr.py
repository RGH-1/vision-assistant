
from pathlib import Path
from time import perf_counter

import easyocr
import torch


IMAGE_PATH = Path("data/ocr_samples/push_sign.jpg")

MIN_CONFIDENCE = 0.40


def main() -> None:
    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(IMAGE_PATH)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Loading EasyOCR (Arabic + English)...")

    start_load = perf_counter()

    reader = easyocr.Reader(["ar", "en"], gpu=True)

    load_time = perf_counter() - start_load

    print(f"Model loaded in {load_time:.2f} seconds")
    print("Reading image...")

    start_ocr = perf_counter()

    results = reader.readtext(str(IMAGE_PATH))

    ocr_time = perf_counter() - start_ocr

    print("\nAll detected text:")

    if not results:
        print("No text detected")

    for bbox, text, confidence in results:
        print(f"Text: {text}")
        print(f"Confidence: {confidence:.2f}")

        if confidence >= MIN_CONFIDENCE and len(text.strip()) >= 2:
            print("Accepted: Yes")
        else:
            print("Accepted: No")

        print("-" * 30)

    print(f"\nOCR processing time: {ocr_time * 1000:.2f} ms")

    gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"Peak PyTorch GPU memory: {gpu_memory:.2f} MB")


if __name__ == "__main__":
    main()
