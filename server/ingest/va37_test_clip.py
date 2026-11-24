
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer


IMAGE_PATHS = [
    Path("data/spike_frames/walk_01/frame_0002s.jpg"),
    Path("data/spike_frames/stairs_01/frame_0002s.jpg"),
    Path("data/ocr_samples/push_sign.jpg"),
]

OUTPUT_DIR = Path("data/spike_results/clip")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    for path in IMAGE_PATHS:
        if not path.is_file():
            raise FileNotFoundError(path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("GPU:", torch.cuda.get_device_name(0))
    print("Loading CLIP image model...")

    model = SentenceTransformer(
        "sentence-transformers/clip-ViT-B-32",
        device="cuda",
    )

    images = []

    for path in IMAGE_PATHS:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))

    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()

    embeddings = model.encode(
        images,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    elapsed_ms = (perf_counter() - start) * 1000

    np.save(OUTPUT_DIR / "image_embeddings.npy", embeddings)

    print("\nEmbedding shape:", embeddings.shape)
    print(f"Total processing time: {elapsed_ms:.2f} ms")
    print(f"Time per image: {elapsed_ms / len(images):.2f} ms")

    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"Peak PyTorch GPU memory: {peak_mb:.2f} MiB")

    for path, embedding in zip(IMAGE_PATHS, embeddings):
        print(f"{path.name}: {len(embedding)} dimensions")


if __name__ == "__main__":
    main()
