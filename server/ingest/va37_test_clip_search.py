
from pathlib import Path
from time import perf_counter

import numpy as np
from sentence_transformers import SentenceTransformer


EMBEDDINGS_PATH = Path(
    "data/spike_results/clip/image_embeddings.npy"
)

IMAGE_NAMES = [
    "Corridor with doors",
    "Wooden stairs",
    "PUSH sign",
]

QUERIES = [
    "A corridor with wooden doors",
    "A wooden staircase",
    "A sign saying PUSH",
    "ممر فيه أبواب خشبية",
    "درج خشبي",
    "لافتة مكتوب عليها ادفع",
]


def main() -> None:
    image_embeddings = np.load(
        EMBEDDINGS_PATH,
        allow_pickle=False,
    )

    print("Loading multilingual CLIP text model...")

    model = SentenceTransformer(
        "sentence-transformers/clip-ViT-B-32-multilingual-v1",
        device="cpu",
    )

    start = perf_counter()

    text_embeddings = model.encode(
        QUERIES,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    elapsed_ms = (perf_counter() - start) * 1000

    print("Image embeddings:", image_embeddings.shape)
    print("Text embeddings:", text_embeddings.shape)

    if image_embeddings.shape != (3, 512):
        raise ValueError("Unexpected image embedding shape")

    if text_embeddings.shape != (len(QUERIES), 512):
        raise ValueError("Unexpected text embedding shape")

    # Cosine similarity because both vectors are normalized.
    scores = text_embeddings @ image_embeddings.T

    for query, query_scores in zip(QUERIES, scores):
        print(f"\nQuestion: {query}")

        ranked_indices = np.argsort(query_scores)[::-1]

        for index in ranked_indices:
            print(
                f"  {IMAGE_NAMES[index]}: "
                f"{query_scores[index]:.3f}"
            )

        best_index = ranked_indices[0]
        print(f"Best match: {IMAGE_NAMES[best_index]}")

    print(f"\nTotal text encoding time: {elapsed_ms:.2f} ms")


if __name__ == "__main__":
    main()
