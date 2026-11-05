
# VA-37 — YOLOE, EasyOCR and CLIP Spike Results

## Environment

- GPU: NVIDIA T1200 Laptop GPU (4 GB VRAM)
- Python: 3.12
- Ultralytics: 8.4.202
- Models: YOLOE-26s-seg, EasyOCR (Arabic + English),
  CLIP ViT-B/32 and multilingual CLIP text encoder

## 1. YOLOE

- Wooden door detected at confidence 0.31.
- Wooden stairs detected at confidence 0.46.
- Both detections exceed the required 0.30 threshold.
- Descriptive prompts performed better than generic labels.
- Results are from selected video frames, not an
  accuracy evaluation across entire videos.

## 2. EasyOCR

- Arabic PUSH sign text: إدفع, confidence 0.78.
- English PUSH sign text: PUSH, confidence 0.62.
- Both accepted at threshold 0.40.
- Another Arabic video frame was correctly recognized
  at confidence 0.24 but rejected by the threshold.
- OCR accuracy depends on the image and text quality.

## 3. CLIP

- Generated three 512-dimensional image embeddings.
- English retrieval: 3/3 queries correct at rank 1.
- Arabic retrieval: 2/3 queries correct at rank 1.
- Overall: 5/6 correct on this small test set.
- Arabic corridor query incorrectly ranked the PUSH
  sign above the corridor.

## 4. Combined GPU Test

- All three models loaded successfully on GPU.
- Total GPU usage after initial inference: 2044.94 MiB.
- Total GPU usage after five repeated passes: 2562.94 MiB.
- Peak PyTorch allocated: 1507.97 MiB.
- No CUDA out-of-memory errors.

Average timing across passes 2–5:

| Model | Time |
|---|---:|
| YOLOE | 63.3 ms |
| EasyOCR | 435.6 ms |
| CLIP | 28.3 ms |
| Total | 527.2 ms |

## 5. Risks and Observations

1. YOLOE detection depends on class wording.
2. Arabic OCR confidence may be low even when text
   is recognized correctly.
3. Multilingual CLIP retrieval is imperfect.
4. EasyOCR dominates sequential processing time.
5. GPU memory fits in the current test, but longer
   and full-pipeline tests remain necessary.

## Conclusion

The three models work on the development GPU.

The initial feasibility test is successful, but
the production ingest pipeline still needs
performance optimization and broader evaluation.
