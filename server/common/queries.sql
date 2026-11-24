-- Assistant candidate query: meaning + OCR + objects within one time window (§7.4).
-- The final weighted ranking and duplicate removal belong to M3 (§6.5.3).
WITH win AS (
    SELECT f.*
    FROM frames AS f
    WHERE f.session_id = %(session_id)s
      AND f.captured_at BETWEEN %(time_from)s AND %(time_to)s
),
txt AS (
    SELECT o.frame_id, max(similarity(o.text_norm, %(text)s)) AS s_text
    FROM ocr_texts AS o
    JOIN win ON win.id = o.frame_id
    WHERE %(text)s <> '' AND o.text_norm %% %(text)s
    GROUP BY o.frame_id
),
obj AS (
    SELECT d.frame_id, max(d.confidence) AS s_obj
    FROM detections AS d
    JOIN win ON win.id = d.frame_id
    WHERE d.label = ANY(%(target_labels)s::text[])
    GROUP BY d.frame_id
)
SELECT
    win.id,
    win.captured_at,
    win.image_path,
    win.dhash,
    win.sharpness,
    1 - (win.clip_emb <=> %(clip_vector)s::vector) AS s_clip,
    COALESCE(txt.s_text, 0) AS s_text,
    COALESCE(obj.s_obj, 0) AS s_obj,
    EXTRACT(EPOCH FROM (%(asked_at)s::timestamptz - win.captured_at)) AS seconds_ago
FROM win
LEFT JOIN txt ON txt.frame_id = win.id
LEFT JOIN obj ON obj.frame_id = win.id
ORDER BY
    (1 - (win.clip_emb <=> %(clip_vector)s::vector))
    + COALESCE(txt.s_text, 0)
    + COALESCE(obj.s_obj, 0) DESC
LIMIT 30;
