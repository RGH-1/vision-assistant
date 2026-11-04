
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS pgcrypto;


CREATE TABLE sessions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at    TIMESTAMPTZ,
    app_lang    TEXT NOT NULL CHECK (app_lang IN ('en','ar','fr')),
    consent_at  TIMESTAMPTZ NOT NULL,
    consent_ver TEXT NOT NULL
);


CREATE TABLE frames (
    id            BIGSERIAL PRIMARY KEY,
    session_id    UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    captured_at   TIMESTAMPTZ NOT NULL,
    image_path    TEXT NOT NULL,
    thumb_path    TEXT NOT NULL,
    original_path TEXT,
    width         INT NOT NULL,
    height        INT NOT NULL,
    sharpness     REAL NOT NULL,
    brightness    REAL NOT NULL,
    contrast      REAL NOT NULL,
    edge_density  REAL NOT NULL,
    enhanced      BOOLEAN NOT NULL DEFAULT false,
    enhance_ops   TEXT[] NOT NULL DEFAULT '{}',
    dhash         BIGINT NOT NULL,
    keep_reason   TEXT NOT NULL CHECK (
        keep_reason IN ('novel','new_objects','text','heartbeat')
    ),
    faces_blurred INT NOT NULL DEFAULT 0,
    scene_text    TEXT NOT NULL DEFAULT '',
    scene_tsv     TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('simple', scene_text)
    ) STORED,
    clip_emb      VECTOR(512) NOT NULL,
    timings_ms    JSONB NOT NULL DEFAULT '{}'
);


CREATE INDEX frames_session_time
    ON frames (session_id, captured_at DESC);

CREATE INDEX frames_clip_hnsw
    ON frames USING hnsw (clip_emb vector_cosine_ops);

CREATE INDEX frames_scene_tsv
    ON frames USING gin (scene_tsv);



CREATE TABLE detections (
    id          BIGSERIAL PRIMARY KEY,
    frame_id    BIGINT NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    label       TEXT NOT NULL,
    confidence  REAL NOT NULL,
    bbox        REAL[] NOT NULL,
    area_ratio  REAL NOT NULL,
    position    TEXT NOT NULL CHECK (
        position IN ('left','center','right')
    )
);

CREATE INDEX detections_label
    ON detections (label, frame_id);

CREATE INDEX detections_frame
    ON detections (frame_id);


CREATE TABLE ocr_texts (
    id          BIGSERIAL PRIMARY KEY,
    frame_id    BIGINT NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    text        TEXT NOT NULL,
    text_norm   TEXT NOT NULL,
    confidence  REAL NOT NULL,
    bbox        REAL[] NOT NULL,
    from_crop   BOOLEAN NOT NULL DEFAULT false
);

CREATE INDEX ocr_frame
    ON ocr_texts (frame_id);

CREATE INDEX ocr_norm_trgm
    ON ocr_texts USING gin (text_norm gin_trgm_ops);


CREATE TABLE questions (
    id             BIGSERIAL PRIMARY KEY,
    session_id     UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    asked_at       TIMESTAMPTZ NOT NULL,
    input_mode     TEXT NOT NULL CHECK (input_mode IN ('voice','text')),
    stt_engine     TEXT,
    transcript     TEXT NOT NULL,
    lang           TEXT NOT NULL,
    intent         TEXT NOT NULL,
    t_from         TIMESTAMPTZ,
    t_to           TIMESTAMPTZ,
    answer         TEXT NOT NULL,
    confidence     TEXT NOT NULL CHECK (
        confidence IN ('high','medium','low','none')
    ),
    abstained      BOOLEAN NOT NULL,
    safety_rewrite BOOLEAN NOT NULL DEFAULT false,
    model          TEXT,
    timings_ms     JSONB NOT NULL
);

CREATE INDEX questions_session_time
    ON questions (session_id, asked_at DESC);


CREATE TABLE question_evidence (
    question_id BIGINT NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    frame_id    BIGINT NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    rank        SMALLINT NOT NULL,
    score       REAL NOT NULL,
    parts       JSONB NOT NULL,
    PRIMARY KEY (question_id, frame_id)
);


CREATE TABLE alerts (
    id          BIGSERIAL PRIMARY KEY,
    session_id  UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    frame_id    BIGINT REFERENCES frames(id) ON DELETE SET NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    kind        TEXT NOT NULL,
    message     TEXT NOT NULL,
    lang        TEXT NOT NULL
);

CREATE INDEX alerts_session_time
    ON alerts (session_id, created_at DESC);
