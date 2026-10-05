-- db/schema.sql
-- ─────────────────────────────────────────────────────────────────────────────
-- Viral Clips Automator — PostgreSQL Schema (v3.0)
-- Run once against the viral_clips database:
--   psql viral_clips -f db/schema.sql
--
-- Requires: CREATE EXTENSION IF NOT EXISTS vector;  (pgvector must be installed)
-- ─────────────────────────────────────────────────────────────────────────────

-- ─────────────────────────────────────────────
-- Table: videos
-- One row per source video ingested
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS videos (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_url       TEXT,                        -- original URL or local path
    raw_path         TEXT NOT NULL,               -- path to raw_video.mp4
    duration_seconds FLOAT,
    status           TEXT DEFAULT 'pending',      -- pending | transcribed | chunked | embedded | analyzed | done | failed
    campaign_id      TEXT,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    updated_at       TIMESTAMPTZ DEFAULT NOW()
);

-- ─────────────────────────────────────────────
-- Table: chunks
-- One row per transcript chunk produced by the chunker
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS chunks (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id            UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    chunk_index         INT NOT NULL,               -- 0-based position in the sequence
    start_time          FLOAT NOT NULL,             -- seconds from video start
    end_time            FLOAT NOT NULL,
    text                TEXT NOT NULL,              -- raw transcript text for this chunk
    token_count         INT,
    is_overlap_tail     BOOLEAN DEFAULT FALSE,      -- TRUE if this chunk is a re-attached overlap region
    silence_gap_before  FLOAT,                     -- seconds of silence before this chunk starts
    embedding           vector(1024),               -- Qwen3-Embedding-0.6B local embedding; NULL until embedder runs
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS chunks_video_id_idx ON chunks(video_id);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);

-- ─────────────────────────────────────────────
-- Table: clips
-- One row per viral moment identified by the analyzer
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS clips (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id            UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    clip_number         INT NOT NULL,
    start_time          FLOAT NOT NULL,
    end_time            FLOAT NOT NULL,
    duration_seconds    FLOAT,
    hook                TEXT,
    reason              TEXT,
    suggested_title     TEXT,
    suggested_hashtags  TEXT[],
    source_chunk_ids    UUID[],                    -- which chunk(s) the LLM was reading when it found this
    embedding           vector(1024),               -- embedding of the clip's hook+text (Qwen3-Embedding-0.6B, query prompt)
    output_path         TEXT,                       -- path to final rendered .mp4, null until rendering
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS clips_video_id_idx ON clips(video_id);
CREATE INDEX IF NOT EXISTS clips_embedding_idx ON clips USING hnsw (embedding vector_cosine_ops);

-- ─────────────────────────────────────────────
-- Table: related_segments
-- Stores similarity links between clips and continuation chunks
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS related_segments (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    clip_id             UUID NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    related_chunk_id    UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    similarity_score    FLOAT NOT NULL,
    confirmed_by_llm    BOOLEAN DEFAULT FALSE,
    decision            TEXT DEFAULT 'noise',       -- stitch | standalone | noise
    stitched_into_clip  BOOLEAN DEFAULT FALSE,      -- TRUE once FFmpeg has included this segment
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ─────────────────────────────────────────────
-- Table: pipeline_runs
-- Tracks per-run stage progress for resumption
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id        UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    stage           TEXT NOT NULL,              -- transcribe | chunk | embed | analyze | similarity | render
    status          TEXT DEFAULT 'pending',     -- pending | running | done | failed
    error_message   TEXT,
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ
);
