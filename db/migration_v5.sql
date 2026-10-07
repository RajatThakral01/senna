-- db/migration_v5.sql
-- Contextual discovery: outlines, candidates, audio events, stage fingerprints.
-- Idempotent. Run: psql viral_clips -f db/migration_v5.sql
-- (Application code also attempts these DDLs at startup via ensure_v5().)

-- ── Video outlines (section- and video-level, JSONB content) ──
CREATE TABLE IF NOT EXISTS outlines (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id     UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    level        TEXT NOT NULL DEFAULT 'section',  -- section | video
    idx          INT NOT NULL DEFAULT 0,           -- order within level
    start_time   FLOAT,
    end_time     FLOAT,
    title        TEXT,
    content      JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS outlines_video_idx ON outlines(video_id);

-- ── Candidate pool (all discovery sources; clips = selected set) ──
CREATE TABLE IF NOT EXISTS candidates (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id         UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    source           TEXT NOT NULL DEFAULT 'outline',  -- outline | audio_event | similarity | manual | legacy
    sentence_ids     JSONB NOT NULL DEFAULT '[]'::jsonb,
    source_ranges    JSONB NOT NULL DEFAULT '[]'::jsonb, -- [[start,end],...] derived from words
    hook             TEXT,
    main_idea        TEXT,
    payoff           TEXT,
    required_context TEXT,
    rationale        TEXT,
    uncertainty      FLOAT,
    scores           JSONB NOT NULL DEFAULT '{}'::jsonb,  -- component scores + reasons
    status           TEXT NOT NULL DEFAULT 'proposed',    -- proposed|shortlisted|selected|rejected
    status_reason    TEXT DEFAULT '',
    clip_id          UUID REFERENCES clips(id) ON DELETE SET NULL,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS candidates_video_idx ON candidates(video_id);

-- ── Audio events (energy spikes + optional labels) ──
CREATE TABLE IF NOT EXISTS audio_events (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id          UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    start_time        FLOAT NOT NULL,
    end_time          FLOAT NOT NULL,
    peak_time         FLOAT,
    energy_increase   FLOAT,        -- relative increase over local baseline
    confidence        FLOAT,        -- detection confidence 0..1
    method            TEXT DEFAULT 'rms',
    version           TEXT DEFAULT 'v1',
    label             TEXT,         -- laughter|applause|music|... (classifier, nullable)
    label_confidence  FLOAT,
    config_fingerprint TEXT DEFAULT '',
    created_at        TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS audio_events_video_idx ON audio_events(video_id);

-- ── Stage fingerprints (invalidate on source/model/prompt/config change) ──
CREATE TABLE IF NOT EXISTS stage_fingerprints (
    video_id     UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    stage        TEXT NOT NULL,
    fingerprint  TEXT NOT NULL DEFAULT '',
    updated_at   TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (video_id, stage)
);

-- ── Clip provenance (which discovery pass produced this clip) ──
ALTER TABLE clips ADD COLUMN IF NOT EXISTS provenance JSONB DEFAULT '{}'::jsonb;
