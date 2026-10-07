-- db/migration_v6.sql
-- Structured edit plans (Phase 5). Idempotent.
-- Run: psql viral_clips -f db/migration_v6.sql
CREATE TABLE IF NOT EXISTS edit_plans (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id   UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    clip_id    UUID NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    version    INT NOT NULL DEFAULT 1,
    plan       JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS edit_plans_clip_idx ON edit_plans(clip_id);
