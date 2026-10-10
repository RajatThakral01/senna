-- db/migration_v7.sql
-- ─────────────────────────────────────────────────────────────────────────────
-- Campaign pipeline (CAMPAIGN_PIPELINE_PLAN.md §7): campaigns, their asset
-- library, jobs over inputs (clip mode = long video, edit mode = supplied
-- short videos) and the deliverables they produce.
-- Idempotent. Applied by db/init_db.sh.
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS campaigns (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT NOT NULL,
    slug            TEXT NOT NULL UNIQUE,          -- folder name under assets/campaigns/
    brief           TEXT NOT NULL DEFAULT '',      -- requirements as given by the campaign owner
    requirements    JSONB NOT NULL DEFAULT '{}'::jsonb,  -- parsed, human-readable summary
    recipe          JSONB NOT NULL DEFAULT '{}'::jsonb,  -- approved edit recipe (campaign/recipe.py)
    recipe_version  INTEGER NOT NULL DEFAULT 0,    -- bumped on every recipe save
    platforms       JSONB NOT NULL DEFAULT '[]'::jsonb,  -- export targets, e.g. ["tiktok","reels"]
    status          TEXT NOT NULL DEFAULT 'draft', -- draft | ready | archived
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS campaign_assets (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_id     UUID NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    kind            TEXT NOT NULL,                 -- logo | audio | font | video | image
    name            TEXT NOT NULL,                 -- file name as shown to the user / parser
    path            TEXT NOT NULL,                 -- stored copy, assets/campaigns/<slug>/<kind>/<name>
    meta            JSONB NOT NULL DEFAULT '{}'::jsonb,  -- probe: duration, width, height, has_audio…
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (campaign_id, kind, name)
);
CREATE INDEX IF NOT EXISTS campaign_assets_campaign_idx ON campaign_assets(campaign_id);

CREATE TABLE IF NOT EXISTS jobs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_id     UUID NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    status          TEXT NOT NULL DEFAULT 'queued',  -- queued | running | done | failed | cancelled
    mode_policy     TEXT NOT NULL DEFAULT 'auto',    -- auto | clip | edit (default for its inputs)
    recipe_version  INTEGER,                          -- campaign recipe version the job runs with
    recipe          JSONB NOT NULL DEFAULT '{}'::jsonb,  -- frozen copy of that recipe
    progress        REAL NOT NULL DEFAULT 0,
    message         TEXT,
    error           TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    started_at      TIMESTAMPTZ,
    finished_at     TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS jobs_campaign_idx ON jobs(campaign_id);

CREATE TABLE IF NOT EXISTS job_inputs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id          UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    idx             INTEGER NOT NULL,              -- order within the job
    source          TEXT NOT NULL,                 -- URL or local path as given
    local_path      TEXT,                          -- resolved local file (after download)
    probe           JSONB NOT NULL DEFAULT '{}'::jsonb,
    mode            TEXT,                          -- clip | edit (decided by the router / user)
    video_id        UUID REFERENCES videos(id) ON DELETE SET NULL,  -- clip mode: pipeline video row
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending | running | done | failed | skipped
    error           TEXT,
    UNIQUE (job_id, idx)
);
CREATE INDEX IF NOT EXISTS job_inputs_job_idx ON job_inputs(job_id);

CREATE TABLE IF NOT EXISTS deliverables (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id          UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    job_input_id    UUID REFERENCES job_inputs(id) ON DELETE CASCADE,
    clip_id         UUID REFERENCES clips(id) ON DELETE SET NULL,  -- clip mode source clip
    variant         TEXT NOT NULL DEFAULT 'main',
    platform        TEXT NOT NULL DEFAULT 'generic',
    path            TEXT,
    recipe_version  INTEGER,
    qa              JSONB NOT NULL DEFAULT '{}'::jsonb,  -- compliance checklist (phase C6)
    status          TEXT NOT NULL DEFAULT 'pending',     -- pending | rendered | failed | approved | rejected
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS deliverables_job_idx ON deliverables(job_id);
