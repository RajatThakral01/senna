-- db/migration_v4.sql
-- Boundary refinement + vertical framing columns (idempotent).
-- Run: psql viral_clips -f db/migration_v4.sql
ALTER TABLE clips ADD COLUMN IF NOT EXISTS refine_status TEXT DEFAULT 'pending';
ALTER TABLE clips ADD COLUMN IF NOT EXISTS refine_reason TEXT DEFAULT '';
ALTER TABLE clips ADD COLUMN IF NOT EXISTS source_ranges JSONB DEFAULT '[]'::jsonb;
ALTER TABLE clips ADD COLUMN IF NOT EXISTS timeline JSONB DEFAULT '[]'::jsonb;
ALTER TABLE clips ADD COLUMN IF NOT EXISTS layout TEXT DEFAULT 'auto';
ALTER TABLE clips ADD COLUMN IF NOT EXISTS layout_reason TEXT DEFAULT '';
-- pipeline_runs already allows arbitrary stage names; 'refine' needs no DDL,
-- but invalidate stale render checkpoints when boundaries/framing change:
-- (handled in code; no schema change required here).
