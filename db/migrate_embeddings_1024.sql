-- db/migrate_embeddings_1024.sql
-- ─────────────────────────────────────────────────────────────────────────────
-- Migrate the viral_clips database to Qwen3-Embedding-0.6B vectors (1024-dim).
--
-- 0. BACKUP FIRST (run in a terminal before applying):
--      pg_dump viral_clips -F c -f backup_before_1024.dump
--    or plain SQL:
--      pg_dump viral_clips -f backup_before_1024.sql
-- 1. APPLY:
--      psql viral_clips -f db/migrate_embeddings_1024.sql
-- 2. VERIFY (also printed automatically at the end of this script):
--      \d chunks
--      \d clips
--    embedding must be vector(1024) and both HNSW indexes must show
--    vector_cosine_ops.
--
-- WARNING: old embeddings were computed with a different model/dimension and
-- cannot be converted. Re-run the pipeline from the embed stage afterwards
-- (see rerun_from_embed.py) to regenerate them.
-- ─────────────────────────────────────────────────────────────────────────────

BEGIN;

-- 0. Pre-flight: show the REAL index names on both tables (compare with the
--    DROP/CREATE statements below — they must match).
SELECT indexname, indexdef
  FROM pg_indexes
 WHERE schemaname = 'current_schema'() AND tablename IN ('chunks', 'clips')
 ORDER BY tablename, indexname;

-- 1. Clear stale embeddings (old dim cannot cast to vector(1024))
UPDATE chunks SET embedding = NULL;
UPDATE clips  SET embedding = NULL;

-- 2. Drop similarity links computed with the old vectors
DELETE FROM related_segments;

-- 3. Drop HNSW indexes (dimension is part of the index).
--    Names below are from db/schema.sql; IF EXISTS keeps this safe even if
--    your DB used different names — but then create the matching DROP for
--    any extra name shown in the pre-flight list above.
DROP INDEX IF EXISTS chunks_embedding_idx;
DROP INDEX IF EXISTS clips_embedding_idx;

-- 4. Alter column dimensions
ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(1024);
ALTER TABLE clips  ALTER COLUMN embedding TYPE vector(1024);

-- 5. Recreate HNSW indexes
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS clips_embedding_idx ON clips USING hnsw (embedding vector_cosine_ops);

COMMIT;

-- 6. Post-flight verification (psql prints these when run with -f):
\echo '=== chunks ==='
\d chunks
\echo '=== clips ==='
\d clips
