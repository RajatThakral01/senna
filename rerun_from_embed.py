"""Rerun the pipeline from stage 5 (embed) onward for an existing video_id.

Usage:
    python rerun_from_embed.py <video_id> [--log-level DEBUG]

What it does, in order:
  1. Validates the video exists (videos table).
  2. Cleanup (dup-safe):
       - DELETE related_segments rows for this video's clips
       - DELETE clips rows for this video (analyze re-inserts fresh rows,
         so re-running can never create duplicates)
       - DELETE pipeline_runs checkpoints for embed/analyze/similarity/render
         so should_skip_stage() lets each stage run again.
     Chunks and transcripts are reused untouched.
  3. Stage 5 (embed):    embed_all_chunks() — NULL/pending chunks get fresh
     1024-dim local vectors.
  4. Stage 6 (analyze):  analyze() — Grok LLM finds viral clips.
  5. Stage 7 (similarity): find_related_segments() — pgvector search +
     LLM stitch/standalone/noise classification.
  6. Stage 8 (render):   render_clips() — cut/vertical/subtitles/logo/music/
     fades + report.json.
  7. Prints a per-clip summary table with each candidate's similarity score
     next to the LLM's stitch/standalone/noise decision.
"""
import argparse
import sys
import uuid

from logger import get_logger, setup_logging, log_stage, bind
from db.connection import get_conn, release_conn
from db.repositories import video_repo, chunk_repo, clip_repo, run_repo
from pipeline.embedder import embed_all_chunks
from pipeline.analyzer import analyze
from pipeline.similarity import find_related_segments
from main import render_clips
from config import get_config

setup_logging()
log = get_logger("rerun")

STAGES = ("embed", "analyze", "similarity", "render")


def _check_uuid(video_id: str) -> str:
    try:
        return str(uuid.UUID(video_id))
    except ValueError:
        raise SystemExit(f"Not a valid UUID: {video_id!r}")


def cleanup(video_id: str) -> dict:
    """Wipe derived rows for a re-run. Returns counts for logging."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                DELETE FROM related_segments
                WHERE clip_id IN (SELECT id FROM clips WHERE video_id = %s)
            """, (video_id,))
            nrel = cur.rowcount
            cur.execute("DELETE FROM clips WHERE video_id = %s", (video_id,))
            nclips = cur.rowcount
            cur.execute("""
                DELETE FROM pipeline_runs
                WHERE video_id = %s AND stage IN ('embed','analyze','similarity','render')
            """, (video_id,))
            nruns = cur.rowcount
            conn.commit()
            return {"related_segments": nrel, "clips": nclips, "pipeline_runs": nruns}
    finally:
        release_conn(conn)


def print_decisions(video_id: str) -> None:
    """Print similarity score next to the LLM decision for every candidate."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT c.clip_number, ch.chunk_index,
                       r.similarity_score, r.decision, r.confirmed_by_llm
                FROM related_segments r
                JOIN clips c  ON c.id = r.clip_id
                JOIN chunks ch ON ch.id = r.related_chunk_id
                WHERE c.video_id = %s
                ORDER BY c.clip_number, r.similarity_score DESC
            """, (video_id,))
            rows = cur.fetchall()
    finally:
        release_conn(conn)
    print(f"\n=== similarity decisions for video {video_id[:8]} ({len(rows)} candidates) ===")
    print(f"{'clip':>4}  {'chunk':>5}  {'sim':>7}  {'decision':>10}  stitched")
    for clip_no, chunk_idx, sim, decision, confirmed in rows:
        print(f"{clip_no:>4}  {chunk_idx:>5}  {sim:>7.4f}  {decision:>10}  {confirmed}")
    if not rows:
        print("(no candidates above threshold)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun pipeline from embed stage onward")
    parser.add_argument("video_id", help="UUID of the existing video row")
    parser.add_argument("--log-level", default=None, help="DEBUG/INFO/WARNING/ERROR")
    args = parser.parse_args()
    if args.log_level:
        setup_logging(level=args.log_level.upper(), force=True)

    video_id = _check_uuid(args.video_id)
    video = video_repo.get_video(video_id)
    if not video:
        raise SystemExit(f"No video row for id {video_id}")
    plog = bind("rerun", video_id=video_id)
    plog.info("rerun start src=%.80s raw=%s", video.get("source_url"), video.get("raw_path"))

    with log_stage("rerun", "cleanup", video_id=video_id):
        counts = cleanup(video_id)
        plog.info("cleanup done deleted=%s", counts)

    config = get_config()

    if True:  # Stage 5: Embed
        run_repo.start_stage(video_id, "embed")
        try:
            with log_stage("rerun", "embed", video_id=video_id):
                chunks = chunk_repo.get_chunks_for_video(video_id)
                if not chunks:
                    raise SystemExit(f"No chunks for video {video_id} — nothing to embed")
                embed_all_chunks(video_id, chunks)
            run_repo.complete_stage(video_id, "embed")
        except Exception as e:
            run_repo.fail_stage(video_id, "embed", str(e))
            raise

    if True:  # Stage 6: Analyze
        run_repo.start_stage(video_id, "analyze")
        try:
            with log_stage("rerun", "analyze", video_id=video_id):
                chunks = chunk_repo.get_chunks_for_video(video_id)
                clips = analyze(video_id, chunks, config)
                plog.info("analyze found %d clips", len(clips))
            run_repo.complete_stage(video_id, "analyze")
        except Exception as e:
            run_repo.fail_stage(video_id, "analyze", str(e))
            raise

    if True:  # Stage 7: Similarity
        run_repo.start_stage(video_id, "similarity")
        try:
            with log_stage("rerun", "similarity", video_id=video_id):
                clips = clip_repo.get_clips_for_video(video_id)
                find_related_segments(
                    video_id, clips,
                    top_k=config["similarity"]["top_k"],
                    threshold=config["similarity"]["threshold"],
                )
            run_repo.complete_stage(video_id, "similarity")
        except Exception as e:
            run_repo.fail_stage(video_id, "similarity", str(e))
            raise

    # Stage 8: Render (raw path + default campaign resolved inside)
    finals = render_clips(video_id)
    plog.info("rerun complete finals=%d", len(finals))

    print_decisions(video_id)


if __name__ == "__main__":
    sys.exit(main())
