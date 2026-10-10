"""Rerun the pipeline for an existing video_id from a chosen stage onward.

Usage:
    python rerun_from_embed.py <video_id> [--from-stage audio_events] [--log-level DEBUG]

Thin wrapper over main.run_pipeline(video_id=..., from_stage=...): the
transcript (and everything before --from-stage) is reused; that stage and
every later one are redone with their derived rows cleared first
(main.reset_from_stage), so re-running never creates duplicates. Default
--from-stage is audio_events, i.e. the old "embed onward" behaviour
(already-embedded chunks are kept, so the embed stage is near-instant).

The same is available directly as:
    python main.py <source> --from-stage <stage>
Afterwards it prints each candidate's similarity score next to the LLM's
stitch/standalone/noise decision.
"""
import argparse
import sys
import uuid

from logger import get_logger, setup_logging
from db.connection import get_conn, release_conn
from db.repositories import video_repo
from main import run_pipeline, STAGES

setup_logging()
log = get_logger("rerun")


def _check_uuid(video_id: str) -> str:
    try:
        return str(uuid.UUID(video_id))
    except ValueError:
        raise SystemExit(f"Not a valid UUID: {video_id!r}")


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
    parser = argparse.ArgumentParser(description="Rerun an existing video from a stage onward")
    parser.add_argument("video_id", help="UUID of the existing video row")
    parser.add_argument("--from-stage", choices=STAGES, default="audio_events",
                        help="first stage to redo (default: audio_events)")
    parser.add_argument("--layout", default=None)
    parser.add_argument("--log-level", default=None, help="DEBUG/INFO/WARNING/ERROR")
    args = parser.parse_args()
    if args.log_level:
        setup_logging(level=args.log_level.upper(), force=True)

    video_id = _check_uuid(args.video_id)
    video = video_repo.get_video(video_id)
    if not video:
        raise SystemExit(f"No video row for id {video_id}")
    source = video.get("source_url") or video.get("raw_path")
    log.info("rerun video=%.8s source=%.80s from_stage=%s", video_id, source, args.from_stage)
    finals = run_pipeline(source, layout=args.layout, video_id=video_id,
                          from_stage=args.from_stage)
    log.info("rerun complete finals=%d", len(finals))
    print_decisions(video_id)


if __name__ == "__main__":
    sys.exit(main())
