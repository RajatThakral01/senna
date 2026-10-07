"""Rerun the pipeline from stage 5 (embed) onward for an existing video_id.

Usage:
    python rerun_from_embed.py <video_id> [--log-level DEBUG]

What it does, in order:
  1. Validates the video exists (videos table).
  2. Cleanup (dup-safe):
       - DELETE related_segments rows for this video's clips
       - DELETE clips rows for this video (analyze re-inserts fresh rows,
         so re-running can never create duplicates)
       - DELETE candidates + outlines + audio_events rows for this video
       - DELETE pipeline_runs checkpoints for
         audio_events/embed/outline/analyze/similarity/refine/render
         so should_skip_stage() lets each stage run again.
     Chunks and transcripts are reused untouched.
  3. Audio events: detect_audio_events() on source audio + event cues.
  4. Stage 5 (embed):    embed_all_chunks() — NULL/pending chunks get fresh
     1024-dim local vectors.
  5. Outline:            build_outline() — video structure (fingerprint-gated).
  6. Stage 6 (analyze):  analyze() — two-pass discovery + legacy fallback.
  7. Stage 7 (similarity): find_related_segments() — pgvector search +
     LLM stitch/standalone/noise classification.
  8. Stage 8 (refine):   refine_all_clips() — sentence-complete boundaries.
  9. Stage 9 (render):   render_clips() — cut/vertical/subtitles/logo/music/
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
from pipeline.boundaries import refine_all_clips, persist_refinements
from main import render_clips, get_video_duration, invalidate_stage
from config import get_config

setup_logging()
log = get_logger("rerun")

STAGES = ("audio_events", "embed", "outline", "analyze", "similarity", "refine", "render")


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
            cur.execute("DELETE FROM candidates WHERE video_id = %s", (video_id,))
            ncand = cur.rowcount
            cur.execute("DELETE FROM outlines WHERE video_id = %s", (video_id,))
            nout = cur.rowcount
            cur.execute("""
                DELETE FROM pipeline_runs
                WHERE video_id = %s AND stage IN ('audio_events','embed','outline','analyze','similarity','refine','render')
            """, (video_id,))
            nruns = cur.rowcount
            conn.commit()
            return {"related_segments": nrel, "clips": nclips, "candidates": ncand,
                    "outlines": nout, "pipeline_runs": nruns}
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
        # derived discovery/event rows cleared alongside clips
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                for _t in ("candidates", "outlines", "audio_events"):
                    cur.execute(f"DELETE FROM {_t} WHERE video_id = %s", (video_id,))
                    counts[_t] = cur.rowcount
                conn.commit()
        finally:
            release_conn(conn)
        plog.info("cleanup done deleted=%s", counts)

    config = get_config()

    if True:  # Stage: Audio events
        from pipeline.audio_events import (
            ensure_source_audio, detect_audio_events, propose_from_events,
            enrich_event_candidates, persist_events)
        from pipeline.audio_classifier import classify_events
        from pipeline.boundaries import load_words as _lw, build_sentences as _bs
        from db.repositories import candidate_repo as _cr
        run_repo.start_stage(video_id, "audio_events")
        try:
            with log_stage("rerun", "audio_events", video_id=video_id):
                a_cfg = dict(config.get("audio_events", {}))
                if a_cfg.get("enabled", True):
                    wav = ensure_source_audio(
                        video.get("raw_path") or "input/raw_video.mp4",
                        "downloads/audio.wav")
                    events, _ = detect_audio_events(wav, config)
                    try:
                        import librosa as _lib
                        _y, _sr = _lib.load(wav, sr=16000, mono=True)
                    except Exception:
                        _y, _sr = None, 16000
                    events, _cstat = classify_events(events, _y, _sr, config)
                    persist_events(video_id, events)
                    _words = _lw("transcripts/transcript.json")
                    _sents = _bs(_words)
                    cues = enrich_event_candidates(
                        propose_from_events(events, _words, _sents, config), config)
                    _cr.clear_candidates(video_id, source="audio_event")
                    for _c in cues:
                        _cr.insert_candidate(
                            video_id, "audio_event", _c["sentence_ids"],
                            _c["source_ranges"], rationale=_c.get("rationale"),
                            uncertainty=_c.get("uncertainty"),
                            scores={"provenance": _c.get("provenance", {})},
                            status="proposed")
                    plog.info("audio_events done events=%d cues=%d",
                              len(events), len(cues))
            run_repo.complete_stage(video_id, "audio_events")
        except Exception as e:
            run_repo.fail_stage(video_id, "audio_events", str(e))
            raise

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

    if True:  # Stage 6: Outline
        from pipeline.boundaries import load_words, build_sentences
        from pipeline.outline import build_outline, persist_outline
        from pipeline.fingerprints import (outline_fingerprint,
                                            transcript_signature)
        run_repo.start_stage(video_id, "outline")
        try:
            with log_stage("rerun", "outline", video_id=video_id):
                tsig = transcript_signature("transcripts/transcript.json")
                ofp = outline_fingerprint(
                    tsig, config.get("ai", {}).get("llm_model", ""), config)
                words = load_words("transcripts/transcript.json")
                sents = build_sentences(words)
                ol = build_outline(words, sents, None, cfg=config)
                persist_outline(video_id, ol)
                run_repo.set_fingerprint(video_id, "outline", ofp)
                plog.info("outline done method=%s", ol.get("method"))
            run_repo.complete_stage(video_id, "outline")
        except Exception as e:
            run_repo.fail_stage(video_id, "outline", str(e))
            raise

    if True:  # Stage 6b: Analyze
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

    # Stage 8: Refine (sentence-complete boundaries, LLM-validated)
    run_repo.start_stage(video_id, "refine")
    try:
        with log_stage("rerun", "refine", video_id=video_id):
            clips = clip_repo.get_clips_for_video(video_id)
            confirmed = clip_repo.get_confirmed_segments_for_video(video_id)
            for c in clips:
                c["related_segments"] = confirmed.get(c["id"], [])
            refine_cfg = dict(config.get("refine", {}))
            if refine_cfg.get("enabled", True):
                vid_dur = get_video_duration(video.get("raw_path") or "input/raw_video.mp4")
                refined = refine_all_clips(clips,
                                           transcript_path="transcripts/transcript.json",
                                           video_duration=vid_dur,
                                           cfg={**config, **refine_cfg})
                from pipeline.fusion import deduplicate_refined
                _kept, _dropped = deduplicate_refined(refined)
                for _dc, _why in _dropped:
                    _dc["refine_status"] = "rejected"
                    _dc["refine_reason"] = _why[:500]
                persist_refinements(video_id, refined)
                plog.info("refine found %d clips (%d rejected)",
                          len(refined),
                          sum(1 for c in refined if c.get("refine_status") == "rejected"))
        run_repo.complete_stage(video_id, "refine")
        invalidate_stage(video_id, "render")
    except Exception as e:
        run_repo.fail_stage(video_id, "refine", str(e))
        raise

    # Stage 9: Render (raw path + default campaign resolved inside)
    finals = render_clips(video_id)
    plog.info("rerun complete finals=%d", len(finals))

    print_decisions(video_id)


if __name__ == "__main__":
    sys.exit(main())
