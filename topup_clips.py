"""topup_clips.py — top up clips with deterministic section spans (dead-LLM path).

Usage:
    .venv/Scripts/python.exe topup_clips.py <video_id>

Only acts when the outline is non-LLM (LLM unreachable): builds deterministic
section candidates excluding existing clip ranges, inserts them (numbered to
follow), then re-runs similarity -> refine -> render. Existing finals are
kept (render skips them); only new clips render.
"""
import sys

sys.path.insert(0, __import__("os").path.dirname(__file__))

from logger import setup_logging, get_logger

setup_logging()
log = get_logger("topup")


def main():
    from config import get_config
    from db.repositories import video_repo, chunk_repo, clip_repo, run_repo
    from pipeline.boundaries import load_words, build_sentences
    from pipeline.analyzer import _deterministic_section_clips
    from pipeline.similarity import find_related_segments
    from pipeline.boundaries import refine_all_clips, persist_refinements
    from pipeline.fusion import deduplicate_refined
    from main import render_clips, get_video_duration, invalidate_stage

    video_id = sys.argv[1]
    config = get_config()
    video = video_repo.get_video(video_id)
    assert video, f"unknown video {video_id}"
    target = int(config.get("discovery", {}).get("target_clips", 15) or 15)

    words = load_words("transcripts/transcript.json")
    sentences = build_sentences(words)
    chunks = chunk_repo.get_chunks_for_video(video_id)
    existing = clip_repo.get_clips_for_video(video_id)
    log.info("existing clips=%d target=%d", len(existing), target)
    if len(existing) >= target:
        print(f"already at target ({len(existing)} >= {target}), nothing to do")
        return

    # rebuild outline dict from DB rows (persisted by analyze)
    from db.repositories import outline_repo
    secs = outline_repo.get_outlines(video_id, "section")
    outline = {"method": "deterministic", "sections": [
        {"title": s.get("title"),
         "start_id": s["content"].get("start_id"),
         "end_id": s["content"].get("end_id"),
         "summary": s["content"].get("summary", "")}
        for s in secs]} if secs else None
    if not outline:
        print("no outline rows, cannot top up")
        return

    new_clips = _deterministic_section_clips(
        video_id, outline, words, sentences, chunks,
        target - len(existing), campaign_text="", cfg=config,
        exclude_ranges=[(c["start_time"], c["end_time"]) for c in existing],
        start_clip_number=len(existing) + 1)
    print(f"inserted {len(new_clips)} top-up clips")
    if not new_clips:
        return

    for st in ("similarity", "refine", "render"):
        invalidate_stage(video_id, st)

    run_repo.start_stage(video_id, "similarity")
    try:
        clips = clip_repo.get_clips_for_video(video_id)
        find_related_segments(video_id, clips,
                              top_k=config["similarity"]["top_k"],
                              threshold=config["similarity"]["threshold"])
        run_repo.complete_stage(video_id, "similarity")
    except Exception as e:
        run_repo.fail_stage(video_id, "similarity", str(e))
        raise

    run_repo.start_stage(video_id, "refine")
    try:
        clips = clip_repo.get_clips_for_video(video_id)
        confirmed = clip_repo.get_confirmed_segments_for_video(video_id)
        for c in clips:
            c["related_segments"] = confirmed.get(c["id"], [])
        refined = refine_all_clips(
            clips, transcript_path="transcripts/transcript.json",
            video_duration=get_video_duration(
                video.get("raw_path") or "input/raw_video.mp4"),
            cfg={**config, **config.get("refine", {})})
        kept, dropped = deduplicate_refined(refined)
        for dc, why in dropped:
            dc["refine_status"] = "rejected"
            dc["refine_reason"] = why[:500]
        persist_refinements(video_id, refined)
        run_repo.complete_stage(video_id, "refine")
    except Exception as e:
        run_repo.fail_stage(video_id, "refine", str(e))
        raise

    finals = render_clips(video_id)
    print(f"done: {len(finals)} finals")


if __name__ == "__main__":
    main()
