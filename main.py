import sys
import os
import json
import subprocess
import argparse
import time
import yaml
from logger import get_logger, log_stage, bind, setup_logging
from input.input_handler import handle_input
from campaign.campaign_parser import parse_campaign
from db.repositories import video_repo, run_repo, clip_repo
from pipeline.transcriber import transcribe_audio
from pipeline.chunker import build_chunks
from pipeline.embedder import embed_all_chunks
from pipeline.analyzer import analyze
from pipeline.similarity import find_related_segments
from pipeline.clipper import cut_clips
from pipeline.regen_srt import generate_srt_for_clip
from pipeline.fast_burn import burn_subtitles
from pipeline.improvements import add_fades, apply_logo, mix_music
from config import get_config, ffmpeg_path as _ffmpeg_bin

setup_logging()
log = get_logger("main")

CONFIG = get_config()
log.info("config loaded models=%s/%s ffmpeg=%s",
         CONFIG.get("ai", {}).get("llm_model"),
         CONFIG.get("embeddings", {}).get("model"),
         CONFIG.get("ffmpeg", {}).get("path"))

def load_template(template_name: str) -> dict:
    log.debug("load_template name=%s", template_name)
    path = f"campaign/templates/{template_name}.json"
    with open(path) as f:
        data = json.load(f)
    log.debug("load_template loaded keys=%s", list(data.keys()))
    return data

def merge_config(template: dict, campaign_config: dict) -> dict:
    merged = {**template, **{k: v for k, v in campaign_config.items() if v is not None}}
    log.debug("merge_config template_keys=%d override_keys=%d",
              len(template), len([k for k, v in campaign_config.items() if v is not None]))
    return merged

def scan_assets():
    logos_path = "assets/logos"
    music_path = "assets/music"
    logos = [f for f in os.listdir(logos_path) if not f.startswith('.')] if os.path.exists(logos_path) else []
    music = [f for f in os.listdir(music_path) if not f.startswith('.')] if os.path.exists(music_path) else []
    log.debug("scan_assets logos=%d music=%d", len(logos), len(music))
    return {"logos": logos, "music": music}

def to_sec(t):
    if isinstance(t, (int, float)): return float(t)
    if len(t.split(':')) == 2: t = '00:' + t
    p = t.split(':')
    return int(p[0])*3600 + int(p[1])*60 + float(p[2])

def should_skip_stage(video_id: str, stage: str) -> bool:
    """Return True if this stage already completed successfully for this video."""
    try:
        status = run_repo.get_stage_status(video_id, stage)
    except Exception:
        log.exception("should_skip_stage db_error", extra={"video_id": video_id, "stage": stage})
        return False
    if status == "done":
        log.info("stage skipped (already done)", extra={"video_id": video_id, "stage": stage})
        return True
    log.debug("stage not skipped status=%s", status, extra={"video_id": video_id, "stage": stage})
    return False

def render_clips(video_id: str, raw_video_path: str = None,
                 campaign_config: dict = None, progress=None):
    """Render stage (step 8 of the pipeline).

    Cut + stitch clips → 9:16 vertical → karaoke subtitles → logo →
    music → fades → output/clip_N_final.mp4, then output/report.json.

    Args:
        video_id:        UUID of the video row in the DB.
        raw_video_path:  Source video file. Resolved from videos.raw_path
                         in the DB when omitted.
        campaign_config: Campaign dict (subtitles/logo/music/fade flags).
                         Re-parsed to defaults (subtitles on, fade on) when
                         omitted — pass the original config to reproduce
                         branding exactly.
        progress:        Optional Gradio progress callback.

    Returns:
        List of {"file": final_path} dicts, one per rendered clip.
    """
    plog = bind("main", video_id=video_id)
    if raw_video_path is None:
        video = video_repo.get_video(video_id)
        if not video:
            raise ValueError(f"Unknown video_id: {video_id}")
        raw_video_path = video.get("raw_path")
        if not raw_video_path or not os.path.exists(raw_video_path):
            raw_video_path = "input/raw_video.mp4"
        plog.info("resolved raw path=%s", raw_video_path)
    if campaign_config is None:
        campaign_config = parse_campaign("", scan_assets())
        plog.info("using default campaign subtitles=%s fade=%s",
                  campaign_config.get("subtitles"), campaign_config.get("fade"))

    if progress: progress(0.8, desc="[8/8] Rendering clips...")
    final_clips = []
    if not should_skip_stage(video_id, "render"):
        run_repo.start_stage(video_id, "render")
        try:
            with log_stage("main", "render", video_id=video_id):
                clips = clip_repo.get_clips_for_video(video_id)
                plog.info("rendering %d clips", len(clips))

                # Attach LLM-confirmed continuations so the cutter stitches them
                confirmed = clip_repo.get_confirmed_segments_for_video(video_id)
                for c in clips:
                    c["related_segments"] = confirmed.get(c["id"], [])
                nstitched = sum(1 for c in clips if c["related_segments"])
                plog.info("clips with stitched continuations=%d", nstitched)

                # Load full transcript for SRT generation
                with open('transcripts/transcript.json', 'r') as f:
                    transcript_result = json.load(f)
                plog.debug("loaded transcript segments=%d", len(transcript_result.get("segments", [])))

                os.makedirs("clips", exist_ok=True)
                os.makedirs("output", exist_ok=True)

                ffmpeg_bin = _ffmpeg_bin()
                plog.debug("ffmpeg path=%s", ffmpeg_bin)

                cut_clip_results = cut_clips(raw_video_path, clips)
                plog.info("cut_clips produced=%d/%d", len(cut_clip_results), len(clips))
                for c in clips:
                    if c["related_segments"]:
                        clip_repo.mark_segments_stitched(c["id"])

                for i, clip in enumerate(clips):
                    n = clip['clip_number']
                    clog = bind("main", video_id=video_id, stage="render", clip_number=n)
                    # Resumable render: skip clips whose final already exists
                    final_path = f"output/clip_{n}_final.mp4"
                    if os.path.exists(final_path) and os.path.getsize(final_path) > 0:
                        clog.info("render clip already done, skipping final=%s", final_path)
                        clip_repo.update_output_path(clip["id"], final_path)
                        final_clips.append({'file': final_path})
                        continue
                    clog.info("render clip start=%s end=%s title=%.60s",
                              clip.get("start_time"), clip.get("end_time"),
                              clip.get("suggested_title", ""))
                    base_path = f"clips/clip_{n}"
                    clip_raw = f"{base_path}.mp4"
                    clip_vert = f"{base_path}_vertical.mp4"

                    # Check if clip_raw exists (cut_clips succeeded)
                    if not os.path.exists(clip_raw):
                        clog.warning("skip render, raw clip missing path=%s", clip_raw)
                        continue

                    clog.debug("verticalize src=%s dst=%s", clip_raw, clip_vert)
                    r = subprocess.run([
                        ffmpeg_bin, '-y', '-i', clip_raw,
                        '-filter_complex',
                        '[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,gblur=sigma=20[bg];[0:v]scale=1080:-1[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2[out]',
                        '-map', '[out]', '-map', '0:a',
                        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-c:a', 'aac',
                        clip_vert
                    ], capture_output=True)
                    if r.returncode != 0:
                        clog.error("verticalize failed rc=%d stderr=%.300s", r.returncode,
                                   (r.stderr or b"").decode(errors="replace"))

                    current_path = clip_vert
                    next_path = current_path

                    if campaign_config.get("subtitles", True):
                        next_path = f"{base_path}_subbed.mp4"
                        srt_path = f"clips/clip_{n}.srt"
                        clog.debug("generating SRT range=%s-%s", clip['start_time'], clip['end_time'])
                        srt = generate_srt_for_clip(transcript_result['segments'], to_sec(clip['start_time']), to_sec(clip['end_time']), n)
                        if srt.strip():
                            with open(srt_path, 'w') as f: f.write(srt)
                            clog.info("SRT written path=%s bytes=%d", srt_path, len(srt))
                            ok = burn_subtitles(n, clip_path=current_path, output_path=next_path)
                            clog.info("burn_subtitles ok=%s out=%s", ok, next_path)
                            if ok and os.path.exists(next_path):
                                current_path = next_path
                            else:
                                clog.warning("subtitle burn failed, continuing without burned subs")
                        else:
                            clog.warning("empty SRT, skipping burn")

                    if campaign_config.get("logo"):
                        next_path = f"{base_path}_logo.mp4"
                        logo_path = campaign_config["logo"]
                        if not os.path.isabs(logo_path) and not logo_path.startswith("assets/"):
                            logo_path = f"assets/logos/{logo_path}"
                        if os.path.exists(logo_path):
                            clog.info("apply_logo src=%s pos=%s", logo_path,
                                      campaign_config.get("logo_position", "top-right"))
                            current_path = apply_logo(current_path, next_path, logo_path, campaign_config.get("logo_position", "top-right"))
                        else:
                            clog.warning("logo missing path=%s, skipping", logo_path)

                    if campaign_config.get("music"):
                        next_path = f"{base_path}_music.mp4"
                        music_path = os.path.join(CONFIG.get('paths', {}).get('assets_dir', 'assets'), "music", campaign_config["music"])
                        if os.path.exists(music_path):
                            clog.info("mix_music src=%s vol=%s", music_path,
                                      campaign_config.get("music_volume", 0.1))
                            current_path = mix_music(current_path, next_path, music_path, campaign_config.get("music_volume", 0.1))
                        else:
                            clog.warning("music missing path=%s, skipping", music_path)

                    if campaign_config.get("fade", True):
                        next_path = f"output/clip_{n}_final.mp4"
                        clog.debug("add_fades in=%s out=%s", current_path, next_path)
                        faded = add_fades(current_path, next_path, fade_duration=1.0)
                        if not faded or not os.path.exists(faded):
                            clog.error("fades failed and no fallback file, skipping clip")
                            continue
                        next_path = faded
                    else:
                        next_path = f"output/clip_{n}_final.mp4"
                        clog.debug("fade disabled, moving %s -> %s", current_path, next_path)
                        if os.path.exists(current_path):
                            os.rename(current_path, next_path)

                    clip_repo.update_output_path(clip["id"], next_path)
                    final_clips.append({'file': next_path})
                    clog.info("render clip done final=%s", next_path)

                plog.info("render complete finals=%d", len(final_clips))
            run_repo.complete_stage(video_id, "render")
        except Exception as e:
            plog.exception("render failed: %s", e)
            run_repo.fail_stage(video_id, "render", str(e))
            raise
    else:
        plog.info("render skipped (already done)")
        clips = clip_repo.get_clips_for_video(video_id)
        final_clips = [{"file": c.get("output_path") or f"output/clip_{c['clip_number']}_final.mp4"} for c in clips]

    # Generate final report.json from DB query
    with log_stage("main", "report", video_id=video_id):
        clips = clip_repo.get_clips_for_video(video_id)
        report = []
        for c in clips:
            report.append({
                "clip_number": c["clip_number"],
                "hook": c["hook"],
                "start_time": c["start_time"],
                "end_time": c["end_time"],
                "duration": c["duration_seconds"],
                "reason": c["reason"],
                "suggested_title": c["suggested_title"],
                "output_file": c["output_path"]
            })
        with open("output/report.json", "w") as f:
            json.dump(report, f, indent=2)
    plog.info("report written clips=%d path=output/report.json", len(report))
    return final_clips

def run_pipeline(video_input: str, campaign_description: str = None, template: str = None, progress=None):
    t0 = time.monotonic()
    log.info("=== PIPELINE START input=%.120s template=%s ===", video_input, template)
    # Step 1: Normalize input to local file
    if progress: progress(0.05, desc="[1/8] Normalizing input...")
    with log_stage("main", "input"):
        raw_video_path = handle_input(video_input)
    log.info("input normalized path=%s", raw_video_path)

    # Step 2: Insert Video into DB
    video_id = video_repo.insert_video(video_input, raw_video_path)
    plog = bind("main", video_id=video_id)
    plog.info("video registered in DB")
    if progress: progress(0.08, desc=f"Registered video {str(video_id)[:8]}")

    # Parse campaign
    if progress: progress(0.1, desc="[2/8] Parsing campaign...")
    with log_stage("main", "campaign", video_id=video_id):
        available_assets = scan_assets()
        campaign_config = parse_campaign(campaign_description or "", available_assets)
        if template or campaign_config.get("template"):
            tmpl_name = template or campaign_config["template"]
            if os.path.exists(f"campaign/templates/{tmpl_name}.json"):
                tmpl = load_template(tmpl_name)
                campaign_config = merge_config(tmpl, campaign_config)
                plog.info("template applied name=%s", tmpl_name)
            else:
                plog.warning("template not found name=%s", tmpl_name)
    plog.info("campaign parsed subtitles=%s logo=%s music=%s fade=%s",
              campaign_config.get("subtitles"), campaign_config.get("logo"),
              campaign_config.get("music"), campaign_config.get("fade"))

    # Step 3: Transcribe
    if progress: progress(0.2, desc="[3/8] Transcribing...")
    if not should_skip_stage(video_id, "transcribe"):
        run_repo.start_stage(video_id, "transcribe")
        try:
            with log_stage("main", "transcribe", video_id=video_id):
                # Extract audio first if transcriber needs it
                audio_path = "downloads/audio.wav"
                os.makedirs("downloads", exist_ok=True)
                ffmpeg_bin = _ffmpeg_bin()
                if not os.path.exists(audio_path):
                    plog.info("extracting audio src=%s dst=%s", raw_video_path, audio_path)
                    r = subprocess.run([ffmpeg_bin, "-i", raw_video_path, "-q:a", "0", "-map", "a", audio_path, "-y"], capture_output=True)
                    if r.returncode != 0:
                        plog.warning("audio extract rc=%d stderr=%.300s", r.returncode, (r.stderr or b"").decode(errors="replace"))
                    else:
                        plog.debug("audio extracted")
                else:
                    plog.info("reusing cached audio path=%s", audio_path)

                transcript_path, transcript_result = transcribe_audio(audio_path)
                nseg = len((transcript_result or {}).get("segments", [])) if isinstance(transcript_result, dict) else -1
                plog.info("transcribed segments=%d path=%s", nseg, transcript_path)
            run_repo.complete_stage(video_id, "transcribe")
        except Exception as e:
            plog.exception("transcribe failed: %s", e)
            run_repo.fail_stage(video_id, "transcribe", str(e))
            raise
    else:
        transcript_path = "transcripts/transcript.json"
        plog.info("transcribe skipped, using cached path=%s", transcript_path)

    # Step 4: Chunk
    if progress: progress(0.4, desc="[4/8] Chunking...")
    if not should_skip_stage(video_id, "chunk"):
        run_repo.start_stage(video_id, "chunk")
        try:
            with log_stage("main", "chunk", video_id=video_id):
                chunks = build_chunks(transcript_path, video_id)
                plog.info("chunked count=%d", len(chunks))
            run_repo.complete_stage(video_id, "chunk")
        except Exception as e:
            plog.exception("chunk failed: %s", e)
            run_repo.fail_stage(video_id, "chunk", str(e))
            raise

    # Step 5: Embed
    if progress: progress(0.5, desc="[5/8] Embedding chunks...")
    if not should_skip_stage(video_id, "embed"):
        run_repo.start_stage(video_id, "embed")
        try:
            with log_stage("main", "embed", video_id=video_id):
                from db.repositories import chunk_repo
                chunks = chunk_repo.get_chunks_for_video(video_id)
                plog.info("embedding %d chunks", len(chunks))
                embed_all_chunks(video_id, chunks)
            run_repo.complete_stage(video_id, "embed")
        except Exception as e:
            plog.exception("embed failed: %s", e)
            run_repo.fail_stage(video_id, "embed", str(e))
            raise

    # Step 6: Analyze
    if progress: progress(0.6, desc="[6/8] Analyzing for clips...")
    if not should_skip_stage(video_id, "analyze"):
        run_repo.start_stage(video_id, "analyze")
        try:
            with log_stage("main", "analyze", video_id=video_id):
                from db.repositories import chunk_repo
                chunks = chunk_repo.get_chunks_for_video(video_id)
                plog.info("analyzing %d chunks", len(chunks))
                clips = analyze(video_id, chunks, CONFIG)
                plog.info("analyze found %d clips", len(clips))
            run_repo.complete_stage(video_id, "analyze")
        except Exception as e:
            plog.exception("analyze failed: %s", e)
            run_repo.fail_stage(video_id, "analyze", str(e))
            raise

    # Step 7: Similarity
    if progress: progress(0.7, desc="[7/8] Finding related segments...")
    if not should_skip_stage(video_id, "similarity"):
        run_repo.start_stage(video_id, "similarity")
        try:
            with log_stage("main", "similarity", video_id=video_id):
                clips = clip_repo.get_clips_for_video(video_id)
                find_related_segments(video_id, clips, top_k=CONFIG['similarity']['top_k'], threshold=CONFIG['similarity']['threshold'])
                nrel = sum(len(c.get("related_segments", [])) for c in clips)
                plog.info("similarity done clips=%d confirmed_segments=%d top_k=%s threshold=%s",
                          len(clips), nrel, CONFIG['similarity']['top_k'], CONFIG['similarity']['threshold'])
            run_repo.complete_stage(video_id, "similarity")
        except Exception as e:
            plog.exception("similarity failed: %s", e)
            run_repo.fail_stage(video_id, "similarity", str(e))
            raise

    # Step 8: Render (extracted - see render_clips())
    final_clips = render_clips(video_id, raw_video_path, campaign_config, progress)

    elapsed = time.monotonic() - t0
    plog.info("=== PIPELINE COMPLETE clips=%d elapsed=%.1fs report=output/report.json ===",
              len(final_clips), elapsed)
    log.info("=== PIPELINE COMPLETE video=%.8s clips=%d elapsed=%.1fs ===",
             video_id, len(final_clips), elapsed)

    return final_clips

def main():
    parser = argparse.ArgumentParser(description="Viral Clips Automator")
    parser.add_argument("video_input", nargs='?', default=None, help="URL or local path to the video")
    parser.add_argument("--campaign", help="Natural language description of the campaign", default="")
    parser.add_argument("--template", help="Name of the campaign template to use", default=None)
    parser.add_argument("--log-level", help="Override log level (DEBUG/INFO/WARNING/ERROR)", default=None)

    args = parser.parse_args()
    if args.log_level:
        setup_logging(level=args.log_level.upper(), force=True)
        log.info("log level overridden to %s", args.log_level.upper())
    if not args.video_input:
        log.warning("no video_input provided")
        print("Usage: python main.py <video_path>")
    else:
        log.info("CLI start input=%.120s campaign=%.120s template=%s",
                 args.video_input, args.campaign, args.template)
        try:
            run_pipeline(args.video_input, args.campaign, args.template)
        except Exception:
            log.exception("CLI pipeline failed")
            raise

if __name__ == "__main__":
    main()
