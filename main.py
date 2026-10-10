import sys
import os
import json
import shutil
import subprocess
import argparse
import time
import yaml
from logger import get_logger, log_stage, bind, setup_logging, run_log
from pipeline import workspace
from input.input_handler import handle_input
from campaign.campaign_parser import parse_campaign
from db.repositories import video_repo, run_repo, clip_repo
from pipeline.transcriber import transcribe_audio
from pipeline.chunker import build_chunks
from pipeline.embedder import embed_all_chunks
from pipeline.analyzer import analyze
from pipeline.similarity import find_related_segments
from pipeline.clipper import cut_clips, get_time_ranges_for_clip
from pipeline.regen_srt import generate_srt_for_clip, generate_srt_for_ranges
from pipeline.fast_burn import burn_subtitles
from pipeline.improvements import add_fades, apply_logo, mix_music
from pipeline.framing import validate_layout, render_vertical
from pipeline.fingerprints import outline_fingerprint, transcript_signature
from config import get_config, ffmpeg_path as _ffmpeg_bin
from campaign.campaign_parser import validate_campaign_config

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


def invalidate_stage(video_id: str, stage: str) -> None:
    """Force a stage to rerun (e.g. render after boundaries/framing changed)."""
    try:
        from db.connection import get_conn, release_conn
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM pipeline_runs WHERE video_id=%s AND stage=%s",
                            (video_id, stage))
                conn.commit()
        finally:
            release_conn(conn)
        log.info("stage invalidated video=%.8s stage=%s", video_id, stage)
    except Exception:
        log.exception("invalidate_stage failed video=%.8s stage=%s", video_id, stage)


STAGES = ("transcribe", "chunk", "audio_events", "embed", "outline",
          "analyze", "similarity", "refine", "render")


def reset_from_stage(video_id: str, stage: str) -> None:
    """Redo `stage` and every later stage for an existing video.

    Clears the checkpoints plus the derived rows those stages would otherwise
    duplicate or reuse (chunks, audio events, outlines, candidates, clips,
    related segments) and the cache fingerprints that gate them. Earlier
    stages (e.g. the transcript) are kept.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; choose from {', '.join(STAGES)}")
    idx = STAGES.index(stage)
    redo = list(STAGES[idx:])
    fps = []
    if idx <= STAGES.index("audio_events"):
        fps.append("audio_events")
    if idx <= STAGES.index("outline"):
        fps.append("outline")
    if idx <= STAGES.index("analyze"):
        fps.append("discovery")
    from db.connection import get_conn, release_conn
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            if idx <= STAGES.index("analyze"):
                # clips cascade to related_segments + edit_plans
                cur.execute("DELETE FROM clips WHERE video_id=%s", (video_id,))
                cur.execute("DELETE FROM candidates WHERE video_id=%s AND source <> 'audio_event'",
                            (video_id,))
            elif stage == "similarity":
                cur.execute("""DELETE FROM related_segments
                               WHERE clip_id IN (SELECT id FROM clips WHERE video_id=%s)""",
                            (video_id,))
            if idx <= STAGES.index("outline"):
                cur.execute("DELETE FROM outlines WHERE video_id=%s", (video_id,))
            if idx <= STAGES.index("audio_events"):
                cur.execute("DELETE FROM audio_events WHERE video_id=%s", (video_id,))
                cur.execute("DELETE FROM candidates WHERE video_id=%s AND source = 'audio_event'",
                            (video_id,))
            if idx <= STAGES.index("chunk"):
                cur.execute("DELETE FROM chunks WHERE video_id=%s", (video_id,))
            if fps:
                cur.execute("DELETE FROM stage_fingerprints WHERE video_id=%s AND stage = ANY(%s)",
                            (video_id, fps))
            cur.execute("DELETE FROM pipeline_runs WHERE video_id=%s AND stage = ANY(%s)",
                        (video_id, redo))
            conn.commit()
    finally:
        release_conn(conn)
    log.info("reset video=%.8s from stage=%s (redo %s)", video_id, stage, ",".join(redo))


def _keep_awake():
    """macOS: hold an idle-sleep assertion while a run is in progress.

    A sleeping Mac pauses the pipeline (one run lost ~15 minutes this way).
    caffeinate -w also exits on its own if this process dies. Disable with
    runtime.keep_awake: false.
    """
    if sys.platform != "darwin" or not CONFIG.get("runtime", {}).get("keep_awake", True):
        return None
    exe = shutil.which("caffeinate")
    if not exe:
        return None
    try:
        return subprocess.Popen([exe, "-i", "-w", str(os.getpid())],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        log.debug("caffeinate unavailable", exc_info=True)
        return None


def get_video_duration(path: str) -> float | None:
    """ffprobe duration in seconds, or None when unavailable."""
    try:
        from config import ffprobe_path
        r = subprocess.run([ffprobe_path(), "-v", "error", "-show_entries",
                            "format=duration", "-of",
                            "default=noprint_wrappers=1:nokey=1", path],
                           capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return float(r.stdout.strip())
    except Exception:
        pass
    return None


def resolve_layout(campaign_config: dict | None, cli_layout: str | None = None) -> str:
    """CLI override > campaign layout > framing.default_layout, validated."""
    if cli_layout:
        return validate_layout(cli_layout)
    if campaign_config and campaign_config.get("layout"):
        return validate_layout(campaign_config["layout"])
    try:
        return validate_layout(CONFIG.get("framing", {}).get("default_layout", "auto"))
    except Exception:
        return "auto"

def select_renderable_clips(clips: list) -> tuple[list, int]:
    """Split clips into renderable vs boundary-rejected.

    Rejected candidates are never rendered nor marked successful; their
    reason is preserved in the DB/report. Returns (renderable, n_rejected).
    """
    renderable = [c for c in clips if c.get("refine_status") != "rejected"]
    return renderable, len(clips) - len(renderable)

def _render_legacy_chain(clog, base_path, n, current_path, srt,
                         logo_file, music_file, campaign_config,
                         fades_cfg, final_path):
    """Legacy multi-pass chain (drawtext -> logo -> music -> fades).

    Used when export.consolidated is false or polish fails. Returns the
    final path, or "" when rendering failed (caller skips the clip).
    """
    next_path = current_path
    if srt:
        next_path = f"{base_path}_subbed.mp4"
        ok = burn_subtitles(n, clip_path=current_path, output_path=next_path,
                            srt_path=f"{base_path}.srt")
        clog.info("burn_subtitles ok=%s out=%s", ok, next_path)
        if ok and os.path.exists(next_path):
            current_path = next_path
        else:
            clog.warning("subtitle burn failed, continuing without burned subs")
            next_path = current_path
    if logo_file:
        _lp = f"{base_path}_logo.mp4"
        clog.info("apply_logo src=%s pos=%s", logo_file,
                  campaign_config.get("logo_position", "top-right"))
        current_path = apply_logo(current_path, _lp, logo_file,
                                  campaign_config.get("logo_position", "top-right"))
        next_path = current_path
    if music_file:
        _mp = f"{base_path}_music.mp4"
        clog.info("mix_music src=%s vol=%s", music_file,
                  campaign_config.get("music_volume", 0.1))
        current_path = mix_music(current_path, _mp, music_file,
                                 campaign_config.get("music_volume", 0.1))
        next_path = current_path
    if campaign_config.get("fade", True):
        fade_dur = float(fades_cfg.get("duration",
                                       CONFIG.get("defaults", {}).get("fade_duration", 0.5)))
        audio_fade = bool(fades_cfg.get("audio", False))
        faded = add_fades(current_path, final_path,
                          fade_duration=fade_dur, audio=audio_fade)
        if not faded or not os.path.exists(faded):
            clog.error("fades failed and no fallback file, skipping clip")
            return ""
        return faded
    if os.path.exists(current_path):
        try:
            if os.path.exists(final_path):
                os.remove(final_path)
        except OSError:
            pass
        os.rename(current_path, final_path)
        return final_path
    return ""


def render_clips(video_id: str, raw_video_path: str = None,
                 campaign_config: dict = None, progress=None, layout: str = None,
                 clip_numbers: list = None):
    """Render stage (step 9 of the pipeline).

    Cut + stitch clips → full-screen 9:16 vertical crop (auto/speaker_crop/
    stacked_split/center_crop; letterbox-free under full_screen_vertical) →
    karaoke subtitles → logo → music → fades (speech fade-out off by default) →
    output/<video>/clip_N_final.mp4, then output/<video>/report.json
    (see pipeline/workspace.py).

    Args:
        video_id:        UUID of the video row in the DB.
        raw_video_path:  Source video file. Resolved from videos.raw_path
                         in the DB when omitted.
        campaign_config: Campaign dict (subtitles/logo/music/fade/layout flags).
                         Re-parsed to defaults (subtitles on, fade on) when
                         omitted — pass the original config to reproduce
                         branding exactly.
        progress:        Optional Gradio progress callback.
        layout:          Manual layout override (CLI). Wins over campaign +
                         config defaults; validated against SUPPORTED_LAYOUTS.

    Returns:
        List of {"file": final_path} dicts, one per rendered clip.
    """
    if clip_numbers is not None:
        clip_numbers = {int(n) for n in clip_numbers}
    plog = bind("main", video_id=video_id)
    ws = workspace.current()
    if ws.video_id != str(video_id):
        ws = workspace.activate(video_id)  # e.g. Review tab re-render
    if clip_numbers is not None:
        plog.info("selective re-render clips=%s", sorted(clip_numbers))
    if raw_video_path is None:
        video = video_repo.get_video(video_id)
        if not video:
            raise ValueError(f"Unknown video_id: {video_id}")
        raw_video_path = video.get("raw_path")
        if not raw_video_path or not os.path.exists(raw_video_path):
            raise FileNotFoundError(
                f"source video for {video_id} not found at {raw_video_path!r}")
        plog.info("resolved raw path=%s", raw_video_path)
    if campaign_config is None:
        campaign_config = parse_campaign("", scan_assets())
        plog.info("using default campaign subtitles=%s fade=%s",
                  campaign_config.get("subtitles"), campaign_config.get("fade"))
    try:
        campaign_config = validate_campaign_config(dict(campaign_config))
    except Exception:
        pass
    layout_mode = resolve_layout(campaign_config, layout)
    framing_cfg = dict(CONFIG.get("framing", {}))
    fades_cfg = dict(CONFIG.get("fades", {}))
    plog.info("render layout=%s subtitles=%s fade=%s audio_fade=%s",
              layout_mode, campaign_config.get("subtitles"),
              campaign_config.get("fade"), fades_cfg.get("audio", False))

    if progress: progress(0.85, desc="[11/11] Rendering clips...")
    final_clips = []
    if not should_skip_stage(video_id, "render"):
        # Only a render that crashed part-way may keep the finals it already
        # produced; a fresh render (new boundaries, framing changes, review
        # re-render) must never reuse stale output files.
        resuming_render = (clip_numbers is None and
                           run_repo.get_stage_status(video_id, "render") in ("running", "failed"))
        run_repo.start_stage(video_id, "render")
        try:
            with log_stage("main", "render", video_id=video_id):
                clips = clip_repo.get_clips_for_video(video_id)
                plog.info("rendering %d clips", len(clips))

                # Attach LLM-confirmed continuations so the cutter stitches them
                confirmed = clip_repo.get_confirmed_segments_for_video(video_id)
                for c in clips:
                    c["related_segments"] = confirmed.get(c["id"], [])
                nlinked = sum(1 for c in clips if c["related_segments"])
                nstitched = sum(1 for c in clips if len(get_time_ranges_for_clip(c)) > 1)
                plog.info("clips with linked continuations=%d, rendered as multi-part=%d",
                          nlinked, nstitched)

                # Load full transcript for SRT generation
                with open(ws.transcript, 'r') as f:
                    transcript_result = json.load(f)
                plog.debug("loaded transcript segments=%d", len(transcript_result.get("segments", [])))

                ws.ensure()

                ffmpeg_bin = _ffmpeg_bin()
                plog.debug("ffmpeg path=%s", ffmpeg_bin)

                # Skip boundary-rejected candidates (recorded reason kept in report)
                clips, nrejected = select_renderable_clips(clips)
                if nrejected:
                    plog.info("skipping %d boundary-rejected clips", nrejected)
                if clip_numbers is not None:
                    clips = [c for c in clips if c["clip_number"] in clip_numbers]
                    plog.info("selective subset clips=%d", len(clips))

                cut_clip_results = cut_clips(raw_video_path, clips, output_dir=ws.clips_dir)
                plog.info("cut_clips produced=%d/%d", len(cut_clip_results), len(clips))
                for c in clips:
                    if c["related_segments"]:
                        clip_repo.mark_segments_stitched(c["id"])

                for i, clip in enumerate(clips):
                    n = clip['clip_number']
                    clog = bind("main", video_id=video_id, stage="render", clip_number=n)
                    final_path = ws.final(n)
                    if os.path.exists(final_path) and os.path.getsize(final_path) > 0:
                        if resuming_render:
                            # crash resume: this clip finished before the crash
                            clog.info("render clip already done, skipping final=%s", final_path)
                            clip_repo.update_output_path(clip["id"], final_path)
                            final_clips.append({'file': final_path})
                            continue
                        os.remove(final_path)  # stale output from an earlier render
                    ranges = get_time_ranges_for_clip(clip)
                    clog.info("render clip ranges=%s title=%.60s",
                              [(round(a, 1), round(b, 1)) for a, b in ranges],
                              clip.get("suggested_title", ""))
                    base_path = os.path.join(ws.clips_dir, f"clip_{n}")
                    clip_raw = f"{base_path}.mp4"
                    clip_vert = f"{base_path}_vertical.mp4"

                    # Check if clip_raw exists (cut_clips succeeded)
                    if not os.path.exists(clip_raw):
                        clog.warning("skip render, raw clip missing path=%s", clip_raw)
                        continue

                    clip_layout = validate_layout(
                        clip.get("layout") or layout_mode)
                    clog.debug("framing src=%s dst=%s layout=%s", clip_raw,
                               clip_vert, clip_layout)
                    # Manual crop ROIs (edit plan) are authoritative at render
                    manual_rois = []
                    try:
                        from pipeline.review import get_manual_rois
                        manual_rois = get_manual_rois(clip["id"])
                        if manual_rois:
                            clog.info("manual ROIs active n=%d", len(manual_rois))
                    except Exception:
                        clog.debug("manual ROI load skipped", exc_info=True)
                    frame_res = {}
                    try:
                        frame_res = render_vertical(clip_raw, clip_vert,
                                                    layout=clip_layout,
                                                    cfg=framing_cfg,
                                                    manual_rois=manual_rois or None)
                        clip["layout"] = frame_res.get("layout", clip_layout)
                        clip["layout_reason"] = frame_res.get("layout_reason", "")
                        kinds = frame_res.get("anchor_kinds") or {}
                        if kinds:
                            total = sum(kinds.values()) or 1
                            cov = ", ".join(
                                f"{k} {v / total:.0%}"
                                for k, v in sorted(kinds.items(),
                                                   key=lambda kv: -kv[1]))
                            clip["layout_reason"] = (
                                f"{clip['layout_reason']} | frame anchors: {cov}")
                        clip["manual_rois"] = frame_res.get("manual_rois", [])
                        clog.info("framing layout=%s reason=%.120s",
                                  clip["layout"], clip["layout_reason"])
                    except Exception:
                        clog.exception("framing failed, center-crop fallback (no bars)")
                        try:
                            from pipeline.framing import render_center_crop
                            render_center_crop(clip_raw, clip_vert)
                        except Exception:
                            clog.error("center-crop fallback failed, skipping clip")
                            continue
                        clip["layout"] = "center_crop"
                        clip["layout_reason"] = "framing exception; static centre crop"

                    current_path = clip_vert
                    next_path = current_path
                    final_path = ws.final(n)

                    # --- captions data (shared by polish + legacy paths) ---
                    srt, cap_data = "", None
                    if campaign_config.get("subtitles", True):
                        srt_path = f"{base_path}.srt"
                        clog.debug("generating SRT ranges=%s", ranges)
                        srt, tl_map = generate_srt_for_ranges(
                            transcript_result['segments'], ranges, n)
                        clip["timeline"] = tl_map
                        if srt.strip():
                            with open(srt_path, 'w', encoding="utf-8") as f:
                                f.write(srt)
                            clog.info("SRT written path=%s bytes=%d", srt_path, len(srt))
                        else:
                            clog.warning("empty SRT, skipping burn")
                            srt = ""
                        if srt:
                            try:
                                from pipeline.captions import build_captions
                                cap_data = build_captions(
                                    transcript_result['segments'], ranges,
                                    dict(CONFIG.get("captions", {})),
                                    frame_res.get("analysis"))
                                clip["caption_placement"] = cap_data["placement"]
                                clip["caption_cues"] = len(cap_data["cues"])
                            except Exception:
                                clog.exception("caption build failed")
                                cap_data = None

                    # --- resolve branding assets ---
                    logo_file = None
                    if campaign_config.get("logo"):
                        _lp = campaign_config["logo"]
                        if not os.path.isabs(_lp) and not _lp.startswith("assets/"):
                            _lp = f"assets/logos/{_lp}"
                        if os.path.exists(_lp):
                            logo_file = _lp
                        else:
                            clog.warning("logo missing path=%s, skipping", _lp)
                    music_file = None
                    if campaign_config.get("music"):
                        _mp = os.path.join(CONFIG.get('paths', {}).get('assets_dir', 'assets'),
                                           "music", campaign_config["music"])
                        if os.path.exists(_mp):
                            music_file = _mp
                        else:
                            clog.warning("music missing path=%s, skipping", _mp)

                    # --- structured edit plan (preview + export share it) ---
                    from pipeline.editplan import (build_edit_plan, persist_plan)
                    try:
                        _src = frame_res.get("analysis", {}) if frame_res else {}
                        _plan = build_edit_plan(
                            clip, campaign_config, CONFIG,
                            framing_analysis=_src or None,
                            caption_data=cap_data,
                            source_info={"width": _src.get("width"),
                                         "height": _src.get("height")},
                            framing_plan=frame_res or None)
                        persist_plan(video_id, clip["id"], n, _plan)
                    except Exception:
                        clog.exception("edit plan persist failed")
                        _plan = None

                    consolidated = bool(CONFIG.get("export", {}).get("consolidated", True))
                    polished = False
                    if consolidated and _plan is not None:
                        try:
                            from pipeline.polish import polish_clip
                            out = polish_clip(
                                clip_vert, final_path, _plan,
                                campaign_paths={"logo": logo_file,
                                                "music": music_file})
                            if out and os.path.exists(out):
                                next_path = out
                                polished = True
                                clog.info("polish done final=%s", next_path)
                            else:
                                clog.warning("polish failed, legacy chain fallback")
                        except Exception:
                            clog.exception("polish failed, legacy chain fallback")

                    if not polished:
                        next_path = _render_legacy_chain(
                            clog, base_path, n, current_path, srt,
                            logo_file, music_file, campaign_config,
                            fades_cfg, final_path)
                        if not next_path:
                            continue

                    clip_repo.update_output_path(clip["id"], next_path)
                    try:
                        from db.repositories.clip_repo import update_refinement as _upd
                        _upd(clip["id"],
                             float(ranges[0][0]), float(ranges[-1][1]),
                             clip.get("refine_status", "refined"),
                             clip.get("refine_reason", ""),
                             [[float(a), float(b)] for a, b in ranges],
                             clip.get("timeline"),
                             clip.get("layout", layout_mode),
                             clip.get("layout_reason", ""))
                    except Exception:
                        clog.debug("layout/refine persist skipped", exc_info=True)
                    final_clips.append({'file': next_path})
                    clog.info("render clip done final=%s", next_path)
                    if not CONFIG.get("export", {}).get("keep_intermediates", False):
                        # cut + vertical + legacy-chain files are ~60MB per clip;
                        # SRT, edit plan and debug video are kept. Review re-renders
                        # re-cut from the source, so nothing is lost.
                        for _f in (clip_raw, clip_vert, f"{base_path}_subbed.mp4",
                                   f"{base_path}_logo.mp4", f"{base_path}_music.mp4"):
                            try:
                                if os.path.exists(_f) and os.path.abspath(_f) != os.path.abspath(next_path):
                                    os.remove(_f)
                            except OSError:
                                clog.debug("could not remove intermediate %s", _f)

                plog.info("render complete finals=%d", len(final_clips))
            run_repo.complete_stage(video_id, "render")
        except Exception as e:
            plog.exception("render failed: %s", e)
            run_repo.fail_stage(video_id, "render", str(e))
            raise
    else:
        plog.info("render skipped (already done)")
        clips, _ = select_renderable_clips(clip_repo.get_clips_for_video(video_id))
        final_clips = []
        for c in clips:
            f = c.get("output_path") or ws.final(c['clip_number'])
            if os.path.exists(f):
                final_clips.append({"file": f})

    # Generate final report.json from DB query
    with log_stage("main", "report", video_id=video_id):
        clips = clip_repo.get_clips_for_video(video_id)
        report = []
        for c in clips:
            ranges = c.get("source_ranges") or [[c["start_time"], c["end_time"]]]
            try:
                out_dur = round(sum(float(b) - float(a) for a, b in ranges), 3)
            except (TypeError, ValueError):
                out_dur = c.get("duration_seconds")
            report.append({
                "clip_number": c["clip_number"],
                "hook": c["hook"],
                "start_time": c.get("refined_start_time", c["start_time"])
                if isinstance(c.get("refined_start_time"), (int, float))
                else c["start_time"],
                "end_time": c.get("refined_end_time", c["end_time"])
                if isinstance(c.get("refined_end_time"), (int, float))
                else c["end_time"],
                "duration": c["duration_seconds"],
                "output_duration": c.get("output_duration", out_dur),
                "source_ranges": c.get("source_ranges", ranges),
                "timeline": c.get("timeline", []),
                "layout": c.get("layout", layout_mode if 'layout_mode' in dir() else "auto"),
                "layout_reason": c.get("layout_reason", ""),
                "refine_status": c.get("refine_status", "unknown"),
                "refine_reason": c.get("refine_reason", ""),
                "reason": c["reason"],
                "suggested_title": c["suggested_title"],
                "output_file": c["output_path"]
            })
        ws.ensure()
        with open(ws.report, "w") as f:
            json.dump(report, f, indent=2)
    plog.info("report written clips=%d path=%s", len(report), ws.report)
    return final_clips

def run_pipeline(video_input: str, campaign_description: str = None, template: str = None, progress=None,
                 layout: str = None, resume: bool = True, from_stage: str = None,
                 video_id: str = None, campaign_config: dict = None, render: bool = True,
                 info: dict = None):
    """Run the full pipeline for one source.

    resume:     reuse the latest video row for the same source, skipping
                stages that already finished (False = brand-new run).
    from_stage: redo this stage and everything after it (see STAGES).
    video_id:   continue a specific existing video (wins over resume).
    campaign_config: ready render flags (subtitles/logo/music/fade/layout);
                skips brief parsing — campaign_description is then only the
                analysis focus (campaign clip mode).
    render:     False stops after boundary refinement (the caller renders).
    info:       optional dict filled with video_id / raw_path / workspace.

    Each run also writes logs/runs/<timestamp>_<source>.log and, on macOS,
    keeps the machine awake until it finishes.
    """
    awake = _keep_awake()
    try:
        with run_log(workspace.source_key(video_input)) as run_log_path:
            log.info("run log file=%s", run_log_path)
            return _run_pipeline_impl(video_input, campaign_description, template, progress,
                                      layout, resume, from_stage, video_id,
                                      campaign_config=campaign_config, render=render,
                                      info=info)
    finally:
        if awake is not None:
            try:
                awake.terminate()
            except Exception:
                pass


def _run_pipeline_impl(video_input, campaign_description, template, progress,
                       layout, resume, from_stage, video_id, campaign_config=None,
                       render=True, info=None):
    t0 = time.monotonic()
    log.info("=== PIPELINE START input=%.120s template=%s layout=%s ===", video_input, template, layout)
    if layout:
        layout = validate_layout(layout)
    # Step 1: Normalize input to local file
    if progress: progress(0.05, desc="[1/10] Normalizing input...")
    with log_stage("main", "input"):
        raw_video_path = handle_input(video_input)
    log.info("input normalized path=%s", raw_video_path)

    # Step 2: Register the video, or resume the previous run of this source
    if video_id is None and resume:
        prev = video_repo.find_latest_by_source(video_input)
        if prev:
            video_id = prev["id"]
            log.info("resuming video=%.8s for this source (finished stages are "
                     "skipped; use --fresh to start over)", video_id)
    if video_id is None:
        video_id = video_repo.insert_video(video_input, raw_video_path)
        log.info("video registered in DB id=%.8s", video_id)
    elif (video_repo.get_video(video_id) or {}).get("raw_path") != raw_video_path:
        video_repo.update_raw_path(video_id, raw_video_path)
    plog = bind("main", video_id=video_id)
    ws = workspace.activate(video_id)
    if info is not None:
        info.update(video_id=str(video_id), raw_path=raw_video_path, workspace=ws)
    if from_stage:
        reset_from_stage(video_id, from_stage)
    if (run_repo.get_stage_status(video_id, "transcribe") == "done"
            and not os.path.exists(ws.transcript)):
        plog.warning("transcript missing from workspace %s; redoing from transcribe",
                     ws.work_dir)
        reset_from_stage(video_id, "transcribe")
    if progress: progress(0.08, desc=f"Registered video {str(video_id)[:8]}")

    # Parse campaign
    if progress: progress(0.1, desc="[2/10] Parsing campaign...")
    with log_stage("main", "campaign", video_id=video_id):
        if campaign_config is not None:
            campaign_config = dict(campaign_config)
            plog.info("campaign config supplied by caller; brief parsing skipped")
        else:
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
        if layout:
            campaign_config["layout"] = layout
            campaign_config["_layout_explicit"] = True
    plog.info("campaign parsed subtitles=%s logo=%s music=%s fade=%s layout=%s",
              campaign_config.get("subtitles"), campaign_config.get("logo"),
              campaign_config.get("music"), campaign_config.get("fade"),
              campaign_config.get("layout"))

    # Step 3: Transcribe
    if progress: progress(0.2, desc="[3/10] Transcribing...")
    if not should_skip_stage(video_id, "transcribe"):
        run_repo.start_stage(video_id, "transcribe")
        try:
            with log_stage("main", "transcribe", video_id=video_id):
                # Extract audio first if transcriber needs it
                audio_path = ws.audio  # per video: never another video's audio
                os.makedirs(ws.work_dir, exist_ok=True)
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

                transcript_path, transcript_result = transcribe_audio(
                    audio_path, output_dir=ws.transcripts_dir)
                nseg = len((transcript_result or {}).get("segments", [])) if isinstance(transcript_result, dict) else -1
                plog.info("transcribed segments=%d path=%s", nseg, transcript_path)
            run_repo.complete_stage(video_id, "transcribe")
        except Exception as e:
            plog.exception("transcribe failed: %s", e)
            run_repo.fail_stage(video_id, "transcribe", str(e))
            raise
    else:
        transcript_path = ws.transcript
        plog.info("transcribe skipped, using cached path=%s", transcript_path)

    # Step 4: Chunk
    if progress: progress(0.4, desc="[4/10] Chunking...")
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

    # Step 5: Audio events (spikes on source audio; fingerprint-gated)
    if progress: progress(0.45, desc="[5/11] Listening for audio events...")
    _afp_now = None
    try:
        import os as _os2
        _wav_probe = (ws.audio if _os2.path.exists(ws.audio) else raw_video_path)
        _d = get_video_duration(_wav_probe) or 0
        _asig = (f"{_wav_probe}:{round(_d, 1)}:"
                 f"{_os2.path.getsize(_wav_probe)}")
        from pipeline.fingerprints import audio_events_fingerprint as _aef
        _afp_now = _aef(_asig, CONFIG)
    except Exception:
        pass
    if (_afp_now and run_repo.get_stage_status(video_id, "audio_events") == "done"
            and not run_repo.fingerprint_matches(video_id, "audio_events", _afp_now)):
        plog.info("audio fingerprint changed, invalidating audio_events+downstream")
        for _st in ("audio_events", "analyze", "similarity", "refine", "render"):
            invalidate_stage(video_id, _st)
    if not should_skip_stage(video_id, "audio_events"):
        run_repo.start_stage(video_id, "audio_events")
        try:
            with log_stage("main", "audio_events", video_id=video_id):
                from pipeline.audio_events import (
                    ensure_source_audio, detect_audio_events, propose_from_events,
                    enrich_event_candidates, persist_events)
                from pipeline.audio_classifier import classify_events
                from pipeline.boundaries import load_words, build_sentences
                from db.repositories import candidate_repo
                a_cfg = dict(CONFIG.get("audio_events", {}))
                if a_cfg.get("enabled", True):
                    wav = ensure_source_audio(raw_video_path, ws.audio)
                    events, _ = detect_audio_events(wav, CONFIG)
                    # optional labels (disabled by default; never faked)
                    try:
                        import librosa as _lib
                        _y, _sr = _lib.load(wav, sr=16000, mono=True)
                    except Exception:
                        _y, _sr = None, 16000
                    events, _cstat = classify_events(events, _y, _sr, CONFIG)
                    persist_events(video_id, events)
                    _words = load_words(ws.transcript)
                    _sents = build_sentences(_words)
                    cues = propose_from_events(events, _words, _sents, CONFIG)
                    cues = enrich_event_candidates(cues, CONFIG)
                    candidate_repo.clear_candidates(video_id, source="audio_event")
                    for _c in cues:
                        candidate_repo.insert_candidate(
                            video_id, "audio_event", _c["sentence_ids"],
                            _c["source_ranges"], hook=_c.get("hook") or None,
                            main_idea=_c.get("main_idea") or None,
                            payoff=_c.get("payoff") or None,
                            required_context=_c.get("required_context") or None,
                            rationale=_c.get("rationale"),
                            uncertainty=_c.get("uncertainty"),
                            scores={"provenance": _c.get("provenance", {})},
                            status="proposed")
                    plog.info("audio_events done events=%d cues=%d classifier=%s",
                              len(events), len(cues), _cstat.get("status"))
                    if _afp_now:
                        run_repo.set_fingerprint(video_id, "audio_events", _afp_now)
                else:
                    plog.info("audio_events disabled by config")
            run_repo.complete_stage(video_id, "audio_events")
        except Exception as e:
            plog.exception("audio_events failed: %s", e)
            run_repo.fail_stage(video_id, "audio_events", str(e))
            raise

    # Step 6: Embed
    if progress: progress(0.5, desc="[6/11] Embedding chunks...")
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

    # Step 7: Outline (whole-video structure; fingerprint-gated)
    if progress: progress(0.55, desc="[7/11] Outlining video...")
    _tsig = transcript_signature(ws.transcript)
    _ofp = outline_fingerprint(_tsig, CONFIG.get("ai", {}).get("llm_model", ""), CONFIG)
    _outline_stale = (not run_repo.fingerprint_matches(video_id, "outline", _ofp))
    if _outline_stale and run_repo.get_stage_status(video_id, "outline") == "done":
        plog.info("outline fingerprint changed, invalidating outline+analyze+similarity+refine+render")
        for _st in ("outline", "analyze", "similarity", "refine", "render"):
            invalidate_stage(video_id, _st)
    if not should_skip_stage(video_id, "outline"):
        run_repo.start_stage(video_id, "outline")
        try:
            with log_stage("main", "outline", video_id=video_id):
                from pipeline.boundaries import load_words, build_sentences
                from pipeline.outline import build_outline, persist_outline
                _words = load_words(ws.transcript)
                _sents = build_sentences(_words)
                _ol = build_outline(_words, _sents, None, cfg=CONFIG)
                persist_outline(video_id, _ol)
                run_repo.set_fingerprint(video_id, "outline", _ofp)
                plog.info("outline done method=%s sections=%d",
                          _ol.get("method"), len(_ol.get("sections", [])))
            run_repo.complete_stage(video_id, "outline")
        except Exception as e:
            plog.exception("outline failed: %s", e)
            run_repo.fail_stage(video_id, "outline", str(e))
            raise

    # Step 8: Analyze
    if progress: progress(0.6, desc="[8/11] Analyzing for clips...")
    if not should_skip_stage(video_id, "analyze"):
        run_repo.start_stage(video_id, "analyze")
        try:
            with log_stage("main", "analyze", video_id=video_id):
                from db.repositories import chunk_repo
                chunks = chunk_repo.get_chunks_for_video(video_id)
                plog.info("analyzing %d chunks", len(chunks))
                clips = analyze(video_id, chunks, CONFIG,
                                campaign_text=campaign_description or "")
                plog.info("analyze found %d clips", len(clips))
            run_repo.complete_stage(video_id, "analyze")
        except Exception as e:
            plog.exception("analyze failed: %s", e)
            run_repo.fail_stage(video_id, "analyze", str(e))
            raise

    # Step 9: Similarity
    if progress: progress(0.7, desc="[9/11] Finding related segments...")
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

    # Step 10: Boundary refinement (sentence-complete ends, LLM-validated)
    refine_cfg = dict(CONFIG.get("refine", {}))
    if progress: progress(0.78, desc="[10/11] Refining boundaries...")
    if not should_skip_stage(video_id, "refine"):
        run_repo.start_stage(video_id, "refine")
        try:
            with log_stage("main", "refine", video_id=video_id):
                from pipeline.boundaries import refine_all_clips, persist_refinements
                clips = clip_repo.get_clips_for_video(video_id)
                confirmed = clip_repo.get_confirmed_segments_for_video(video_id)
                for c in clips:
                    c["related_segments"] = confirmed.get(c["id"], [])
                if refine_cfg.get("enabled", True):
                    vid_dur = get_video_duration(raw_video_path)
                    refined = refine_all_clips(
                        clips, transcript_path=ws.transcript,
                        video_duration=vid_dur,
                        cfg={**CONFIG, **refine_cfg})
                    nrej = sum(1 for c in refined if c.get("refine_status") == "rejected")
                    # post-refine dedup: boundaries can expand into each other
                    from pipeline.fusion import deduplicate_refined
                    _kept, _dropped = deduplicate_refined(refined)
                    for _dc, _why in _dropped:
                        _dc["refine_status"] = "rejected"
                        _dc["refine_reason"] = _why[:500]
                    if _dropped:
                        plog.info("post-refine dedup dropped=%d", len(_dropped))
                    plog.info("refine done clips=%d rejected=%d",
                              len(refined),
                              sum(1 for c in refined if c.get("refine_status") == "rejected"))
                    persist_refinements(video_id, refined)
                else:
                    plog.info("refine disabled by config, keeping analyzer boundaries")
            run_repo.complete_stage(video_id, "refine")
            # Boundaries changed -> rendered outputs are stale: force render rerun.
            invalidate_stage(video_id, "render")
        except Exception as e:
            plog.exception("refine failed: %s", e)
            run_repo.fail_stage(video_id, "refine", str(e))
            raise

    if not render:
        plog.info("=== PIPELINE STOPPED BEFORE RENDER (caller renders) elapsed=%.1fs ===",
                  time.monotonic() - t0)
        return []

    # Step 11: Render (extracted - see render_clips())
    final_clips = render_clips(video_id, raw_video_path, campaign_config, progress,
                               layout=layout)

    elapsed = time.monotonic() - t0
    plog.info("=== PIPELINE COMPLETE clips=%d elapsed=%.1fs report=%s ===",
              len(final_clips), elapsed, ws.report)
    log.info("=== PIPELINE COMPLETE video=%.8s clips=%d elapsed=%.1fs ===",
             video_id, len(final_clips), elapsed)

    return final_clips

def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("campaign", "job"):
        from campaign.cli import main as campaign_main   # campaign jobs (C5)
        sys.exit(campaign_main(sys.argv[1:]))
    parser = argparse.ArgumentParser(description="Viral Clips Automator")
    parser.add_argument("video_input", nargs='?', default=None, help="URL or local path to the video")
    parser.add_argument("--campaign", help="Natural language description of the campaign", default="")
    parser.add_argument("--template", help="Name of the campaign template to use", default=None)
    parser.add_argument("--layout", help="Vertical layout: auto, speaker_crop, stacked_split, center_crop (branded_fit only without full_screen_vertical)",
                        default=None, choices=["auto", "speaker_crop", "stacked_split", "center_crop", "branded_fit"])
    parser.add_argument("--no-refine", help="Skip boundary refinement (keep analyzer timestamps)",
                        action="store_true")
    parser.add_argument("--log-level", help="Override log level (DEBUG/INFO/WARNING/ERROR)", default=None)
    parser.add_argument("--fresh", action="store_true",
                        help="Start a new run even if this source was processed before "
                             "(default: resume it, skipping finished stages)")
    parser.add_argument("--from-stage", choices=STAGES, default=None,
                        help="Redo this stage and everything after it for the resumed video")
    parser.add_argument("--video-id", default=None,
                        help="Continue a specific existing video id")
    parser.add_argument("--debug-framing", action="store_true",
                        help="Also write side-by-side framing debug videos "
                             "(work/<video>/clips/clip_N_vertical_debug.mp4)")

    args = parser.parse_args()
    if args.log_level:
        setup_logging(level=args.log_level.upper(), force=True)
        log.info("log level overridden to %s", args.log_level.upper())
    if args.debug_framing:
        CONFIG.setdefault("framing", {})["debug_video"] = True
    if args.no_refine:
        try:
            CONFIG["refine"]["enabled"] = False
            log.info("boundary refinement disabled via --no-refine")
        except Exception:
            pass
    if not args.video_input:
        log.warning("no video_input provided")
        print("Usage: python main.py <video_path>")
    else:
        log.info("CLI start input=%.120s campaign=%.120s template=%s layout=%s",
                 args.video_input, args.campaign, args.template, args.layout)
        try:
            run_pipeline(args.video_input, args.campaign, args.template,
                         layout=args.layout, resume=not args.fresh,
                         from_stage=args.from_stage, video_id=args.video_id)
        except Exception:
            log.exception("CLI pipeline failed")
            raise

if __name__ == "__main__":
    main()
