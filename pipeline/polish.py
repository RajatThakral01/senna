"""
pipeline/polish.py

Consolidated final polish in ONE ffmpeg pass (Phase 5).

Merges what used to be 3-5 sequential encodes (subtitle burn passes, logo
overlay, music mix, fades) into a single command over the framed vertical:

  video: ass captions + video fade in/out + logo overlay + setsar=1 (+fps)
  audio: loudness normalization + optional denoise + music ducking + mix
         + peak limiter (+ optional audio fade, default off — never on speech)

Intermediate encodes are avoided for quality (no generation loss) and speed.
The legacy multi-pass chain (improvements.py) is kept for fallback and for
`export.consolidated: false`. Framing itself stays a separate pass: it is a
per-frame cv2 crop loop that cannot be expressed as a static filter graph
(documented; see editplan.limits).
"""
import time

from logger import get_logger

log = get_logger("pipeline.polish")


def probe_streams(path):
    """Return {has_video, has_audio, width, height, fps, duration} (best effort)."""
    import subprocess
    from config import ffprobe_path
    info = {"has_video": False, "has_audio": False, "width": 0, "height": 0,
            "fps": 0.0, "duration": 0.0}
    try:
        r = subprocess.run(
            [ffprobe_path(), "-v", "error", "-show_entries",
             "stream=codec_type,width,height,avg_frame_rate",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1", path],
            capture_output=True, text=True)
        if r.returncode != 0:
            return info
        cur = {}
        for line in (r.stdout or "").splitlines():
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k == "codec_type" and cur:
                _commit_stream(info, cur)
                cur = {}
            cur[k.strip()] = v.strip()
        if cur:
            _commit_stream(info, cur)
    except Exception:
        pass
    return info


def _commit_stream(info, cur):
    if cur.get("codec_type") == "video":
        info["has_video"] = True
        try:
            info["width"] = int(cur.get("width", 0))
            info["height"] = int(cur.get("height", 0))
        except ValueError:
            pass
        try:
            n, d = cur.get("avg_frame_rate", "0/1").split("/")
            info["fps"] = float(n) / float(d) if float(d) else 0.0
        except (ValueError, ZeroDivisionError):
            pass
    elif cur.get("codec_type") == "audio":
        info["has_audio"] = True
    try:
        if cur.get("duration"):
            info["duration"] = float(cur["duration"])
    except ValueError:
        pass


def build_video_chain(ass_file=None, fade_duration=0.5, out_duration=0.0,
                      logo_scale=120, logo_pos="W-w-20:20", use_logo=False,
                      fps=None):
    """Video filter chain ending at [v]. Logo overlay appended when use_logo."""
    parts = []
    if fps:
        parts.append(f"fps={fps}")
    if ass_file:
        parts.append(f"ass={ass_file}")
    if fade_duration and fade_duration > 0 and out_duration > 0:
        parts.append(f"fade=t=in:st=0:d={fade_duration}")
        parts.append(f"fade=t=out:st={max(0.0, out_duration - fade_duration):.3f}"
                     f":d={fade_duration}")
    parts.append("setsar=1")
    base = "[0:v]" + ",".join(parts)
    if use_logo:
        return (f"{base}[vbase];[2:v]scale={logo_scale}:-1[logo];"
                f"[vbase][logo]overlay={logo_pos}[v]", True)
    return base + "[v]", False


def build_audio_chain(has_speech, has_music, music_volume=0.1,
                      normalize=True, denoise=False, duck=True,
                      peak_db=-1.5, audio_fade=False, fade_duration=0.5,
                      out_duration=0.0):
    """Audio filter chain ending at [aout], or '' when no audio at all."""
    if not has_speech and not has_music:
        return ""
    if has_speech and not has_music:
        chain = "[0:a]"
        if denoise:
            chain += "afftdn,"
        if normalize:
            chain += f"loudnorm=I=-16:TP={peak_db}:LRA=11,"
        chain += "alimiter=limit=0.95[aout]"
        filt = chain
    elif has_music and not has_speech:
        # music-only source: normalize music itself to target
        filt = (f"[1:a]volume={music_volume},"
                f"loudnorm=I=-16:TP={peak_db}:LRA=11,"
                f"alimiter=limit=0.95[aout]")
    else:
        speech = "[0:a]"
        if denoise:
            speech += "afftdn,"
        if normalize:
            speech += f"loudnorm=I=-16:TP={peak_db}:LRA=11"
        speech += "[snorm]"
        if duck:
            music = (f"[1:a]volume={music_volume}[m];"
                     f"[m][snorm]sidechaincompress=threshold=0.02:ratio=8:"
                     f"attack=200:release=1000[dm]")
            mix = "[dm][snorm]amix=inputs=2:duration=first:dropout_transition=0[a]"
        else:
            music = f"[1:a]volume={music_volume}[m]"
            mix = "[m][snorm]amix=inputs=2:duration=first:dropout_transition=0[a]"
        filt = f"{speech};{music};{mix};[a]alimiter=limit=0.95[aout]"
    if audio_fade and fade_duration and out_duration:
        # fade sits in end_padding silence, applied after limiting
        filt += (f";[aout]afade=t=in:st=0:d={fade_duration},"
                 f"afade=t=out:st={max(0.0, out_duration - fade_duration):.3f}"
                 f":d={fade_duration}[af];[af]anull[aout]")
    return filt


def build_polish_cmd(vertical_path, output_path, ass_rel=None, logo_path=None,
                      logo_position="top-right", music_path=None,
                      music_volume=0.1, fade_duration=0.5, out_duration=0.0,
                      audio_cfg=None, exp_cfg=None, has_audio=True):
    """Pure command builder (testable without ffmpeg). Returns (cmd, desc)."""
    from config import ffmpeg_path
    audio_cfg = audio_cfg or {}
    exp_cfg = exp_cfg or {}
    codec = exp_cfg.get("codec", "libx264")
    preset = exp_cfg.get("preset", "fast")
    crf = exp_cfg.get("crf", 23)
    abitrate = exp_cfg.get("audio_bitrate", "128k")
    fps = exp_cfg.get("fps")  # None = keep source fps

    pos_map = {"top-right": "W-w-20:20", "top-left": "20:20",
               "bottom-right": "W-w-20:H-h-20", "bottom-left": "20:H-h-20"}
    use_logo = bool(logo_path)
    vchain, _ = build_video_chain(
        ass_file=ass_rel, fade_duration=fade_duration if fade_duration else 0,
        out_duration=out_duration,
        logo_pos=pos_map.get(logo_position, "W-w-20:20"),
        use_logo=use_logo, fps=fps)

    has_music = bool(music_path)
    achain = build_audio_chain(
        has_speech=has_audio, has_music=has_music, music_volume=music_volume,
        normalize=audio_cfg.get("normalize_loudness", True),
        denoise=audio_cfg.get("denoise", False),
        duck=audio_cfg.get("duck_music", True),
        peak_db=audio_cfg.get("peak_limit_db", -1.5),
        audio_fade=audio_cfg.get("audio_fade", False),
        fade_duration=fade_duration, out_duration=out_duration)

    cmd = [ffmpeg_path(), "-y", "-i", vertical_path]
    if has_music:
        cmd += ["-stream_loop", "-1", "-i", music_path]
    if use_logo:
        cmd += ["-i", logo_path]
    cmd += ["-filter_complex", f"{vchain}" + (f";{achain}" if achain else ""),
            "-map", "[v]"]
    if achain:
        cmd += ["-map", "[aout]", "-c:a", "aac", "-b:a", str(abitrate)]
    else:
        cmd += ["-an"]
    cmd += ["-c:v", codec, "-preset", str(preset), "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-shortest", output_path]
    desc = (f"ass={bool(ass_rel)} logo={use_logo} music={has_music} "
            f"norm={audio_cfg.get('normalize_loudness', True)} "
            f"duck={audio_cfg.get('duck_music', True)} "
            f"afade={audio_cfg.get('audio_fade', False)}")
    return cmd, desc


def polish_clip(vertical_path, output_path, plan, campaign_paths=None):
    """Run the consolidated polish for one edit plan. Returns output or ''.

    plan: edit plan dict (build_edit_plan). campaign_paths: resolved
    {logo, music} absolute/relative paths (or {}). Never raises.
    """
    import os
    import subprocess
    import tempfile
    t0 = time.monotonic()
    log.info("polish start clip=%s", plan.get("clip_number"))
    campaign_paths = campaign_paths or {}
    try:
        info = probe_streams(vertical_path)
        out_dur = plan.get("output_duration") or info.get("duration") or 0
        # stage ASS text to a CWD-relative file (Windows colon-safe)
        ass_rel = None
        cues = (plan.get("captions") or {}).get("cues", [])
        if cues:
            from pipeline.captions import build_ass
            cap_block = plan.get("captions") or {}
            ass_text = build_ass(cues, cap_block.get("style", {}),
                                 cap_block.get("placement", "bottom"))
            _dir = os.path.dirname(os.path.abspath(output_path)) or "."
            fd, ap = tempfile.mkstemp(suffix=".ass", dir=_dir)
            with os.fdopen(fd, "w", encoding="utf-8-sig") as f:
                f.write(ass_text)
            try:
                ass_rel = os.path.relpath(ap, ".").replace("\\", "/")
            except ValueError:
                ass_rel = ap.replace("\\", "/")
        else:
            ap = None
        br = plan.get("branding", {})
        au = plan.get("audio", {})
        ex = plan.get("export", {})
        cmd, desc = build_polish_cmd(
            vertical_path, output_path, ass_rel=ass_rel,
            logo_path=campaign_paths.get("logo") if br.get("logo") else None,
            logo_position=br.get("logo_position", "top-right"),
            music_path=campaign_paths.get("music") if br.get("music") else None,
            music_volume=br.get("music_volume", 0.1),
            fade_duration=au.get("fade_duration", 0.5) if br.get("fade") else 0,
            out_duration=out_dur, audio_cfg=au, exp_cfg=ex,
            has_audio=info.get("has_audio", True))
        log.info("polish cmd %s", desc)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if ap:
            try:
                os.remove(ap)
            except OSError:
                pass
        if r.returncode != 0:
            log.error("polish failed rc=%d err=%.500s", r.returncode,
                      (r.stderr or "")[-500:])
            return ""
        log.info("polish done out=%s elapsed=%.1fs", output_path,
                 time.monotonic() - t0)
        return output_path
    except Exception:
        log.exception("polish exception")
        return ""
