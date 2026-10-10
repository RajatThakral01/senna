"""pipeline/media.py — probe any media file (video, audio, image, font).

Used by the campaign asset library (what did the user upload?) and by the job
router (is this input a long video to clip, or a short one to edit?).
"""
import json
import os
import subprocess

from logger import get_logger

log = get_logger("pipeline.media")

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
FONT_EXT = {".ttf", ".otf"}


def kind_from_ext(path: str) -> str:
    ext = os.path.splitext(str(path))[1].lower()
    if ext in VIDEO_EXT:
        return "video"
    if ext in AUDIO_EXT:
        return "audio"
    if ext in IMAGE_EXT:
        return "image"
    if ext in FONT_EXT:
        return "font"
    return "unknown"


def _fps(rate: str) -> float:
    try:
        num, den = str(rate).split("/")
        return round(float(num) / float(den), 3) if float(den) else 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe(path: str) -> dict:
    """Describe a media file. Never raises.

    Returns {kind, exists, size_bytes, duration, has_video, has_audio, width,
    height, fps, aspect, video_codec, audio_codec, alpha} — kind is
    video | audio | image | font | unknown (from the streams when ffprobe can
    read the file, else from the extension).
    """
    info = {"kind": kind_from_ext(path), "exists": os.path.isfile(path),
            "size_bytes": os.path.getsize(path) if os.path.isfile(path) else 0,
            "duration": 0.0, "has_video": False, "has_audio": False, "width": 0,
            "height": 0, "fps": 0.0, "aspect": None, "video_codec": None,
            "audio_codec": None, "alpha": False}
    if not info["exists"] or info["kind"] == "font":
        return info
    try:
        from config import ffprobe_path
        r = subprocess.run([ffprobe_path(), "-v", "error", "-print_format", "json",
                            "-show_streams", "-show_format", path],
                           capture_output=True, text=True, timeout=60)
        data = json.loads(r.stdout or "{}") if r.returncode == 0 else {}
    except Exception:
        log.debug("probe failed path=%s", path, exc_info=True)
        return info
    for st in data.get("streams", []):
        if st.get("codec_type") == "video" and not info["has_video"]:
            info.update(has_video=True, width=int(st.get("width") or 0),
                        height=int(st.get("height") or 0),
                        fps=_fps(st.get("avg_frame_rate") or st.get("r_frame_rate") or "0/1"),
                        video_codec=st.get("codec_name"),
                        alpha="a" in str(st.get("pix_fmt", "")).replace("yuv", ""))
        elif st.get("codec_type") == "audio" and not info["has_audio"]:
            info.update(has_audio=True, audio_codec=st.get("codec_name"))
    try:
        info["duration"] = round(float((data.get("format") or {}).get("duration") or 0.0), 3)
    except (TypeError, ValueError):
        pass
    if info["width"] and info["height"]:
        info["aspect"] = round(info["width"] / info["height"], 4)
    # a still image probes as a 1-frame "video"; an mp3 with cover art has a picture stream
    if info["kind"] == "audio":
        info["has_video"] = False
    elif info["kind"] == "unknown":
        info["kind"] = ("video" if info["has_video"] and info["duration"] > 0.5
                        else "audio" if info["has_audio"] else
                        "image" if info["has_video"] else "unknown")
    return info


def aspect_label(width: int, height: int) -> str:
    """Nearest common aspect name: 9:16, 4:5, 1:1, 16:9 (or WxH)."""
    if not width or not height:
        return "unknown"
    r = width / height
    named = {"9:16": 9 / 16, "4:5": 4 / 5, "1:1": 1.0, "16:9": 16 / 9}
    name, val = min(named.items(), key=lambda kv: abs(kv[1] - r))
    return name if abs(val - r) / val < 0.04 else f"{width}x{height}"
