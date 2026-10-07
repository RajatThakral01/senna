# clipper.py
import subprocess
import os
import time
from logger import get_logger
from config import ffmpeg_path as _ffmpeg_path

log = get_logger("pipeline.clipper")

def time_to_seconds(time_str):
    """Converts HH:MM:SS or float to total seconds."""
    if isinstance(time_str, (int, float)):
        return float(time_str)
    parts = time_str.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(time_str)

def get_time_ranges_for_clip(clip: dict) -> list[tuple[float, float]]:
    """
    Returns a list of (start, end) tuples for FFmpeg.
    Prefers refined source_ranges (boundary step) when present; otherwise
    primary + sorted confirmed continuations, overlapping ranges merged.
    """
    refined = clip.get("source_ranges")
    if refined:
        try:
            ranges = sorted((float(a), float(b)) for a, b in refined)
            merged = []
            for r in ranges:
                if not merged or r[0] > merged[-1][1] + 0.05:
                    merged.append(r)
                else:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], r[1]))
            # honour explicit output timeline when supplied
            tl = clip.get("timeline")
            if tl:
                ordered = sorted(tl, key=lambda m: m.get("output_start", 0))
                return [(float(m["source_start"]), float(m["source_end"])) for m in ordered]
            return merged
        except (TypeError, ValueError, KeyError):
            pass  # fall through to legacy path

    start = time_to_seconds(clip["start_time"])
    end = time_to_seconds(clip["end_time"])

    related = sorted(
        clip.get("related_segments", []),
        key=lambda r: time_to_seconds(r["start_time"])
    )
    # validate non-contiguous joins (distant != continuation)
    try:
        from config import get_config as _gc
        from pipeline.fusion import limit_stitch_ranges as _lim
        related, _dropped = _lim((start, end), related, _gc())
        for _seg, _why in _dropped:
            log.info("clip %s stitch dropped %.1f-%.1f: %s",
                     clip.get("clip_number"),
                     float(_seg.get("start_time", 0)),
                     float(_seg.get("end_time", 0)), _why)
    except Exception:
        log.debug("stitch limits skipped", exc_info=True)
    ranges = [(start, end)]
    for seg in related:
        ranges.append((time_to_seconds(seg["start_time"]), time_to_seconds(seg["end_time"])))

    # Prevent overlapping ranges if related segment overlaps with main clip or each other
    merged_ranges = []
    for r in sorted(ranges):
        if not merged_ranges:
            merged_ranges.append(r)
        else:
            last = merged_ranges[-1]
            if r[0] <= last[1]:
                merged_ranges[-1] = (last[0], max(last[1], r[1]))
            else:
                merged_ranges.append(r)

    return merged_ranges

def cut_clips(video_path, clips, output_dir="clips"):
    """
    Cuts video into clips using FFmpeg based on timestamps and related segments.
    Returns list of clip file paths.
    """
    clip_paths = []
    t0 = time.monotonic()
    log.info("cut_clips start video_src=%.80s clips=%d out_dir=%s", video_path, len(clips), output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for clip in clips:
        clip_num = clip["clip_number"]
        ranges = get_time_ranges_for_clip(clip)
        output_path = os.path.join(output_dir, f"clip_{clip_num}.mp4")

        log.info("cutting clip %d: %d segment(s) ranges=%s", clip_num, len(ranges), [(round(a, 1), round(b, 1)) for a, b in ranges])
        
        ffmpeg_path = _ffmpeg_path()
        log.debug("clip %d ffmpeg=%s", clip_num, ffmpeg_path)

        cmd = [ffmpeg_path, "-y"]
        
        # Build inputs
        for r_start, r_end in ranges:
            cmd.extend(["-ss", str(r_start), "-to", str(r_end), "-i", video_path])

        if len(ranges) == 1:
            # Single segment
            cmd.extend([
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "aac",
                "-avoid_negative_ts", "1",
                "-reset_timestamps", "1",
                output_path
            ])
        else:
            # Concat multiple segments
            filter_complex = ""
            for i in range(len(ranges)):
                filter_complex += f"[{i}:v][{i}:a]"
            filter_complex += f"concat=n={len(ranges)}:v=1:a=1[v][a]"
            
            cmd.extend([
                "-filter_complex", filter_complex,
                "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                output_path
            ])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.error("cut clip %d failed rc=%d err=%.300s", clip_num, result.returncode, result.stderr[-300:] if result.stderr else "")
            continue

        size = os.path.getsize(output_path) if os.path.exists(output_path) else -1
        log.info("clip %d saved path=%s bytes=%d", clip_num, output_path, size)
        clip_paths.append({
            "clip_number": clip_num,
            "path": output_path,
            "start_time": clip["start_time"],
            "end_time": clip["end_time"],
            "hook": clip.get("hook", ""),
            "suggested_title": clip.get("suggested_title", ""),
            "suggested_hashtags": clip.get("suggested_hashtags", ""),
            "reason": clip.get("reason", "")
        })
        log.debug("clip %d meta hook=%.40s", clip_num, clip.get("hook", ""))

    log.info("cut_clips done ok=%d/%d elapsed=%.1fs", len(clip_paths), len(clips), time.monotonic() - t0)
    return clip_paths