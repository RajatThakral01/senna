# clipper.py
import subprocess
import os

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
    First tuple is always the primary clip.
    Subsequent tuples are confirmed continuation segments, sorted by their
    position in the video (not by similarity score).
    """
    start = time_to_seconds(clip["start_time"])
    end = time_to_seconds(clip["end_time"])
    ranges = [(start, end)]

    related = sorted(
        clip.get("related_segments", []),
        key=lambda r: time_to_seconds(r["start_time"])
    )
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

    for clip in clips:
        clip_num = clip["clip_number"]
        ranges = get_time_ranges_for_clip(clip)
        output_path = os.path.join(output_dir, f"clip_{clip_num}.mp4")

        print(f"✂️   Cutting clip {clip_num}: {len(ranges)} segment(s)")
        
        ffmpeg_path = os.path.expanduser("~/miniforge3/bin/ffmpeg")
        if not os.path.exists(ffmpeg_path):
            ffmpeg_path = "ffmpeg" # fallback to system ffmpeg

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
            print(f"❌ Error cutting clip {clip_num}: {result.stderr[-200:]}")
            continue
            
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
        print(f"✅  Clip {clip_num} saved: {output_path}")

    return clip_paths