#!/usr/bin/env python3
import subprocess
import os
import time
from logger import get_logger
from config import ffmpeg_path, ffprobe_path

log = get_logger("pipeline.improvements")

def add_fades(input_path, output_path, fade_duration=0.3, audio=True):
    """Apply video (+optional audio) fades.

    Speech fade-out is DISABLED by default (audio=False): the clip ends on a
    completed sentence with end_padding silence, so no afade touches speech.
    Pass audio=True only to fade music/ambience, and only after the final
    spoken word (callers ensure the fade window sits inside end_padding).
    """
    log.info("fades start in=%.60s out=%.60s dur=%.2f audio=%s", input_path,
             output_path, fade_duration, audio)
    t0 = time.monotonic()
    probe = subprocess.run([
        ffprobe_path(),
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        input_path
    ], capture_output=True, text=True)

    if probe.returncode != 0 or not probe.stdout.strip():
        log.error("fades probe failed rc=%d err=%.300s, keeping original",
                  probe.returncode, (probe.stderr or "")[-300:])
        return input_path
    try:
        duration = float(probe.stdout.strip())
    except ValueError:
        log.error("fades bad duration output=%.100s, keeping original", probe.stdout.strip())
        return input_path
    fade_out_start = duration - fade_duration
    log.debug("fades duration=%.2f out_start=%.2f", duration, fade_out_start)

    cmd = [
        ffmpeg_path(),
        '-y', '-i', input_path,
        '-vf', (
            f'fade=t=in:st=0:d={fade_duration},'
            f'fade=t=out:st={fade_out_start:.3f}:d={fade_duration}'
        ),
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
    ]
    if audio:
        cmd += ['-af',
                f'afade=t=in:st=0:d={fade_duration},'
                f'afade=t=out:st={fade_out_start:.3f}:d={fade_duration}',
                '-c:a', 'aac']
    else:
        cmd += ['-c:a', 'copy']
    cmd += [output_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("fades failed rc=%d err=%.300s", result.returncode, (result.stderr or "")[-300:])
        return input_path  # keep original on failure
    else:
        log.info("fades done out=%s elapsed=%.1fs", output_path, time.monotonic() - t0)
        return output_path

def apply_logo(input_path: str, output_path: str, logo_path: str, position: str = "top-right") -> str:
    """
    Overlays a logo PNG onto the video.
    Position options: top-right, top-left, bottom-right, bottom-left
    """
    log.info("logo start in=%.60s logo=%.60s pos=%s", input_path, logo_path, position)
    t0 = time.monotonic()
    position_map = {
        "top-right":    "W-w-20:20",
        "top-left":     "20:20",
        "bottom-right": "W-w-20:H-h-20",
        "bottom-left":  "20:H-h-20"
    }
    overlay = position_map.get(position, "W-w-20:20")

    cmd = [
        ffmpeg_path(),
        "-i", input_path, "-i", logo_path,
        "-filter_complex", f"[1:v]scale=120:-1[logo];[0:v][logo]overlay={overlay}",
        "-codec:a", "copy",
        "-y", output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("logo failed rc=%d err=%.300s, keeping original", result.returncode, (result.stderr or "")[-300:])
        return input_path # Return original path on failure
    else:
        log.info("logo done out=%s elapsed=%.1fs", output_path, time.monotonic() - t0)
        return output_path

def mix_music(input_path: str, output_path: str, music_path: str, music_volume: float = 0.3) -> str:
    """
    Mixes background music under the original video audio.
    music_volume: 0.0 to 1.0, where 1.0 = original speech volume
    """
    log.info("music start in=%.60s music=%.60s vol=%.2f", input_path, music_path, music_volume)
    t0 = time.monotonic()
    cmd = [
        ffmpeg_path(),
        "-i", input_path,
        "-i", music_path,
        "-filter_complex",
        f"[1:a]volume={music_volume}[music];[0:a][music]amix=inputs=2:duration=first[aout]",
        "-map", "0:v",
        "-map", "[aout]",
        "-codec:v", "copy",
        "-shortest",
        "-y", output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("music failed rc=%d err=%.300s, keeping original", result.returncode, (result.stderr or "")[-300:])
        return input_path # Return original path on failure
    else:
        log.info("music done out=%s elapsed=%.1fs", output_path, time.monotonic() - t0)
        return output_path