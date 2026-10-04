import os
import shutil
import subprocess
import time
import gdown
import requests
from logger import get_logger

log = get_logger("input")

def detect_source_type(input_string: str) -> str:
    if "youtube.com" in input_string or "youtu.be" in input_string:
        st = "youtube_live" if "live" in input_string else "youtube"
    elif "drive.google.com" in input_string:
        st = "google_drive"
    elif input_string.endswith((".mp4", ".mov", ".webm", ".mkv")):
        st = "local_file" if input_string.startswith(("/", "./")) else "direct_url"
    else:
        st = "unknown"
    log.debug("detect_source input=%.100s -> %s", input_string, st)
    return st

def download_from_drive(url: str, output_path: str):
    log.info("drive download start url=%.100s out=%s", url, output_path)
    t0 = time.monotonic()
    gdown.download(url, output_path, quiet=False, fuzzy=True)
    size = os.path.getsize(output_path) if os.path.exists(output_path) else -1
    log.info("drive download done bytes=%d elapsed=%.1fs", size, time.monotonic() - t0)

def _run_yt_dlp(cmd: list, label: str):
    """Run yt-dlp capturing output; log stderr tail on failure."""
    log.debug("%s cmd=%.200s", label, " ".join(cmd[:-1]) + " <url>")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tail = (r.stderr or "")[-800:]
        log.warning("%s failed rc=%d err=%.800s", label, r.returncode, tail)
        raise subprocess.CalledProcessError(r.returncode, cmd, output=r.stdout, stderr=r.stderr)
    return r

def _handle_youtube(url: str):
    """Handles standard YouTube video downloads using yt-dlp.

    Tries full-quality (default client) first, falls back to the Android
    player client (usually 360p) when the CDN blocks default-client URLs
    with HTTP 403.
    """
    log.info("youtube download start url=%.100s", url)
    t0 = time.monotonic()
    OUTPUT_PATH = "input/raw_video.mp4"
    # Using the same pattern as the project's downloader.py
    ffmpeg_path = os.path.expanduser('~/miniforge3/bin/')
    cmd = [
        'yt-dlp',
        '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        '--merge-output-format', 'mp4',
        '--ffmpeg-location', ffmpeg_path,
        '--retries', '3',
        '-o', OUTPUT_PATH,
        url
    ]
    try:
        _run_yt_dlp(cmd, "youtube-hq")
    except subprocess.CalledProcessError as e1:
        log.warning("youtube-hq failed, retrying via android client (360p)")
        fallback = [
            'yt-dlp',
            '--extractor-args', 'youtube:player_client=android',
            '-f', 'best',
            '--merge-output-format', 'mp4',
            '--ffmpeg-location', ffmpeg_path,
            '--retries', '3',
            '-o', OUTPUT_PATH,
            url
        ]
        try:
            _run_yt_dlp(fallback, "youtube-android-fallback")
        except subprocess.CalledProcessError:
            log.exception("youtube download failed (both clients) url=%.100s", url)
            raise e1
    # Safety fallback: rename .webm to .mp4 if format negotiation failed
    webm_path = OUTPUT_PATH + ".webm"
    if not os.path.exists(OUTPUT_PATH) and os.path.exists(webm_path):
        shutil.move(webm_path, OUTPUT_PATH)
        log.info("renamed webm output to %s", OUTPUT_PATH)
    size = os.path.getsize(OUTPUT_PATH) if os.path.exists(OUTPUT_PATH) else -1
    log.info("youtube download done bytes=%d elapsed=%.1fs path=%s", size, time.monotonic() - t0, OUTPUT_PATH)

def _handle_youtube_live(url: str):
    """Handles YouTube live streams, recording for 300 seconds."""
    log.info("youtube-live record start (max 300s) url=%.100s", url)
    t0 = time.monotonic()
    OUTPUT_PATH = "input/raw_video.mp4"
    ffmpeg_path = os.path.expanduser('~/miniforge3/bin/')
    cmd = [
        'yt-dlp',
        '--live-from-start',
        '--merge-output-format', 'mp4',
        '--downloader', 'ffmpeg',
        '--downloader-args', 'ffmpeg_i:-t 300',
        '--ffmpeg-location', ffmpeg_path,
        '-o', OUTPUT_PATH,
        url
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        log.exception("youtube-live record failed")
        raise
    size = os.path.getsize(OUTPUT_PATH) if os.path.exists(OUTPUT_PATH) else -1
    log.info("youtube-live record done bytes=%d elapsed=%.1fs", size, time.monotonic() - t0)

def copy_local_file(input_path: str, output_path: str):
    log.info("copy local src=%s dst=%s", input_path, output_path)
    if not os.path.exists(input_path):
        log.error("local file missing src=%s", input_path)
        raise FileNotFoundError(input_path)
    shutil.copy(input_path, output_path)
    log.info("copy done bytes=%d", os.path.getsize(output_path))

def download_direct_url(url: str, output_path: str):
    log.info("direct download start url=%.100s out=%s", url, output_path)
    t0 = time.monotonic()
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = 0
        with open(output_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                total += len(chunk)
    log.info("direct download done bytes=%d elapsed=%.1fs", total, time.monotonic() - t0)

def handle_input(input_string: str) -> str:
    log.info("handle_input src=%.120s", input_string)
    source_type = detect_source_type(input_string)
    output_path = "input/raw_video.mp4"
    os.makedirs("input", exist_ok=True)

    if source_type == "youtube":
        _handle_youtube(input_string)
    elif source_type == "youtube_live":
        _handle_youtube_live(input_string)
    elif source_type == "google_drive":
        download_from_drive(input_string, output_path)
    elif source_type == "local_file":
        copy_local_file(input_string, output_path)
    elif source_type == "direct_url":
        download_direct_url(input_string, output_path)
    else:
        log.error("unknown source type input=%.120s", input_string)
        raise ValueError(f"Unknown or unsupported source type for input: {input_string}")

    log.info("handle_input done type=%s path=%s", source_type, output_path)
    return output_path
