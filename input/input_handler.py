import os
import shutil
import subprocess
import sys
import time
import gdown
import requests
from logger import get_logger
from config import ffmpeg_path as _ffmpeg_bin

log = get_logger("input")

def _ytdlp():
    """yt-dlp invocation bound to this interpreter (no PATH dependency)."""
    return [sys.executable, '-m', 'yt_dlp']

def _ffmpeg_location_args() -> list:
    """['--ffmpeg-location', dir] when ffmpeg resolves to a real file, else []."""
    exe = _ffmpeg_bin()
    loc = os.path.dirname(exe)
    if loc and os.path.exists(os.path.join(loc, os.path.basename(exe))):
        return ['--ffmpeg-location', loc]
    return []  # ffmpeg on PATH — yt-dlp finds it itself

# Minimum accepted source height. Below this a 9:16 crop holds too few real
# pixels (360p -> 202x360 upscaled 5.3x = blur), so the download fails loudly
# instead of producing blurry clips. Override via YT_MIN_HEIGHT env var.
MIN_SOURCE_HEIGHT = int(os.getenv("YT_MIN_HEIGHT", "720"))

# HD-first format ladder: 720p+ mp4 preferred, any 720p+ next, plain best
# only as a last resort (still gated by the resolution probe below).
HQ_FORMAT = ('bestvideo[height>=720][ext=mp4]+bestaudio[ext=m4a]/'
             'bestvideo[height>=720]+bestaudio/'
             'best[height>=720]/best')


def _js_runtime_args() -> list:
    """['--js-runtimes', name] for the first available runtime.

    yt-dlp needs a JS runtime to unlock HD YouTube formats; without one the
    HQ attempt fails and everything degrades to 360p.
    """
    for name in ("deno", "node"):
        if shutil.which(name):
            return ["--js-runtimes", name]
    log.warning("no JS runtime (deno/node) on PATH — yt-dlp HD formats "
                "may be unavailable, SD fallback likely")
    return []


def _probe_height(path: str) -> int:
    """Video stream height via ffprobe, or 0 when unreadable."""
    try:
        from config import ffprobe_path
        r = subprocess.run(
            [ffprobe_path(), "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=height", "-of",
             "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=60)
        return int((r.stdout or "").strip() or 0)
    except Exception:
        return 0

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

    Client ladder (first probe pass wins):
      1. youtube-hq — default client + JS runtime, HD-first format ladder.
      2. android fallback — recovers when the CDN 403s default-client URLs
         (often caps at 360p).
    After each attempt the file is probed: anything below MIN_SOURCE_HEIGHT
    is rejected and the next client is tried. If no client reaches HD the
    download FAILS LOUDLY — a 360p source would upscale ~5x into blur,
    so silent SD is worse than an explicit error.
    """
    log.info("youtube download start url=%.100s (min %dp)", url, MIN_SOURCE_HEIGHT)
    t0 = time.monotonic()
    OUTPUT_PATH = "input/raw_video.mp4"
    js = _js_runtime_args()
    attempts = [
        ("youtube-hq", [
            *_ytdlp(),
            '-f', HQ_FORMAT,
            '--merge-output-format', 'mp4',
            *js,
            *_ffmpeg_location_args(),
            '--retries', '3',
            '-o', OUTPUT_PATH,
            url
        ]),
        ("youtube-android-fallback", [
            *_ytdlp(),
            '--extractor-args', 'youtube:player_client=android',
            '-f', 'best',
            '--merge-output-format', 'mp4',
            *js,
            *_ffmpeg_location_args(),
            '--retries', '3',
            '-o', OUTPUT_PATH,
            url
        ]),
    ]
    last_error = None
    for label, cmd in attempts:
        # drop a stale SD file so the probe below always checks this attempt
        try:
            if os.path.exists(OUTPUT_PATH):
                os.remove(OUTPUT_PATH)
        except OSError:
            pass
        try:
            _run_yt_dlp(cmd, label)
        except subprocess.CalledProcessError as e:
            log.warning("%s failed, trying next client", label)
            last_error = e
            continue
        height = _probe_height(OUTPUT_PATH)
        log.info("%s done height=%dp", label, height)
        if height >= MIN_SOURCE_HEIGHT:
            break
        log.warning("%s only reached %dp (< %dp), trying next client",
                    label, height, MIN_SOURCE_HEIGHT)
        last_error = RuntimeError(f"{label} produced {height}p")
    else:
        # loop exhausted without break — no HD source obtained
        log.error("youtube download failed to reach %dp url=%.100s",
                  MIN_SOURCE_HEIGHT, url)
        raise RuntimeError(
            f"YouTube download never reached {MIN_SOURCE_HEIGHT}p — "
            f"refusing SD source (a 9:16 crop of SD upscales ~5x into blur). "
            f"Install deno/node for HD formats or pick another video.") \
            from last_error
    size = os.path.getsize(OUTPUT_PATH) if os.path.exists(OUTPUT_PATH) else -1
    log.info("youtube download done bytes=%d elapsed=%.1fs path=%s", size, time.monotonic() - t0, OUTPUT_PATH)

def _handle_youtube_live(url: str):
    """Handles YouTube live streams, recording for 300 seconds."""
    log.info("youtube-live record start (max 300s) url=%.100s", url)
    t0 = time.monotonic()
    OUTPUT_PATH = "input/raw_video.mp4"
    cmd = [
        *_ytdlp(),
        '--live-from-start',
        '--merge-output-format', 'mp4',
        '--downloader', 'ffmpeg',
        '--downloader-args', 'ffmpeg_i:-t 300',
        *_js_runtime_args(),
        *_ffmpeg_location_args(),
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
