import os
import shutil
import subprocess
import gdown
import requests

def detect_source_type(input_string: str) -> str:
    if "youtube.com" in input_string or "youtu.be" in input_string:
        if "live" in input_string:
            return "youtube_live"
        return "youtube"
    if "drive.google.com" in input_string:
        return "google_drive"
    if input_string.endswith((".mp4", ".mov", ".webm", ".mkv")):
        if input_string.startswith("/") or input_string.startswith("./"):
            return "local_file"
        return "direct_url"
    return "unknown"

def download_from_drive(url: str, output_path: str):
    gdown.download(url, output_path, quiet=False, fuzzy=True)

def _handle_youtube(url: str):
    """Handles standard YouTube video downloads using yt-dlp."""
    print(f"Downloading YouTube video: {url}")
    OUTPUT_PATH = "input/raw_video.mp4"
    # Using the same pattern as the project's downloader.py
    ffmpeg_path = os.path.expanduser('~/miniforge3/bin/')
    cmd = [
        'yt-dlp',
        '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        '--merge-output-format', 'mp4',
        '--ffmpeg-location', ffmpeg_path,
        '-o', OUTPUT_PATH,
        url
    ]
    subprocess.run(cmd, check=True)
    # Safety fallback: rename .webm to .mp4 if format negotiation failed
    webm_path = OUTPUT_PATH + ".webm"
    if not os.path.exists(OUTPUT_PATH) and os.path.exists(webm_path):
        import shutil
        shutil.move(webm_path, OUTPUT_PATH)
        print(f"Renamed webm output to {OUTPUT_PATH}")

def _handle_youtube_live(url: str):
    """Handles YouTube live streams, recording for 300 seconds."""
    print(f"Recording YouTube live stream (max 300s): {url}")
    ffmpeg_path = os.path.expanduser('~/miniforge3/bin/')
    cmd = [
        'yt-dlp',
        '--live-from-start',
        '--merge-output-format', 'mp4',
        '--downloader', 'ffmpeg',
        '--downloader-args', f'ffmpeg_i:-t 300',
        '--ffmpeg-location', ffmpeg_path,
        '-o', OUTPUT_PATH,
        url
    ]
    subprocess.run(cmd, check=True)

def download_youtube_video(url: str, output_path: str):
    cmd = [
        "yt-dlp",
        "-o", output_path,
        url
    ]
    subprocess.run(cmd, check=True)

def copy_local_file(input_path: str, output_path: str):
    shutil.copy(input_path, output_path)

def download_direct_url(url: str, output_path: str):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def handle_input(input_string: str) -> str:
    source_type = detect_source_type(input_string)
    output_path = "input/raw_video.mp4"

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
        raise ValueError(f"Unknown or unsupported source type for input: {input_string}")

    return output_path
