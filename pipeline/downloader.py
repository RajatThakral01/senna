# downloader.py
import yt_dlp
import os

def download_video(youtube_url, output_dir="downloads"):
    """
    Downloads a YouTube video and extracts audio separately.
    Returns paths to the video file and audio file.
    """

    # Settings for video download
    video_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": os.path.join(output_dir, "video.%(ext)s"),
        "merge_output_format": "mp4",
    }

    # Settings for audio extraction
    audio_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, "audio.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
    }

    print("⬇️  Downloading video...")
    with yt_dlp.YoutubeDL(video_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        video_title = info.get("title", "Unknown")
        duration = info.get("duration", 0)

    print("🎵  Extracting audio...")
    with yt_dlp.YoutubeDL(audio_opts) as ydl:
        ydl.extract_info(youtube_url, download=True)

    video_path = os.path.join(output_dir, "video.mp4")
    audio_path = os.path.join(output_dir, "audio.mp3")

    print(f"✅  Downloaded: '{video_title}' ({duration//60} mins)")
    return video_path, audio_path, video_title