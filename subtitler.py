# subtitler.py
import requests
import subprocess
import os
import json
from config import LLM_API_KEY, LLM_MODEL, LLM_API_URL

def time_to_seconds(time_str):
    parts = time_str.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

def seconds_to_srt_time(seconds):
    """Converts seconds to SRT format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def get_clip_transcript_segments(transcript_result, start_time_str, end_time_str):
    """
    Extracts transcript segments that fall within the clip's time range.
    """
    start_sec = time_to_seconds(start_time_str)
    end_sec = time_to_seconds(end_time_str)

    segments = []
    for seg in transcript_result["segments"]:
        # Include segment if it overlaps with the clip window
        if seg["end"] >= start_sec and seg["start"] <= end_sec:
            # Adjust timestamps relative to clip start
            adjusted_start = max(0, seg["start"] - start_sec)
            adjusted_end = min(end_sec - start_sec, seg["end"] - start_sec)
            segments.append({
                "start": adjusted_start,
                "end": adjusted_end,
                "text": seg["text"].strip()
            })
    return segments

def generate_srt_with_m27(segments, clip_number):
    """
    Sends clip segments to M2.7 to format as clean subtitles.
    Returns formatted SRT content as string.
    """

    # Build input for M2.7
    raw_segments = "\n".join([
        f"[{s['start']:.2f}s --> {s['end']:.2f}s] {s['text']}"
        for s in segments
    ])

    prompt = f"""You are a subtitle formatter for short viral videos (TikTok, Reels, Shorts).

Here are raw transcript segments with timestamps (in seconds) for a short clip:

{raw_segments}

Format these as clean SRT subtitles following these rules:
1. Maximum 5-6 words per subtitle line
2. Break at natural speech pauses
3. Each subtitle should appear for 1.5-3 seconds
4. Fix any transcription errors or grammar issues
5. Keep it punchy and easy to read quickly
6. Return ONLY valid SRT format, nothing else

SRT format:
1
00:00:00,000 --> 00:00:02,500
Text here

2
00:00:02,500 --> 00:00:05,000
More text here"""

    response = requests.post(
        LLM_API_URL,
        headers={
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1500,
            "temperature": 0.2
        }
    )

    if response.status_code != 200:
        raise Exception(f"API error: {response.status_code}")

    srt_content = response.json()["choices"][0]["message"]["content"].strip()

    # Clean up if M2.7 wraps in code fences
    if "```" in srt_content:
        srt_content = srt_content.split("```")[1]
        if srt_content.startswith("srt"):
            srt_content = srt_content[3:]

    return srt_content.strip()

def save_srt(srt_content, clip_number, output_dir="clips"):
    """Saves SRT content to file."""
    srt_path = os.path.join(output_dir, f"clip_{clip_number}.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_content)
    return srt_path

def burn_subtitles(clip_path, srt_path, clip_number, output_dir="output"):
    """
    Burns subtitles onto video using FFmpeg.
    Returns path to final output video.
    """
    output_path = os.path.join(output_dir, f"clip_{clip_number}_final.mp4")

    # FFmpeg subtitle filter with styling - escape special chars in path
    escaped_srt_path = srt_path.replace(":", "\\:").replace("\\", "\\\\")
    subtitle_filter = (
        f"subtitles={escaped_srt_path}:force_style='"
        f"FontName=Arial,"
        f"FontSize=18,"
        f"PrimaryColour=&H00FFFFFF,"
        f"OutlineColour=&H00000000,"
        f"Outline=2,"
        f"Bold=1,"
        f"Alignment=2'"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i", clip_path,
        "-vf", subtitle_filter,
        "-c:a", "copy",
        output_path
    ]

    print(f"🔤  Burning subtitles onto clip {clip_number}...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌  Error burning subtitles: {result.stderr}")
        return None

    print(f"✅  Final clip saved: {output_path}")
    return output_path

def process_clip_subtitles(clip_info, transcript_result, clips_dir="clips", output_dir="output"):
    """
    Full subtitle pipeline for one clip:
    1. Extract relevant transcript segments
    2. Send to M2.7 to format subtitles
    3. Save as .srt
    4. Burn onto video with FFmpeg
    """
    clip_num = clip_info["clip_number"]

    print(f"\n📝  Processing subtitles for clip {clip_num}...")

    # Step 1: Get transcript segments for this clip's time range
    segments = get_clip_transcript_segments(
        transcript_result,
        clip_info["start_time"],
        clip_info["end_time"]
    )

    if not segments:
        print(f"⚠️   No transcript segments found for clip {clip_num}")
        return None

    # Step 2: Format with M2.7
    srt_content = generate_srt_with_m27(segments, clip_num)

    # Step 3: Save .srt file
    srt_path = save_srt(srt_content, clip_num, clips_dir)
    print(f"✅  SRT saved: {srt_path}")

    # Step 4: Burn onto video
    final_path = burn_subtitles(clip_info["path"], srt_path, clip_num, output_dir)

    return final_path