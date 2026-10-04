#!/usr/bin/env python3
import subprocess
import os

def remove_silences(input_path, output_path):
    cmd = [
        os.path.expanduser('~/miniforge3/bin/ffmpeg'),
        '-y', '-i', input_path,
        '-af', (
            'silenceremove='
            'stop_periods=-1:'
            'stop_duration=0.3:'
            'stop_threshold=-35dB'
        ),
        '-c:v', 'copy',
        '-c:a', 'aac',
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Silence removal error: {result.stderr[-300:]}")
    else:
        print(f"Silences removed: {output_path}")

def add_fades(input_path, output_path, fade_duration=0.3):
    probe = subprocess.run([
        os.path.expanduser('~/miniforge3/bin/ffprobe'),
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        input_path
    ], capture_output=True, text=True)

    duration = float(probe.stdout.strip())
    fade_out_start = duration - fade_duration

    cmd = [
        os.path.expanduser('~/miniforge3/bin/ffmpeg'),
        '-y', '-i', input_path,
        '-vf', (
            f'fade=t=in:st=0:d={fade_duration},'
            f'fade=t=out:st={fade_out_start:.3f}:d={fade_duration}'
        ),
        '-af', (
            f'afade=t=in:st=0:d={fade_duration},'
            f'afade=t=out:st={fade_out_start:.3f}:d={fade_duration}'
        ),
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-c:a', 'aac',
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Fade error: {result.stderr[-300:]}")
    else:
        print(f"Fades added: {output_path}")

def apply_logo(input_path: str, output_path: str, logo_path: str, position: str = "top-right") -> str:
    """
    Overlays a logo PNG onto the video.
    Position options: top-right, top-left, bottom-right, bottom-left
    """
    position_map = {
        "top-right":    "W-w-20:20",
        "top-left":     "20:20",
        "bottom-right": "W-w-20:H-h-20",
        "bottom-left":  "20:H-h-20"
    }
    overlay = position_map.get(position, "W-w-20:20")
    
    cmd = [
        os.path.expanduser('~/miniforge3/bin/ffmpeg'),
        "-i", input_path, "-i", logo_path,
        "-filter_complex", f"[1:v]scale=120:-1[logo];[0:v][logo]overlay={overlay}",
        "-codec:a", "copy",
        "-y", output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Logo application error: {result.stderr[-300:]}")
        return input_path # Return original path on failure
    else:
        print(f"Logo applied: {output_path}")
        return output_path

def mix_music(input_path: str, output_path: str, music_path: str, music_volume: float = 0.3) -> str:
    """
    Mixes background music under the original video audio.
    music_volume: 0.0 to 1.0, where 1.0 = original speech volume
    """
    cmd = [
        os.path.expanduser('~/miniforge3/bin/ffmpeg'),
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
        print(f"Music mixing error: {result.stderr[-300:]}")
        return input_path # Return original path on failure
    else:
        print(f"Music mixed: {output_path}")
        return output_path