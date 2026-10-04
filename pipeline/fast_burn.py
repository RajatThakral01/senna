#!/usr/bin/env python3
import subprocess
import os

def parse_srt(srt_path):
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    subtitles = []
    blocks = content.strip().split('\n\n')
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            time_line = lines[1]
            text = ' '.join(lines[2:])
            start_str, end_str = time_line.split('-->')
            start = parse_srt_time(start_str.strip())
            end = parse_srt_time(end_str.strip())
            subtitles.append((start, end, text))

    return subtitles

def parse_srt_time(time_str):
    time_str = time_str.replace(',', '.')
    parts = time_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds

def escape_drawtext(text):
    text = text.replace('\\', '\\\\')
    text = text.replace("'", "\u2019")
    text = text.replace(':', '\\:')
    text = text.replace('%', '\\%')
    text = text.replace(',', '\\,')
    return text

def build_filter_complex(subtitles):
    fontfile = '/System/Library/Fonts/Helvetica.ttc'
    chain = []
    prev_tag = "[0:v]"
    for i, (start, end, text) in enumerate(subtitles):
        escaped = escape_drawtext(text.upper())
        
        # Active word (yellow)
        out_tag = f"[v{i+1}]"
        f = (
            f"{prev_tag}drawtext="
            f"text='{escaped}'"
            f":fontfile={fontfile}"
            f":fontsize=95"
            f":fontcolor=0xFFE000"
            f":borderw=5"
            f":bordercolor=black"
            f":x=(w-text_w)/2"
            f":y=h*0.82"
            f":enable='between(t,{start:.3f},{end:.3f})'"
            f"{out_tag}"
        )
        chain.append(f)
        prev_tag = out_tag

    return ";".join(chain), prev_tag if chain else "[0:v]"

def burn_subtitles(clip_num, clip_path, output_path):
    srt_path = f"clips/clip_{clip_num}.srt"

    print(f"\nProcessing clip {clip_num}...")

    subtitles = parse_srt(srt_path)
    print(f"  Found {len(subtitles)} subtitle segments")

    filter_chain, last_tag = build_filter_complex(subtitles)

    cmd = [
        os.path.expanduser('~/miniforge3/bin/ffmpeg'),
        '-y',
        '-i', clip_path,
        '-filter_complex', filter_chain,
        '-map', last_tag,
        '-map', '0:a',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-c:a', 'copy',
        output_path
    ]

    print(f"  Running FFmpeg...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ERROR: FFmpeg failed")
        print(result.stderr[-3000:] if len(result.stderr) > 3000 else result.stderr)
        return False

    size = os.path.getsize(output_path)
    print(f"  Done: {output_path} ({size / (1024*1024):.1f} MB)")
    return True

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    for clip_num in range(1, 6):
        success = burn_subtitles(clip_num)
        if not success:
            print(f"Failed at clip {clip_num}")
            break

    print("\n" + "="*50)
    print("Final output files:")
    print("="*50)
    subprocess.run(['ls', '-lh', 'output/'])

if __name__ == '__main__':
    main()