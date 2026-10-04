#!/usr/bin/env python3
import subprocess
import json
import os

def escape_drawtext(text):
    text = text.replace('\\', '\\\\')
    text = text.replace("'", "\u2019")
    text = text.replace(':', '\\:')
    text = text.replace('%', '\\%')
    text = text.replace(',', '\\,')
    return text

def load_transcript():
    with open('transcripts/transcript.json') as f:
        return json.load(f)

def get_words_for_clip(transcript, clip_start, clip_end):
    """Extract word-level data for a clip's time window."""
    words_data = []

    for seg in transcript['segments']:
        seg_start = seg['start']
        seg_end = seg['end']

        # Skip segments outside clip window
        if seg_end < clip_start or seg_start > clip_end:
            continue

        seg_text = seg['text'].strip()

        # Use word-level data if available
        if 'words' in seg and seg['words']:
            words = seg['words']
            for w in words:
                word_start = w['start']
                word_end = w['end']

                # Adjust relative to clip start
                rel_start = max(0, word_start - clip_start)
                rel_end = max(0, word_end - clip_start)

                # Cap at 0.8s
                if rel_end - rel_start > 0.8:
                    rel_end = rel_start + 0.8

                words_data.append({
                    'word': w['word'],
                    'start': rel_start,
                    'end': rel_end,
                    'seg_start': seg_start - clip_start,
                    'seg_end': seg_end - clip_start,
                    'seg_text': seg_text
                })
        else:
            # Fallback: split text evenly
            tokens = seg_text.split()
            seg_duration = seg_end - seg_start
            if len(tokens) > 0 and seg_duration > 0:
                word_duration = seg_duration / len(tokens)
                for j, token in enumerate(tokens):
                    word_start = seg_start + j * word_duration
                    word_end = word_start + word_duration
                    rel_start = max(0, word_start - clip_start)
                    rel_end = max(0, word_end - clip_start)
                    if rel_end - rel_start > 0.8:
                        rel_end = rel_start + 0.8
                    words_data.append({
                        'word': token,
                        'start': rel_start,
                        'end': rel_end,
                        'seg_start': seg_start - clip_start,
                        'seg_end': seg_end - clip_start,
                        'seg_text': seg_text
                    })

    return words_data

def convert_to_vertical(input_path, output_path):
    """
    Creates 9:16 vertical video with blurred background.
    No cropping — full video visible in center.
    Blurred version fills the black bars top and bottom.
    """
    cmd = [
        os.path.expanduser('~/miniforge3/bin/ffmpeg'),
        '-y', '-i', input_path,
        '-filter_complex',
        (
            '[0:v]scale=1080:1920:force_original_aspect_ratio=increase,'
            'crop=1080:1920,'
            'gblur=sigma=20[bg];'
            '[0:v]scale=1080:-1[fg];'
            '[bg][fg]overlay=(W-w)/2:(H-h)/2[out]'
        ),
        '-map', '[out]',
        '-map', '0:a',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-c:a', 'copy',
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr[-500:]}")

def build_karaoke_filter(words_data):
    """Build filter_complex for karaoke effect - yellow word only."""
    chain = []
    prev_tag = "[0:v]"

    for i, wd in enumerate(words_data):
        word = wd['word'].upper()

        word_start = wd['start']
        word_end = wd['end']

        # Layer: yellow active word only
        f = (
            f"{prev_tag}drawtext="
            f"text={escape_drawtext(word)}"
            f":fontfile=/System/Library/Fonts/Helvetica.ttc"
            f":fontsize=95"
            f":fontcolor=0xFFE000"
            f":borderw=5"
            f":bordercolor=black"
            f":x=(w-text_w)/2"
            f":y=h-th-60"
            f":enable=between(t\\,{word_start:.3f}\\,{word_end:.3f})"
            f"[v{i+1}]"
        )
        chain.append(f)
        prev_tag = f"[v{i+1}]"

    final_tag = f"[v{len(words_data)}]" if words_data else "[0:v]"
    return ";".join(chain), final_tag

def process_clip(clip_num, clip_start, clip_end):
    """Process a single clip with karaoke subtitles."""
    input_mp4 = f"clips/clip_{clip_num}.mp4"
    vertical_mp4 = f"clips/clip_{clip_num}_vertical.mp4"
    final_output = f"output/clip_{clip_num}_final.mp4"

    print(f"\n=== Processing clip {clip_num} ===")

    # Step 1: Convert to vertical
    print(f"Converting to vertical...")
    if not os.path.exists(vertical_mp4):
        convert_to_vertical(input_mp4, vertical_mp4)
    else:
        print(f"  Using cached: {vertical_mp4}")

    # Step 2: Get word data
    transcript = load_transcript()
    words_data = get_words_for_clip(transcript, clip_start, clip_end)
    print(f"  Found {len(words_data)} words")

    if not words_data:
        print(f"  WARNING: No words found for clip {clip_num}")
        return False

    # Step 3: Build filter chain
    filter_chain, last_tag = build_karaoke_filter(words_data)

    # Step 4: Run FFmpeg
    print(f"Rendering karaoke subtitles...")
    cmd = [
        os.path.expanduser('~/miniforge3/bin/ffmpeg'),
        '-y',
        '-i', vertical_mp4,
        '-filter_complex', filter_chain,
        '-map', last_tag,
        '-map', '0:a',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-c:a', 'copy',
        final_output
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ERROR: FFmpeg failed")
        print(result.stderr[-2000:])
        return False

    size = os.path.getsize(final_output)
    print(f"  Done: {final_output} ({size / (1024*1024):.1f} MB)")
    return True

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    clips = [
        {'num': 1, 'start': 11*60+57, 'end': 12*60+54},
        {'num': 2, 'start': 29*60+1, 'end': 30*60+5},
        {'num': 3, 'start': 9*60+10, 'end': 10*60+3},
        {'num': 4, 'start': 25*60+39, 'end': 26*60+30},
        {'num': 5, 'start': 5*60+12, 'end': 6*60+9},
    ]

    # Clip 1 only
    clip = clips[0]
    # Remove old vertical to force regenerate
    if os.path.exists(f"clips/clip_{clip['num']}_vertical.mp4"):
        os.remove(f"clips/clip_{clip['num']}_vertical.mp4")
    if os.path.exists(f"output/clip_{clip['num']}_final.mp4"):
        os.remove(f"output/clip_{clip['num']}_final.mp4")
    success = process_clip(clip['num'], clip['start'], clip['end'])
    if not success:
        print(f"Clip 1 failed")
        return

    # Verify dimensions
    output_path = f"output/clip_{clip['num']}_final.mp4"
    size = os.path.getsize(output_path)
    print(f"\nClip 1 complete: {size / (1024*1024):.1f} MB")

if __name__ == '__main__':
    main()