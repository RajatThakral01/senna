#!/usr/bin/env python3
import json
import os
from logger import get_logger

log = get_logger("pipeline.regen_srt")

def sec_to_srt_time(sec):
    """Convert seconds to SRT time format HH:MM:SS,mmm."""
    hours = int(sec // 3600)
    minutes = int((sec % 3600) // 60)
    seconds = sec % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ',')

def generate_srt_for_clip(transcript_segments, clip_start, clip_end, clip_num):
    """Generate SRT content for a clip time range."""
    log.debug("srt gen clip=%s range=%.1f-%.1f segs=%d", clip_num, clip_start, clip_end, len(transcript_segments))
    words = []
    for seg in transcript_segments:
        if 'words' in seg:
            for word in seg['words']:
                if word['start'] >= clip_start and word['end'] <= clip_end + 0.5:
                    words.append(word)

    srt_lines = []
    for i, word in enumerate(words, 1):
        start_rel = word['start'] - clip_start
        end_rel = word['end'] - clip_start
        text = word['word'].strip()

        srt_lines.append(f"{i}")
        srt_lines.append(f"{sec_to_srt_time(start_rel)} --> {sec_to_srt_time(end_rel)}")
        srt_lines.append(text)
        srt_lines.append("")

    out = '\n'.join(srt_lines)
    log.debug("srt gen clip=%s words=%d bytes=%d", clip_num, len(words), len(out))
    if not words:
        log.warning("srt gen clip=%s no words in range", clip_num)
    return out

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    with open('transcripts/transcript.json') as f:
        data = json.load(f)

    clips = [
        {'num': 1, 'start': 11*60+57, 'end': 12*60+54},
        {'num': 2, 'start': 29*60+1, 'end': 30*60+5},
        {'num': 3, 'start': 9*60+10, 'end': 10*60+3},
        {'num': 4, 'start': 25*60+39, 'end': 26*60+30},
        {'num': 5, 'start': 5*60+12, 'end': 6*60+9},
    ]

    for clip in clips:
        srt_content = generate_srt_for_clip(data['segments'], clip['start'], clip['end'], clip['num'])
        srt_path = f"clips/clip_{clip['num']}.srt"
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        log.info("generated %s bytes=%d", srt_path, len(srt_content))

if __name__ == '__main__':
    main()