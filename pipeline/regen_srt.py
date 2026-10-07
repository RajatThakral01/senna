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
    return generate_srt_for_ranges(transcript_segments,
                                   [(clip_start, clip_end)], clip_num)[0]


def generate_srt_for_ranges(transcript_segments, ranges, clip_num):
    """Generate SRT for ALL retained ranges with concat timeline mapping.

    Args:
        transcript_segments: transcript.json segments (with words[].start/end).
        ranges: list of (source_start, source_end) tuples in source time.
        clip_num: clip number (for logging).

    Returns:
        (srt_text, mapping) where mapping is a list of
        {source_start, source_end, output_start, output_end} describing the
        source->output timeline after gap removal (concatenation).

    Words are kept when fully inside any range (+0.5s end tolerance, as
    before) and retimed to output time so subtitles stay in sync after
    stitching.
    """
    log.debug("srt gen ranges clip=%s ranges=%s segs=%d", clip_num, ranges,
              len(transcript_segments))
    ranges = sorted((float(a), float(b)) for a, b in ranges)
    mapping, out_t = [], 0.0
    for a, b in ranges:
        d = b - a
        mapping.append({"source_start": a, "source_end": b,
                        "output_start": out_t, "output_end": out_t + d})
        out_t += d

    def to_output(src_t):
        for m in mapping:
            if m["source_start"] <= src_t <= m["source_end"] + 0.5:
                return src_t - m["source_start"] + m["output_start"]
        return None

    words = []
    for seg in transcript_segments:
        if 'words' in seg:
            for word in seg['words']:
                try:
                    ws, we = float(word['start']), float(word['end'])
                except (KeyError, TypeError, ValueError):
                    continue  # missing timestamps: skip gracefully
                inside = any(a <= ws and we <= b + 0.5 for a, b in ranges)
                if inside:
                    wtext = str(word.get("display", word.get("word", ""))).strip()
                    words.append({"word": wtext, "start": ws, "end": we})
    words.sort(key=lambda w: w["start"])

    srt_lines = []
    for i, word in enumerate(words, 1):
        out_s, out_e = to_output(word["start"]), to_output(word["end"])
        if out_s is None or out_e is None:
            continue
        text = str(word["word"]).strip()
        if not text:
            continue
        srt_lines.append(f"{i}")
        srt_lines.append(f"{sec_to_srt_time(max(0.0, out_s))} --> {sec_to_srt_time(max(0.0, out_e))}")
        srt_lines.append(text)
        srt_lines.append("")

    out = '\n'.join(srt_lines)
    log.debug("srt gen ranges clip=%s words=%d bytes=%d ranges=%d", clip_num,
              len(words), len(out), len(ranges))
    if not words:
        log.warning("srt gen clip=%s no words in ranges", clip_num)
    return out, mapping

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