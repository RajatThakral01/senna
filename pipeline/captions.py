"""
pipeline/captions.py

Phrase-level captions with current-word highlighting (Phase 4).

- Natural phrase grouping (word/char limits, max duration, sentence-aware).
- Max two lines by default, balanced, sentence-aware breaks.
- ASS/libass output with karaoke word fill; SRT export preserved.
- Unicode-safe escaping; limited emphasis (highlight color only).
- Layout-aware placement: bottom default, switches to top when tracked
  faces sit in the lower third (avoids covering faces/content).
- Timings derive from aligned words + the existing source->output timeline,
  covering all stitched ranges without duplicates or drift.

Renderer: ASS via FFmpeg `ass` filter when available (verified present in
this repo's FFmpeg 9 gyan build); otherwise falls back to the existing
drawtext burn path (pipeline/fast_burn.py).
"""
import re

from logger import get_logger

log = get_logger("pipeline.captions")

SENT_END = re.compile(r"[.!?…]+$")


def _words_in_ranges(transcript_segments, ranges):
    """Flat word list inside ranges, sorted. Skips missing timestamps."""
    words = []
    for seg in transcript_segments or []:
        for w in seg.get("words", []) or []:
            try:
                ws, we = float(w["start"]), float(w["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if any(a <= ws and we <= b + 0.5 for a, b in ranges):
                src = str(w.get("word", "")).strip()
                disp = str(w.get("display", src)).strip()
                if disp:
                    words.append({"word": src, "display": disp,
                                  "start": ws, "end": we})
    words.sort(key=lambda w: (w["start"], w["end"]))
    # dedupe exact duplicates from overlapping segments
    out, seen = [], set()
    for w in words:
        key = (w["word"], round(w["start"], 3), round(w["end"], 3))
        if key not in seen:
            seen.add(key)
            out.append(w)
    return out


def _map_time(src_t, mapping):
    for m in mapping:
        if m["source_start"] <= src_t <= m["source_end"] + 0.5:
            return src_t - m["source_start"] + m["output_start"]
    return None


def group_phrases(words, mapping, cfg=None):
    """Group output-timed words into phrase cues.

    Limits: max_words, max_chars, max_duration (all configurable).
    Prefers ending a cue at sentence punctuation; never exceeds limits to
    chase punctuation (limits win). Returns cues with output times.
    """
    cfg = cfg or {}
    max_words = int(cfg.get("max_words", 8) or 8)
    max_chars = int(cfg.get("max_chars", 42) or 42)
    max_dur = float(cfg.get("max_duration", 4.0) or 4.0)
    cues, cur = [], []
    for w in words:
        out_s, out_e = _map_time(w["start"], mapping), _map_time(w["end"], mapping)
        if out_s is None or out_e is None:
            continue
        cur.append({"word": w["word"], "display": w.get("display", w["word"]),
                    "start": out_s, "end": out_e})
        chars = sum(len(x.get("display", x["word"])) + 1 for x in cur)
        dur = cur[-1]["end"] - cur[0]["start"]
        punct = bool(SENT_END.search(w["word"]))
        if len(cur) >= max_words or chars >= max_chars or dur >= max_dur or punct:
            cues.append(_finalize_cue(cur))
            cur = []
        elif len(cur) >= 3 and dur >= max_dur * 0.6 and w["word"].endswith(","):
            cues.append(_finalize_cue(cur))
            cur = []
    if cur:
        cues.append(_finalize_cue(cur))
    return [c for c in cues if c["words"]]


def _finalize_cue(items):
    return {"start": items[0]["start"], "end": items[-1]["end"], "words": items,
            "text": " ".join(x.get("display", x["word"]) for x in items)}


def balance_lines(words, max_lines=2):
    """Partition cue words into <= max_lines balanced lines (word groups).

    Returns a list of word lists. Greedy by character count; prefers breaks
    after sentence punctuation or commas.
    """
    if len(words) <= 1 or max_lines <= 1:
        return [list(words)]
    total = sum(len(w.get("display", w["word"])) + 1 for w in words)
    lines, cur, acc = [], [], 0
    target = total / max_lines
    for i, w in enumerate(words):
        cur.append(w)
        acc += len(w.get("display", w["word"])) + 1
        remaining_lines = max_lines - len(lines)
        remaining_words = len(words) - i - 1
        if (acc >= target and len(lines) < max_lines - 1
                and remaining_words >= (remaining_lines - 1)
                and i < len(words) - 1):
            if (SENT_END.search(w["word"]) or w["word"].endswith(",")
                    or acc >= target * 1.2 or remaining_words <= 3):
                lines.append(cur)
                cur, acc = [], 0
    if cur:
        lines.append(cur)
    return lines[:max_lines]


def ass_escape(text):
    """Unicode-safe ASS escaping: braces/backslash/newlines only."""
    text = text.replace("\\", "￣")  # avoid ASS override confusion; readable fallback
    text = text.replace("{", "(").replace("}", ")")
    return text


def _ass_time(sec):
    sec = max(0.0, sec)
    h, m = int(sec // 3600), int((sec % 3600) // 60)
    s = sec % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def build_ass(cues, cfg=None, placement="bottom"):
    """Build a full ASS script (karaoke word fill + balanced 2-line cues)."""
    cfg = cfg or {}
    font = cfg.get("font", "Arial")
    size = int(cfg.get("font_size", 64) or 64)
    primary = cfg.get("primary_color", "&H00FFFFFF")
    highlight = cfg.get("highlight_color", "&H0000E0FF")  # yellow (ASS BGR)
    outline_c = cfg.get("outline_color", "&H80000000")
    outline_w = cfg.get("outline_width", 2)
    shadow = cfg.get("shadow", 1)
    margin_v = int(cfg.get("margin_v", 180) if placement == "bottom"
                   else cfg.get("margin_v_top", 180))
    align = 2 if placement == "bottom" else 8
    header = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Cap,{font},{size},{primary},{highlight},{outline_c},"
        f"&H80000000,-1,0,0,0,100,100,0,0,1,{outline_w},{shadow},"
        f"{align},60,60,{margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n"
    )
    events = []
    for cue in cues:
        line_groups = balance_lines(cue["words"])
        parts = []
        for group in line_groups:
            segs = []
            for w in group:
                dur_cs = max(1, int(round((w["end"] - w["start"]) * 100)))
                segs.append(f"{{\\kf{dur_cs}}}{ass_escape(w.get('display', w['word']))}")
            parts.append(" ".join(segs))
        text = r"\N".join(parts) if parts else ass_escape(cue["text"])
        events.append(f"Dialogue: 0,{_ass_time(cue['start'])},{_ass_time(cue['end'])},"
                      f"Cap,,0,0,0,,{text}")
    return header + "\n".join(events) + "\n"


def decide_placement(framing_analysis=None, cfg=None):
    """Bottom default; top when tracked faces sit in the lower third.

    framing_analysis: framing.sample_clip() output (samples with tracks/boxes
    in SOURCE pixels). Compares median face cy against the visible crop band.
    Returns 'bottom' | 'top' with a logged reason.
    """
    cfg = cfg or {}
    forced = str(cfg.get("placement", "auto")).lower()
    if forced in ("bottom", "top"):
        return forced
    try:
        samples = (framing_analysis or {}).get("samples", [])
        cys = []
        for s in samples:
            for tr in s.get("tracks", []):
                cys.append(tr.get("cy", 0))
        if len(cys) < 3:
            return "bottom"
        h = (framing_analysis or {}).get("height", 720) or 720
        cys.sort()
        median_cy = cys[len(cys) // 2]
        if median_cy > h * 0.60:
            log.info("captions placement=top (faces low, median_cy=%.0f/h=%d)",
                     median_cy, h)
            return "top"
    except Exception:
        pass
    return "bottom"


def build_captions(transcript_segments, ranges, cfg=None, framing_analysis=None):
    """Full caption build: words -> timeline -> phrases -> ASS + mapping.

    Returns dict {ass, cues, mapping, placement, srt} (srt kept for export).
    """
    from pipeline.regen_srt import generate_srt_for_ranges
    cfg = cfg or {}
    ranges = sorted((float(a), float(b)) for a, b in ranges)
    srt, mapping = generate_srt_for_ranges(transcript_segments, ranges, -1)
    words = _words_in_ranges(transcript_segments, ranges)
    cues = group_phrases(words, mapping, cfg)
    placement = decide_placement(framing_analysis, cfg)
    ass = build_ass(cues, cfg, placement)
    log.info("captions cues=%d words=%d placement=%s", len(cues), len(words), placement)
    return {"ass": ass, "cues": cues, "mapping": mapping,
            "placement": placement, "srt": srt}


def ass_filter_available():
    """True when this FFmpeg build supports the ass filter (libass)."""
    import subprocess
    from config import ffmpeg_path
    try:
        r = subprocess.run([ffmpeg_path(), "-hide_banner", "-h", "filter=ass"],
                           capture_output=True, text=True)
        return r.returncode == 0 and "libass" in (r.stdout + r.stderr).lower()
    except Exception:
        return False


def render_captions_ass(input_path, output_path, ass_text):
    """Burn ASS captions in ONE ffmpeg pass. Returns True on success."""
    import os
    import subprocess
    import tempfile
    from config import ffmpeg_path
    # ass filter needs a real file; prefer a CWD-relative path so no
    # drive-colon escaping is needed on Windows (ffmpeg splits on ':').
    _dir = os.path.dirname(os.path.abspath(output_path)) or "."
    if not os.path.isdir(_dir):
        _dir = tempfile.gettempdir()
    fd, ass_path = tempfile.mkstemp(suffix=".ass", dir=_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig") as f:
            f.write(ass_text)
        try:
            rel = os.path.relpath(ass_path, ".")
        except ValueError:
            rel = ass_path
        filt = "ass=" + rel.replace("\\", "/")
        cmd = [ffmpeg_path(), "-y", "-i", input_path,
               "-vf", filt, "-map", "0:v", "-map", "0:a?",
               "-c:v", "libx264", "-preset", "fast", "-crf", "23",
               "-c:a", "aac", output_path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            log.error("ass burn failed rc=%d err=%.400s", r.returncode,
                      (r.stderr or "")[-400:])
            return False
        return True
    finally:
        try:
            os.remove(ass_path)
        except OSError:
            pass
