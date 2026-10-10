"""pipeline/compositor.py — apply a campaign edit recipe to one video (CAMPAIGN_PIPELINE_PLAN C3).

compose(input_path, recipe, output_path, platform) renders ONE deliverable:

  1. timing   — trim.start/end/max_duration (cuts are pulled back to the end of
                the last spoken word when a transcript is available, so a max
                duration never cuts a word in half)
  2. framing  — target aspect from the reframe op (else the platform preset);
                wide → 9:16 with method auto/face uses the face-tracking
                virtual camera (pipeline/framing.py); other aspect changes use
                a static crop centred on the main face (or the centre);
                fit_blur = fitted video over a blurred fill
  3. one ffmpeg pass:
       video  crop/scale → logos (size %, opacity, margin, timing) → one ASS
              file with captions + text overlays (libass: fonts, boxes, fades)
       audio  original (volume / mute / kept under a replacement) + replacement
              track (offset, loop, fades) + music (ducked under speech) →
              loudness-normalised mix; silent track when there is no audio
       export H.264/AAC per platform preset, faststart

Ops listed in recipe.LATER_OPS are skipped (reported in `skipped`).
Times in the recipe are on the OUTPUT timeline (0 = first frame delivered).
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
import time

from logger import get_logger
from pipeline import media

log = get_logger("pipeline.compositor")

DEFAULT_FONT_FAMILY = "Arial"
_FONT_DIRS = ("/System/Library/Fonts/Supplemental", "/Library/Fonts",
              "/usr/share/fonts/truetype/dejavu", "/usr/share/fonts")
ROLE_SIZE = {"hook": 78, "cta": 66, "handle": 48, "disclaimer": 36, "custom": 60}
ALIGN = {"top-left": 7, "top": 8, "top-center": 8, "top-right": 9, "center": 5,
         "bottom-left": 1, "bottom": 2, "bottom-center": 2, "bottom-right": 3}


class ComposeError(RuntimeError):
    pass


# ── helpers ──────────────────────────────────────────────────────────────────

def _ffmpeg():
    from config import ffmpeg_path
    return ffmpeg_path()


def _run(cmd, cwd=None, what="ffmpeg"):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0:
        raise ComposeError(f"{what} failed: {(r.stderr or '')[-800:]}")
    return r


def _ops(recipe):
    from campaign.recipe import LATER_OPS
    ops = [o for o in (recipe or {}).get("ops") or [] if isinstance(o, dict)]
    single = {o["op"]: o for o in ops if o["op"] not in ("logo", "text_overlay")}
    return (single, [o for o in ops if o["op"] == "logo"],
            [o for o in ops if o["op"] == "text_overlay"],
            sorted({o["op"] for o in ops if o["op"] in LATER_OPS}))


def _window(op, dur):
    """(start, end) on the output timeline for a timed overlay."""
    if op.get("from_end"):
        return max(0.0, dur - float(op["from_end"])), dur
    a = float(op.get("start") or 0.0)
    b = float(op["end"]) if op.get("end") is not None else dur
    return min(a, dur), min(max(b, a), dur)


def ass_color(hex_color, alpha=0):
    """#RRGGBB → ASS &HAABBGGRR."""
    h = str(hex_color or "#FFFFFF").lstrip("#")
    if len(h) != 6:
        h = "FFFFFF"
    return f"&H{alpha:02X}{h[4:6]}{h[2:4]}{h[0:2]}".upper()


def font_family(path):
    """Family name of a font file (what libass matches on), or None."""
    try:
        from PIL import ImageFont
        return ImageFont.truetype(path, 20).getname()[0]
    except Exception:
        return None


# ── timing ───────────────────────────────────────────────────────────────────

def plan_timing(duration, trim=None, words=None):
    """(start, out_duration, cut_info). words = [{start, end}] in SOURCE seconds."""
    trim = trim or {}
    start = min(max(0.0, float(trim.get("start") or 0.0)), max(0.0, duration - 0.5))
    end = float(trim["end"]) if trim.get("end") is not None else duration
    end = min(max(end, start + 0.5), duration)
    info = {"requested_max": trim.get("max_duration"), "snapped_to_word": False}
    mx = trim.get("max_duration")
    if mx and end - start > float(mx):
        end = start + float(mx)
        if words:
            # never cut a word in half: end after the last word that fits
            fit = [w for w in words if start <= w["start"] and w["end"] <= end]
            if fit and fit[-1]["end"] > start + 1.0:
                end = min(fit[-1]["end"] + 0.25, start + float(mx))
                info["snapped_to_word"] = True
    out = round(end - start, 3)
    mn = trim.get("min_duration")
    info["too_short"] = bool(mn and out + 0.05 < float(mn))
    return start, out, info


# ── framing ──────────────────────────────────────────────────────────────────

def _static_crop(src_w, src_h, tgt_aspect, cx=None, cy=None, headroom=0.38):
    """Largest crop of the target aspect, centred on (cx, cy) if given."""
    if src_w / src_h > tgt_aspect:
        ch, cw = src_h, int(round(src_h * tgt_aspect))
    else:
        cw, ch = src_w, int(round(src_w / tgt_aspect))
    cw, ch = cw - cw % 2, ch - ch % 2
    cx = src_w / 2 if cx is None else cx
    cy = src_h / 2 if cy is None else cy
    x = int(min(max(cx - cw / 2, 0), src_w - cw))
    y = int(min(max(cy - ch * headroom, 0), src_h - ch)) if cy != src_h / 2 else (src_h - ch) // 2
    return cw, ch, x, y


def _subject_center(path, cfg):
    """Median position of the most prominent face in a clip, or (None, None)."""
    try:
        from pipeline import framing as F
        a = F.sample_clip(path, dict(cfg or {}, active_speaker=False, saliency_fallback=False))
        samples = a.get("samples") or []
        tid = F._scene_subject(samples, a.get("width"))
        pts = [(tr["cx"], tr["cy"]) for s in samples for tr in s.get("tracks", [])
               if tr["id"] == tid]
        if not pts:
            return None, None
        xs = sorted(p[0] for p in pts)
        ys = sorted(p[1] for p in pts)
        return xs[len(xs) // 2], ys[len(ys) // 2]
    except Exception:
        log.debug("subject centre failed", exc_info=True)
        return None, None


# ── ASS (captions + text overlays) ───────────────────────────────────────────

def _style(name, font, size, primary, secondary, outline_c, back_c, bold, border_style,
           outline, shadow, align, ml, mr, mv):
    return (f"Style: {name},{font},{size},{primary},{secondary},{outline_c},{back_c},"
            f"{-1 if bold else 0},0,0,0,100,100,0,0,{border_style},{outline},{shadow},"
            f"{align},{ml},{mr},{mv},1")


def build_overlay_ass(W, H, dur, cues=None, cap_op=None, overlays=(), fonts=None):
    """One ASS script for captions (cue list from captions.group_phrases) and
    text overlays, sized to the output. fonts maps font name → family."""
    from pipeline.captions import ass_escape, balance_lines, _ass_time
    fonts = fonts or {}
    k = min(W, H) / 1080.0
    styles, events = [], []
    cap_top_from_bottom = 0          # px from the bottom edge to the caption block's top
    # bottom safe zone: phone UIs cover ~20% of a vertical frame, little of a wide one
    low = 0.22 if W / H < 0.9 else 0.12 if W / H < 1.3 else 0.07
    if cap_op and cap_op.get("enabled", True) and cues:
        fam = fonts.get(cap_op.get("font")) or cap_op.get("font") or DEFAULT_FONT_FAMILY
        size = int(round((cap_op.get("size") or 64) * k))
        primary = ass_color(cap_op.get("color") or "#FFFFFF")
        highlight = ass_color(cap_op.get("highlight_color") or "#FFE000")
        style = cap_op.get("style") or "karaoke"
        pos = cap_op.get("position") or "auto"
        align = {"top": 8, "center": 5}.get(pos, 2)
        mv = int(H * (0.12 if align == 8 else low if align == 2 else 0))
        if align == 2:   # room for two caption lines + a gap
            cap_top_from_bottom = mv + int(size * 2.5) + int(H * 0.02)
        boxed = style == "boxed"
        styles.append(_style("Cap", fam, size, primary, highlight, "&H00000000",
                             "&H80000000" if boxed else "&H64000000", True,
                             3 if boxed else 1, int(round((8 if boxed else 4) * k)),
                             0 if boxed else 1, align, int(W * 0.07), int(W * 0.07), mv))
        up = bool(cap_op.get("uppercase"))
        for cue in cues:
            lines = []
            for group in balance_lines(cue["words"]):
                segs = []
                for w in group:
                    word = ass_escape(w.get("display", w["word"]))
                    word = word.upper() if up else word
                    if style == "karaoke":
                        cs = max(1, int(round((w["end"] - w["start"]) * 100)))
                        segs.append(f"{{\\kf{cs}}}{word}")
                    else:
                        segs.append(word)
                lines.append(" ".join(segs))
            if lines:
                events.append(f"Dialogue: 0,{_ass_time(cue['start'])},"
                              f"{_ass_time(min(cue['end'], dur))},Cap,,0,0,0,,"
                              + r"\N".join(lines))
    for i, o in enumerate(overlays):
        a, b = _window(o, dur)
        if b - a < 0.05 or not o.get("text"):
            continue
        role = o.get("role") or "custom"
        fam = fonts.get(o.get("font")) or o.get("font") or DEFAULT_FONT_FAMILY
        size = int(round((o.get("size") or ROLE_SIZE.get(role, 60)) * k))
        align = ALIGN.get(o.get("position") or "center", 5)
        box = o.get("box", True)
        mv = int(H * 0.10)
        if align in (1, 2, 3):
            mv = max(int(H * (low - 0.04)), cap_top_from_bottom)
        a8 = int(round((1 - float(o["opacity"])) * 255)) if o.get("opacity") is not None else 0
        styles.append(_style(f"T{i}", fam, size, ass_color(o.get("color") or "#FFFFFF", a8),
                             ass_color(o.get("color") or "#FFFFFF", a8),
                             f"&H{max(a8, 0):02X}000000",
                             "&H59000000" if box else "&H00000000", True,
                             3 if box else 1, int(round((14 if box else 4) * k)),
                             0, align, int(W * 0.06), int(W * 0.06), mv))
        anim = o.get("animation") or "fade"
        tag = ("{\\fad(250,250)}" if anim == "fade" else
               "{\\fscx70\\fscy70\\t(0,140,\\fscx108\\fscy108)\\t(140,240,\\fscx100\\fscy100)}"
               if anim == "pop" else "")
        text = ass_escape(str(o["text"])).replace("\n", r"\N")
        events.append(f"Dialogue: 1,{_ass_time(a)},{_ass_time(b)},T{i},,0,0,0,,{tag}{text}")
    if not events:
        return None
    header = ("[Script Info]\nScriptType: v4.00+\n"
              f"PlayResX: {W}\nPlayResY: {H}\nWrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
              "[V4+ Styles]\n"
              "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
              "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, "
              "Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, "
              "MarginV, Encoding\n" + "\n".join(styles) + "\n\n"
              "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
              "Effect, Text\n")
    return header + "\n".join(events) + "\n"


# ── transcript (captions) ────────────────────────────────────────────────────

def transcribe_segment(input_path, start, dur, work_dir, keep_models=True):
    """WhisperX transcript of the delivered segment (times relative to it)."""
    from pipeline.transcriber import transcribe_audio
    wav = os.path.join(work_dir, "segment.wav")
    _run([_ffmpeg(), "-y", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", input_path,
          "-vn", "-ac", "1", "-ar", "16000", wav], what="audio extract")
    _, result = transcribe_audio(wav, output_dir=work_dir, keep_models=keep_models)
    return (result or {}).get("segments") or []


def _flat_words(segments):
    return [w for s in segments or [] for w in s.get("words") or []
            if "start" in w and "end" in w]


# ── colour / bookends (C8) ───────────────────────────────────────────────────

COLOR_PRESETS = {
    "vivid": "eq=saturation=1.35:contrast=1.06",
    "warm": "colorbalance=rm=0.07:gm=0.01:bm=-0.07,eq=saturation=1.05",
    "cool": "colorbalance=rm=-0.06:bm=0.07",
    "bw": "hue=s=0",
}


def color_filter(op):
    """ffmpeg filter string for a color op (preset, then eq tweaks), or None."""
    if not op:
        return None
    parts = []
    if op.get("preset") in COLOR_PRESETS:
        parts.append(COLOR_PRESETS[op["preset"]])
    b, c, s = (float(op.get("brightness") or 0.0), float(op.get("contrast") or 1.0),
               float(op.get("saturation") if op.get("saturation") is not None else 1.0))
    if abs(b) > 1e-3 or abs(c - 1) > 1e-3 or abs(s - 1) > 1e-3:
        parts.append(f"eq=brightness={b:.3f}:contrast={c:.3f}:saturation={s:.3f}")
    return ",".join(parts) or None


FADE = 0.35   # s, "fade" transition into / out of the body


def _fit(W, H):
    return (f"split=2[bg][fg];[bg]scale={W}:{H}:force_original_aspect_ratio=increase,"
            f"crop={W}:{H},boxblur=24:2[bgb];[fg]scale={W}:{H}:"
            f"force_original_aspect_ratio=decrease[fgs];[bgb][fgs]overlay=(W-w)/2:(H-h)/2")


def _render_piece(kind, op, W, H, fps, pre, work, fonts):
    """Intro / outro / end card as a W×H clip with stereo 48 kHz audio. → (path, dur)."""
    out = os.path.abspath(os.path.join(work, f"{kind}.mp4"))   # ffmpeg runs with cwd=work
    asset = op.get("asset_path")
    info = media.probe(asset) if asset else {"exists": False}
    is_video = info.get("exists") and info.get("kind") == "video" and info.get("has_video")
    loud = f"loudnorm=I={pre['loudness_lufs']}:TP=-1.5:LRA=11"
    cmd = [_ffmpeg(), "-y"]
    if kind == "end_card":
        d = float(op.get("duration") or 2.0)
        cmd += ["-f", "lavfi", "-t", f"{d:.3f}", "-i", f"color=c=0x101010:s={W}x{H}:r={fps:.3f}"]
        graph = "[0:v]null[b0]"
        if info.get("exists"):
            cmd += ["-loop", "1", "-t", f"{d:.3f}", "-i", os.path.abspath(asset)]
            iw = int(W * 0.5)
            graph += (f";[1:v]scale={iw}:-1,format=rgba[img];"
                      f"[b0][img]overlay=(W-w)/2:H*0.40-h/2:shortest=1[b1]")
            cur = "b1"
        else:
            cur = "b0"
        ass = build_overlay_ass(W, H, d, None, None,
                                [{"text": op["text"], "role": "cta", "size": 92,
                                  "position": "bottom" if info.get("exists") else "center",
                                  "box": False, "animation": "fade"}] if op.get("text") else [],
                                fonts)
        if ass:
            with open(os.path.join(work, "end_card.ass"), "w", encoding="utf-8") as f:
                f.write(ass)
            fd = ":fontsdir=fonts" if fonts else ""
            graph += f";[{cur}]ass=filename=end_card.ass{fd}[b2]"
            cur = "b2"
        graph += f";[{cur}]fade=t=in:d=0.25,format=yuv420p[v]"
        a_in = len([x for x in cmd if x == "-i"])
        cmd += ["-f", "lavfi", "-t", f"{d:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
        amap = f"{a_in}:a"
    elif is_video:
        d = min(float(info["duration"] or 0), 30.0)
        cmd += ["-t", f"{d:.3f}", "-i", os.path.abspath(asset)]
        graph = f"[0:v]{_fit(W, H)},fps={fps:.3f},setsar=1,format=yuv420p[v]"
        if info.get("has_audio"):
            graph += f";[0:a]aresample=48000,aformat=channel_layouts=stereo,{loud}[a]"
            amap = "[a]"
        else:
            cmd += ["-f", "lavfi", "-t", f"{d:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
            amap = "1:a"
    elif info.get("exists"):          # still image (or a video file without frames)
        d = float(op.get("duration") or 2.0)
        cmd += ["-loop", "1", "-t", f"{d:.3f}", "-i", os.path.abspath(asset),
                "-f", "lavfi", "-t", f"{d:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
        graph = f"[0:v]{_fit(W, H)},fps={fps:.3f},setsar=1,format=yuv420p[v]"
        amap = "1:a"
    else:
        raise ComposeError(f"{kind}: file missing ({op.get('asset')})")
    cmd += ["-filter_complex", graph, "-map", "[v]", "-map", amap, "-t", f"{d:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", out]
    _run(cmd, cwd=work, what=kind)
    return out, media.probe(out)["duration"] or d


def add_bookends(body, output_path, single, W, H, pre, work, fonts, applied, warnings):
    """intro + body + outro + end card → output_path. Returns (body_offset, total)."""
    fps = media.probe(body).get("fps") or 30.0
    bdur = media.probe(body)["duration"] or 0.0
    pieces, offset = [], 0.0
    for kind in ("intro", "body", "outro", "end_card"):
        if kind == "body":
            pieces.append(("body", body, bdur))
            continue
        op = single.get(kind)
        if not op:
            continue
        try:
            path, d = _render_piece(kind, op, W, H, fps, pre, work, fonts)
            pieces.append((kind, path, d))
            if kind == "intro":
                offset = d
            applied.append(kind)
        except Exception as e:
            log.exception("%s failed", kind)
            warnings.append(f"{kind} skipped: {e}")
    if len(pieces) == 1:
        shutil.copy2(body, output_path)
        return 0.0, bdur
    cmd = [_ffmpeg(), "-y"]
    for _, path, _ in pieces:
        cmd += ["-i", os.path.abspath(path)]
    fade_in = (single.get("intro") or {}).get("transition") == "fade" and pieces[0][0] == "intro"
    fade_out = (single.get("outro") or {}).get("transition") == "fade"
    parts, labels = [], ""
    for i, (kind, _, d) in enumerate(pieces):
        v, a = f"[{i}:v]settb=AVTB,setsar=1", f"[{i}:a]aresample=48000"
        if kind == "body" and fade_in:
            v += f",fade=t=in:d={FADE}"
            a += f",afade=t=in:d={FADE}"
        if kind == "body" and fade_out and any(k == "outro" for k, _, _ in pieces):
            v += f",fade=t=out:st={max(0.0, d - FADE):.3f}:d={FADE}"
            a += f",afade=t=out:st={max(0.0, d - FADE):.3f}:d={FADE}"
        if kind == "intro" and fade_in:
            v += f",fade=t=out:st={max(0.0, d - FADE):.3f}:d={FADE}"
            a += f",afade=t=out:st={max(0.0, d - FADE):.3f}:d={FADE}"
        if kind == "outro" and fade_out:
            v += f",fade=t=in:d={FADE}"
            a += f",afade=t=in:d={FADE}"
        parts += [f"{v}[v{i}]", f"{a}[a{i}]"]
        labels += f"[v{i}][a{i}]"
    parts.append(f"{labels}concat=n={len(pieces)}:v=1:a=1[v][a]")
    cmd += ["-filter_complex", ";".join(parts), "-map", "[v]", "-map", "[a]",
            "-c:v", pre["video_codec"], "-preset", pre["preset"], "-crf", str(pre["crf"]),
            "-pix_fmt", pre["pix_fmt"], "-c:a", pre["audio_codec"], "-b:a", pre["audio_bitrate"],
            "-ar", "48000", "-movflags", "+faststart", os.path.abspath(output_path)]
    _run(cmd, cwd=work, what="bookends")
    return offset, sum(d for _, _, d in pieces)


# ── main entry ───────────────────────────────────────────────────────────────

def compose(input_path, recipe, output_path, platform="generic", work_dir=None,
            transcript_segments=None, transcript_offset=0.0, cfg=None):
    """Render one deliverable. Returns a result dict (see module docstring).

    transcript_segments: optional WhisperX-style segments of the INPUT file
    (e.g. clip mode passes the clip's words), shifted by -transcript_offset so
    that time 0 is the input's first frame. Without it, captions transcribe
    the delivered segment.
    """
    from campaign.presets import preset, output_size
    from config import get_config
    t0 = time.monotonic()
    cfg = cfg or get_config()
    info = media.probe(input_path)
    if not info["has_video"]:
        raise ComposeError(f"no video stream in {input_path}")
    single, logos, overlays, skipped = _ops(recipe)
    pre = preset(platform)
    aspect = (single.get("reframe") or {}).get("aspect") or pre["aspect"]
    W, H = output_size(aspect)
    own_work = work_dir is None
    work = work_dir or tempfile.mkdtemp(prefix="compose_")
    os.makedirs(work, exist_ok=True)
    warnings, applied = [], []

    words_src = None
    if transcript_segments is not None:
        shifted = []
        for s in transcript_segments:
            ws = [dict(w, start=w["start"] - transcript_offset, end=w["end"] - transcript_offset)
                  for w in s.get("words") or [] if "start" in w and "end" in w]
            if ws:
                shifted.append({"start": ws[0]["start"], "end": ws[-1]["end"], "words": ws,
                                "text": " ".join(w.get("word", "") for w in ws)})
        transcript_segments = shifted
        words_src = _flat_words(shifted)

    start, dur, cut = plan_timing(info["duration"], single.get("trim"), words_src)
    if single.get("trim"):
        applied.append("trim")
    if cut["too_short"]:
        warnings.append(f"video is {dur:.1f}s, shorter than the required "
                        f"{single['trim']['min_duration']:g}s")
    sp = single.get("speed")
    if sp and abs(float(sp.get("factor") or 1) - 1) > 0.01:
        f = float(sp["factor"])
        fast = os.path.join(work, "speed.mp4")
        filt = f"[0:v]setpts=PTS/{f:.4f}[v]"
        amap = []
        if info["has_audio"]:
            filt += f";[0:a]atempo={f:.4f}[a]"
            amap = ["-map", "[a]", "-c:a", "aac", "-b:a", "192k"]
        _run([_ffmpeg(), "-y", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", input_path,
              "-filter_complex", filt, "-map", "[v]", *amap, "-c:v", "libx264",
              "-preset", "veryfast", "-crf", "16", fast], what="speed")
        if transcript_segments is not None:   # words onto the sped-up timeline
            transcript_segments = [
                dict(sg, start=(sg["start"] - start) / f, end=(sg["end"] - start) / f,
                     words=[dict(w, start=(w["start"] - start) / f, end=(w["end"] - start) / f)
                            for w in sg["words"]])
                for sg in transcript_segments if sg["end"] > start and sg["start"] < start + dur]
        input_path, info = fast, media.probe(fast)
        start, dur = 0.0, min(dur / f, info["duration"] or dur / f)
        applied.append("speed")
    if pre.get("max_duration") and dur > pre["max_duration"]:
        warnings.append(f"{dur:.0f}s exceeds the {pre['label']} preset limit "
                        f"({pre['max_duration']}s)")

    # ── framing ──
    src_path, src_offset = input_path, start
    vf = []
    src_w, src_h = info["width"], info["height"]
    tgt = W / H
    reframe = single.get("reframe") or {}
    method = reframe.get("method") or "auto"
    mismatch = abs(src_w / src_h - tgt) / tgt > 0.02
    framing_kind = "scale"
    # a crop that would keep < half the picture (e.g. vertical → 16:9) loses the
    # subject and upscales a thin strip: auto falls back to fit-over-blur
    kept = min(src_w / src_h, tgt) / max(src_w / src_h, tgt)
    if mismatch and method == "auto" and kept < 0.5 and tgt > src_w / src_h:
        method = "fit_blur"
        warnings.append(f"source is {media.aspect_label(src_w, src_h)}; {aspect} output uses "
                        "a blurred background instead of cropping away "
                        f"{100 - kept * 100:.0f}% of the picture")
    if mismatch:
        if method == "fit_blur":
            framing_kind = "fit_blur"
        elif method in ("auto", "face") and abs(tgt - 9 / 16) < 0.01 and src_w / src_h > tgt:
            seg = os.path.join(work, "segment.mp4")
            _run([_ffmpeg(), "-y", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", input_path,
                  "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-c:a", "aac",
                  "-b:a", "192k", seg], what="segment cut")
            from pipeline.framing import render_vertical
            vert = os.path.join(work, "segment_vertical.mp4")
            res = render_vertical(seg, vert, layout="auto", cfg=dict(cfg.get("framing", {}),
                                                                       debug_video=False))
            src_path, src_offset, framing_kind = vert, 0.0, f"virtual camera ({res['layout']})"
            src_w, src_h = 1080, 1920
        else:
            cx = cy = None
            if method in ("auto", "face"):
                probe_seg = os.path.join(work, "probe_segment.mp4")
                _run([_ffmpeg(), "-y", "-ss", f"{start:.3f}", "-t", f"{min(dur, 30):.3f}",
                      "-i", input_path, "-an", "-c:v", "libx264", "-preset", "ultrafast",
                      "-crf", "28", probe_seg], what="probe cut")
                cx, cy = _subject_center(probe_seg, cfg.get("framing", {}))
            cw, ch, x, y = _static_crop(src_w, src_h, tgt, cx, cy)
            vf.append(f"crop={cw}:{ch}:{x}:{y}")
            framing_kind = "face-centred crop" if cx is not None else "centre crop"
        if single.get("reframe"):
            applied.append("reframe")
    if framing_kind == "fit_blur":
        vf.append(f"split=2[bg][fg];[bg]scale={W}:{H}:force_original_aspect_ratio=increase,"
                  f"crop={W}:{H},boxblur=24:2[bgb];[fg]scale={W}:{H}:"
                  f"force_original_aspect_ratio=decrease[fgs];[bgb][fgs]overlay=(W-w)/2:(H-h)/2")
    else:
        vf.append(f"scale={W}:{H}:flags=lanczos")
    vf.append("setsar=1")
    col = color_filter(single.get("color"))
    if col:
        vf.append(col)
        applied.append("color")
    if info["fps"] and info["fps"] > pre["max_fps"] + 0.5:
        vf.append(f"fps={pre['max_fps']}")

    # ── captions / text (one ASS) ──
    cap_op = single.get("captions")
    cues = None
    fonts, font_dir = {}, os.path.join(work, "fonts")
    for o in [cap_op] + overlays:
        if o and o.get("font_path") and os.path.isfile(o["font_path"]):
            os.makedirs(font_dir, exist_ok=True)
            shutil.copy2(o["font_path"], font_dir)
            fonts[o["font"]] = font_family(o["font_path"]) or DEFAULT_FONT_FAMILY
    rep_track = (single.get("audio.replace") or {}).get("track_path")
    if cap_op and cap_op.get("enabled", True):
        if not info["has_audio"] and not (rep_track and os.path.isfile(rep_track)):
            warnings.append("captions skipped: the video has no speech audio")
        else:
            try:
                segs = transcript_segments
                seg_range = (start, start + dur)
                rep = single.get("audio.replace") or {}
                if (rep_track and os.path.isfile(rep_track)
                        and not (rep.get("keep_original") and info["has_audio"])):
                    # the original voice is gone: caption what the viewer hears
                    tlen = media.probe(rep_track).get("duration") or dur
                    t0 = float(rep.get("start") or 0)
                    segs = transcribe_segment(rep_track, t0,
                                              max(0.1, min(dur, tlen - t0)), work)
                    seg_range = (0.0, dur)
                    log.info("captions transcribed from the replacement track %s",
                             os.path.basename(rep_track))
                elif segs is None:
                    segs = transcribe_segment(input_path, start, dur, work)
                    seg_range = (0.0, dur)
                from pipeline.captions import group_phrases, _words_in_ranges
                from pipeline.regen_srt import generate_srt_for_ranges
                _, mapping = generate_srt_for_ranges(segs, [seg_range], -1)
                cues = group_phrases(_words_in_ranges(segs, [seg_range]), mapping,
                                     cfg.get("captions", {}))
                if cues:
                    applied.append("captions")
                else:
                    warnings.append("captions: no speech found in the video")
            except Exception as e:
                log.exception("captions failed")
                warnings.append(f"captions failed: {e}")
    wm = single.get("watermark")
    draw = list(overlays)
    if wm and wm.get("text"):
        draw.append({"text": wm["text"], "role": "watermark", "position": wm.get("position"),
                     "size": 40, "box": False, "animation": "none", "opacity": wm.get("opacity")})
    ass = build_overlay_ass(W, H, dur, cues, cap_op, draw, fonts)
    if overlays and ass:
        applied.append("text_overlay")
    if wm and ass:
        applied.append("watermark")
    ass_name = None
    if ass:
        ass_name = "overlay.ass"
        with open(os.path.join(work, ass_name), "w", encoding="utf-8") as f:
            f.write(ass)

    # ── inputs ──
    cmd = [_ffmpeg(), "-y", "-ss", f"{src_offset:.3f}", "-t", f"{dur:.3f}",
           "-i", os.path.abspath(src_path)]
    n_in = 1
    replace = single.get("audio.replace")
    music = single.get("audio.add_music")
    tracks = {}
    for key, op, loop in (("replace", replace, (replace or {}).get("loop", True)),
                          ("music", music, True)):
        if not op:
            continue
        path = op.get("track_path")
        if not path or not os.path.isfile(path):
            warnings.append(f"audio.{key}: track file missing ({op.get('track')}); skipped")
            continue
        if loop:
            cmd += ["-stream_loop", "-1"]
        if op.get("start"):
            cmd += ["-ss", f"{float(op['start']):.3f}"]
        cmd += ["-i", os.path.abspath(path)]
        tracks[key] = n_in
        n_in += 1
    logo_inputs = []
    for lo in logos:
        path = lo.get("asset_path")
        if not path or not os.path.isfile(path):
            warnings.append(f"logo: file missing ({lo.get('asset')}); skipped")
            continue
        cmd += ["-loop", "1", "-i", os.path.abspath(path)]
        logo_inputs.append((n_in, lo))
        n_in += 1

    # ── video graph ──
    graph = [f"[0:v]{','.join(vf)}[v0]"]
    cur = "v0"
    for i, (idx, lo) in enumerate(logo_inputs):
        lw = max(16, int(round(W * float(lo.get("size_pct") or 12) / 100.0)))
        m = int(round(W * float(lo.get("margin_pct") if lo.get("margin_pct") is not None else 3) / 100.0))
        pos = lo.get("position") or "top-right"
        x = {"left": f"{m}", "right": f"W-w-{m}"}.get(
            "left" if "left" in pos else "right" if "right" in pos else "c", "(W-w)/2")
        y = f"{m}" if pos.startswith("top") else f"H-h-{m}" if pos.startswith("bottom") else "(H-h)/2"
        a, b = _window(lo, dur)
        graph.append(f"[{idx}:v]scale={lw}:-1,format=rgba,"
                     f"colorchannelmixer=aa={float(lo.get('opacity') or 1.0):.3f}[lg{i}]")
        graph.append(f"[{cur}][lg{i}]overlay=x={x}:y={y}:shortest=1:"
                     f"enable='between(t,{a:.3f},{b:.3f})'[vl{i}]")
        cur = f"vl{i}"
        if "logo" not in applied:
            applied.append("logo")
    if ass_name:
        fd = ":fontsdir=fonts" if fonts else ""
        graph.append(f"[{cur}]ass=filename={ass_name}{fd}[vass]")
        cur = "vass"
    graph.append(f"[{cur}]format=yuv420p[vout]")

    # ── audio graph ──
    orig_level = 1.0
    if single.get("audio.volume"):
        orig_level = float(single["audio.volume"].get("level", 1.0))
        applied.append("audio.volume")
    if single.get("audio.mute_original"):
        orig_level = 0.0
        applied.append("audio.mute_original")
    if replace and "replace" in tracks:
        orig_level = float(replace.get("keep_original") or 0.0)
        applied.append("audio.replace")
    labels = []
    src_has_audio = info["has_audio"] if src_path == input_path else media.probe(src_path)["has_audio"]
    has_orig = src_has_audio and orig_level > 0.001
    music_duck = music and "music" in tracks and music.get("duck", True) and has_orig
    if has_orig:
        graph.append(f"[0:a]aresample=48000,aformat=channel_layouts=stereo,"
                     f"volume={orig_level:.3f}" + ("[orig];[orig]asplit=2[o][okey]" if music_duck else "[o]"))
        labels.append("o")
    fade_tail = max(0.0, dur - 0.25)
    if "replace" in tracks:
        fi, fo = float(replace.get("fade_in") or 0), float(replace.get("fade_out") or 0)
        ch = (f"[{tracks['replace']}:a]aresample=48000,aformat=channel_layouts=stereo,"
              f"atrim=0:{dur:.3f},asetpts=N/SR/TB,volume={float(replace.get('volume') or 1.0):.3f}")
        if fi:
            ch += f",afade=t=in:d={fi:.2f}"
        if fo:
            ch += f",afade=t=out:st={max(0.0, dur - fo):.3f}:d={fo:.2f}"
        graph.append(ch + "[r]")
        labels.append("r")
    if "music" in tracks:
        fo = float(music.get("fade_out") or 0)
        ch = (f"[{tracks['music']}:a]aresample=48000,aformat=channel_layouts=stereo,"
              f"atrim=0:{dur:.3f},asetpts=N/SR/TB,volume={float(music.get('volume') or 0.15):.3f}")
        if fo:
            ch += f",afade=t=out:st={max(0.0, dur - fo):.3f}:d={fo:.2f}"
        if music_duck:
            graph.append(ch + "[m0]")
            graph.append("[m0][okey]sidechaincompress=threshold=0.02:ratio=8:attack=20:release=400[m]")
        else:
            graph.append(ch + "[m]")
        labels.append("m")
        applied.append("audio.add_music")
    loud = f"loudnorm=I={pre['loudness_lufs']}:TP=-1.5:LRA=11,alimiter=limit=0.95"
    if labels:
        mix = (f"{''.join('[' + x + ']' for x in labels)}amix=inputs={len(labels)}:"
               f"duration=longest:normalize=0" if len(labels) > 1 else f"[{labels[0]}]anull")
        graph.append(f"{mix},{loud},afade=t=out:st={fade_tail:.3f}:d=0.25,"
                     f"apad,atrim=0:{dur:.3f}[aout]")
        amap = "[aout]"
    else:
        cmd += ["-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
        amap = f"{n_in}:a"
        warnings.append("no audio in the result; a silent track was added")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    ends = [single[k] for k in ("intro", "outro", "end_card") if single.get(k)]
    final_path = output_path
    if ends:
        output_path = os.path.join(work, "body.mp4")
    cmd += ["-filter_complex", ";".join(graph), "-map", "[vout]", "-map", amap,
            "-c:v", pre["video_codec"], "-preset", pre["preset"], "-crf", str(pre["crf"]),
            "-pix_fmt", pre["pix_fmt"], "-c:a", pre["audio_codec"], "-b:a", pre["audio_bitrate"],
            "-ar", "48000", "-movflags", "+faststart", "-t", f"{dur:.3f}",
            os.path.abspath(output_path)]
    log.info("compose %s → %s platform=%s %dx%d dur=%.1fs framing=%s ops=%s skipped=%s",
             os.path.basename(input_path), os.path.basename(output_path), pre["platform"], W, H,
             dur, framing_kind, applied, skipped)
    try:
        _run(cmd, cwd=work, what="compose")
    finally:
        with open(os.path.join(work, "compose_cmd.json"), "w") as f:
            json.dump(cmd, f, indent=1)
    body_offset, total = 0.0, dur
    if ends:
        body_offset, total = add_bookends(output_path, final_path, single, W, H, pre, work,
                                          fonts, applied, warnings)
        output_path = final_path
    out = media.probe(output_path)
    if own_work:
        shutil.rmtree(work, ignore_errors=True)
    return {"output_path": output_path, "platform": pre["platform"], "width": out["width"],
            "height": out["height"], "duration": out["duration"], "expected_duration": total,
            "body_offset": body_offset, "body_duration": dur,
            "source_start": start, "framing": framing_kind, "applied": applied,
            "skipped": skipped, "warnings": warnings, "cut": cut,
            "elapsed": round(time.monotonic() - t0, 1)}


def compose_all(input_path, recipe, out_dir, basename, work_dir=None, **kw):
    """Render the recipe for every export platform. Platforms sharing an aspect
    reuse one render (copied) since encoding settings are identical.

    basename may contain "{platform}" (filled per platform); otherwise
    "_<platform>" is appended."""
    from campaign.presets import preset
    results, by_aspect = [], {}
    aspect_op = next((o.get("aspect") for o in (recipe or {}).get("ops") or []
                      if o.get("op") == "reframe"), None)
    for plat in (recipe or {}).get("export") or ["generic"]:
        p = preset(plat)
        aspect = aspect_op or p["aspect"]
        stem = (basename.replace("{platform}", p["platform"]) if "{platform}" in basename
                else f"{basename}_{p['platform']}")
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_.") + ".mp4"
        path = os.path.join(out_dir, name)
        if aspect in by_aspect and by_aspect[aspect]["duration"]:
            shutil.copy2(by_aspect[aspect]["output_path"], path)
            r = dict(by_aspect[aspect], output_path=path, platform=p["platform"], reused=True)
        else:
            r = compose(input_path, recipe, path, platform=p["platform"],
                        work_dir=os.path.join(work_dir, p["platform"]) if work_dir else None, **kw)
            by_aspect[aspect] = r
        results.append(r)
    return results
