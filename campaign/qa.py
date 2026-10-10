"""campaign/qa.py — compliance checks on a rendered deliverable (CAMPAIGN_PIPELINE_PLAN C6).

check_file(path, recipe, platform, report) measures the FILE (not the
compositor's intentions) and returns
    {"status": "pass"|"warn"|"fail", "checks": [{id, label, status, detail}]}

  duration    within the platform preset limit and the recipe's trim / clip spec
  format      expected resolution + aspect, H.264 video, AAC audio
  size        under the platform's file-size limit
  audio       an audio stream exists; integrated loudness near the preset (ebur128)
  replaced    audio.replace without the original kept → the output sounds like
              the replacement track (spectrum / envelope match)
  logo        the logo's pixels are present where the recipe puts it
  text        captions / text overlays the recipe asks for were drawn
  skipped     ops not rendered yet (C8) are listed
"""
import os
import re
import subprocess

import numpy as np

from logger import get_logger
from pipeline import media

log = get_logger("campaign.qa")

DUR_TOL = 0.35          # s, file vs planned duration
LIMIT_TOL = 2.0         # s, slack on clip-spec min/max (sentence-snapped ends)
LUFS_TOL = 2.0          # LU around the preset target


def _ffmpeg():
    from config import ffmpeg_path
    return ffmpeg_path()


def _ops(recipe):
    return {o.get("op"): o for o in (recipe or {}).get("ops") or []}


def _check(cid, label, status, detail=""):
    return {"id": cid, "label": label, "status": status, "detail": detail}


# ── measurements ─────────────────────────────────────────────────────────────

def loudness(path):
    """Integrated loudness (LUFS) via ebur128, or None."""
    r = subprocess.run([_ffmpeg(), "-hide_banner", "-nostats", "-i", path, "-vn",
                        "-af", "ebur128=framelog=quiet", "-f", "null", "-"],
                       capture_output=True, text=True)
    m = re.findall(r"I:\s+(-?[\d.]+|-inf)\s+LUFS", r.stderr)
    if not m or m[-1] == "-inf":
        return None
    return float(m[-1])


def _pcm(path, start=0.0, dur=None, sr=8000, loop=False):
    cmd = [_ffmpeg(), "-loglevel", "error"]
    if loop:
        cmd += ["-stream_loop", "-1"]
    cmd += ["-ss", f"{max(0.0, start):.3f}", "-i", path]
    if dur:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    r = subprocess.run(cmd, capture_output=True)
    return np.frombuffer(r.stdout, dtype=np.float32)


def _bands(x, sr=8000, n=48):
    """Normalised average spectrum in log-spaced bands (60 Hz – 4 kHz)."""
    if len(x) < sr // 4:
        return None
    frames = len(x) // 1024
    spec = np.abs(np.fft.rfft(x[:frames * 1024].reshape(frames, 1024) * np.hanning(1024),
                              axis=1)).mean(axis=0)
    f = np.fft.rfftfreq(1024, 1 / sr)
    edges = np.geomspace(60, sr / 2, n + 1)
    b = np.array([spec[(f >= lo) & (f < hi)].sum() for lo, hi in zip(edges[:-1], edges[1:])])
    b = np.log1p(b * 100)
    nrm = np.linalg.norm(b)
    return b / nrm if nrm else None


def _envelope(x, sr=8000, hop=0.05):
    n = int(sr * hop)
    k = len(x) // n
    return np.sqrt((x[:k * n].reshape(k, n) ** 2).mean(axis=1) + 1e-12) if k else np.array([])


def audio_match(out_path, track_path, track_start=0.0, loop=True, original=None,
                orig_start=0.0, seconds=None, out_offset=0.0):
    """How much the output sounds like the replacement track (and not the original).
    Returns {score, vs_original, envelope_corr, verdict}."""
    dur = seconds or media.probe(out_path).get("duration") or 0
    dur = min(dur, 60.0)
    lead = min(0.5, dur * 0.1)        # skip fade-in
    out = _pcm(out_path, out_offset + lead, dur - 2 * lead)
    trk = _pcm(track_path, track_start + lead, dur - 2 * lead, loop=loop)
    bo, bt = _bands(out), _bands(trk)
    if bo is None or bt is None:
        return {"verdict": "unknown"}
    res = {"score": round(float(bo @ bt), 3)}
    eo, et = _envelope(out), _envelope(trk)
    k = min(len(eo), len(et))
    if k > 10 and eo[:k].std() > 1e-3 and et[:k].std() > 1e-3:
        res["envelope_corr"] = round(float(np.corrcoef(np.log(eo[:k]), np.log(et[:k]))[0, 1]), 3)
    if original:
        borig = _bands(_pcm(original, orig_start + lead, dur - 2 * lead))
        if borig is not None:
            res["vs_original"] = round(float(bo @ borig), 3)
    corr = res.get("envelope_corr")
    better = res["score"] > res.get("vs_original", -1) + 0.02
    if (corr is not None and corr > 0.5) or (res["score"] > 0.9 and better):
        res["verdict"] = "match"
    elif corr is not None and corr < 0.2 and not better:
        res["verdict"] = "mismatch"
    else:
        res["verdict"] = "unclear"
    return res


def logo_geometry(W, H, lo, logo_w, logo_h):
    """(x, y, w, h) of a logo op on a W×H frame — same rules as the compositor."""
    lw = max(16, int(round(W * float(lo.get("size_pct") or 12) / 100.0)))
    lh = max(1, int(round(logo_h * lw / max(1, logo_w))))
    m = int(round(W * float(lo.get("margin_pct") if lo.get("margin_pct") is not None else 3) / 100.0))
    pos = lo.get("position") or "top-right"
    x = m if "left" in pos else W - lw - m if "right" in pos else (W - lw) // 2
    y = m if pos.startswith("top") else H - lh - m if pos.startswith("bottom") else (H - lh) // 2
    return x, y, lw, lh


def _frame(path, t, W, H):
    r = subprocess.run([_ffmpeg(), "-loglevel", "error", "-ss", f"{t:.3f}", "-i", path,
                        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       capture_output=True)
    if len(r.stdout) != W * H * 3:
        return None
    return np.frombuffer(r.stdout, dtype=np.uint8).reshape(H, W, 3).astype(np.float32)


def logo_present(out_path, lo, W, H, duration, offset=0.0):
    """Mean colour error of the logo's opaque pixels at its expected spot (best of
    a ±6 px search, 3 frames). Returns {error, limit, verdict}."""
    import cv2
    from pipeline.compositor import _window
    img = cv2.imread(lo["asset_path"], cv2.IMREAD_UNCHANGED)
    if img is None:
        return {"verdict": "unknown", "detail": "logo file unreadable"}
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    x, y, w, h = logo_geometry(W, H, lo, img.shape[1], img.shape[0])
    logo = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    rgb = logo[:, :, 2::-1]
    alpha = logo[:, :, 3] / 255.0 * float(lo.get("opacity") or 1.0)
    mask = cv2.erode((alpha > 0.45).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if mask.sum() < 20:
        return {"verdict": "unknown", "detail": "logo has too few opaque pixels to check"}
    a, b = _window(lo, duration)
    a, b = a + offset, b + offset          # body starts after an intro
    eff = float(alpha[mask].mean())         # logo alpha × op opacity on the checked pixels
    limit = 255 * (1 - eff) * 0.8 + 32
    best = None
    for t in np.linspace(a + 0.2 * (b - a), a + 0.8 * (b - a), 3):
        fr = _frame(out_path, float(t), W, H)
        if fr is None:
            continue
        for dy in range(-6, 7, 2):
            for dx in range(-6, 7, 2):
                yy, xx = y + dy, x + dx
                if yy < 0 or xx < 0 or yy + h > H or xx + w > W:
                    continue
                region = fr[yy:yy + h, xx:xx + w]
                err = float(np.abs(region - rgb)[mask].mean())
                best = err if best is None else min(best, err)
    if best is None:
        return {"verdict": "unknown", "detail": "could not read frames"}
    return {"error": round(best, 1), "limit": round(limit, 1),
            "verdict": "match" if best <= limit else "mismatch"}


# ── the report ───────────────────────────────────────────────────────────────

def check_file(path, recipe, platform="generic", report=None, source_path=None,
               clip_spec=None) -> dict:
    """QA one deliverable. report = the compose result (applied/skipped/
    expected_duration/source_start); source_path = the input it came from."""
    from campaign.presets import output_size, preset
    report = report or {}
    ops = _ops(recipe)
    pre = preset(platform)
    checks = []
    p = media.probe(path)
    if not p["exists"] or not p["has_video"]:
        return {"status": "fail", "checks": [_check("file", "File", "fail",
                                                     "missing or has no video")]}

    # duration
    dur = p["duration"] or 0
    # with an intro / outro / end card the recipe's trim + clip limits apply to the body
    body_off = float(report.get("body_offset") or 0.0)
    body_dur = float(report.get("body_duration") or dur)
    exp = report.get("expected_duration")
    st, why = "pass", f"{dur:.1f}s"
    if exp and abs(dur - exp) > DUR_TOL:
        st, why = "fail", f"{dur:.2f}s but {exp:.2f}s was planned"
    lim = pre.get("max_duration")
    if lim and dur > lim + 0.05:
        st, why = "fail", f"{dur:.1f}s is over the {pre['label']} limit of {lim}s"
    trim = ops.get("trim") or {}
    spec = clip_spec or {}
    hi = trim.get("max_duration") or spec.get("max_duration")
    lo = trim.get("min_duration") or spec.get("min_duration")
    if st == "pass" and hi and body_dur > hi + (LIMIT_TOL if spec.get("max_duration") else 0.05):
        st, why = "fail", f"{body_dur:.1f}s is over the campaign maximum of {hi:g}s"
    if st == "pass" and lo and body_dur < lo - LIMIT_TOL:
        st, why = "warn", f"{body_dur:.1f}s is under the campaign minimum of {lo:g}s"
    if body_off or abs(body_dur - dur) > DUR_TOL:
        why += f" ({body_dur:.1f}s video + intro/outro)"
    checks.append(_check("duration", "Duration", st, why))

    # format
    rf = ops.get("reframe") or {}
    W, H = output_size(rf.get("aspect") or pre["aspect"])
    fmt_ok = (p["width"], p["height"]) == (W, H)
    codec_ok = p.get("video_codec") == "h264" and (not p["has_audio"] or p.get("audio_codec") == "aac")
    checks.append(_check("format", "Resolution / codecs", "pass" if fmt_ok and codec_ok else "fail",
                         f"{p['width']}x{p['height']} {p.get('video_codec')}/{p.get('audio_codec')}"
                         + ("" if fmt_ok else f" (expected {W}x{H})")))

    # size
    mb = (p.get("size_bytes") or 0) / 1e6
    cap = pre.get("max_size_mb")
    checks.append(_check("size", "File size", "fail" if cap and mb > cap else "pass",
                         f"{mb:.1f} MB" + (f" (limit {cap} MB)" if cap else "")))

    # audio + loudness
    if not p["has_audio"]:
        checks.append(_check("audio", "Audio", "fail", "no audio stream"))
    else:
        lufs = loudness(path)
        target = pre.get("loudness_lufs", -14.0)
        if lufs is None:
            checks.append(_check("audio", "Audio / loudness", "warn", "silent"
                                 if not ops.get("audio.mute_original") else "muted as requested"))
        else:
            off = abs(lufs - target)
            checks.append(_check("audio", "Audio / loudness",
                                 "pass" if off <= LUFS_TOL else "warn" if off <= 4 else "fail",
                                 f"{lufs:.1f} LUFS (target {target:g})"))

    # replacement audio
    rep = ops.get("audio.replace")
    if rep and rep.get("track_path") and os.path.isfile(rep["track_path"]) and p["has_audio"]:
        if rep.get("keep_original"):
            checks.append(_check("replaced", "Audio replaced", "pass",
                                 f"mixed with the original at {rep['keep_original']:g}"))
        else:
            m = audio_match(path, rep["track_path"], float(rep.get("start") or 0),
                            loop=rep.get("loop", True),
                            original=None if ops.get("speed") else source_path,
                            orig_start=float(report.get("source_start") or 0),
                            seconds=body_dur, out_offset=body_off)
            st = {"match": "pass", "mismatch": "fail"}.get(m["verdict"], "warn")
            checks.append(_check("replaced", "Audio replaced", st,
                                 ", ".join(f"{k} {v}" for k, v in m.items() if k != "verdict")
                                 or m["verdict"]))

    # logos
    logos = [o for o in (recipe or {}).get("ops") or [] if o.get("op") == "logo"]
    for i, o in enumerate(logos):
        if not o.get("asset_path") or not os.path.isfile(o["asset_path"]):
            checks.append(_check(f"logo{i}", "Logo", "fail", "logo file missing"))
            continue
        lp = logo_present(path, o, p["width"], p["height"], body_dur, body_off)
        st = {"match": "pass", "mismatch": "fail"}.get(lp["verdict"], "warn")
        checks.append(_check(f"logo{i}", f"Logo {o.get('asset')}", st,
                             lp.get("detail") or f"pixel error {lp.get('error')} "
                                                 f"(limit {lp.get('limit')}) at {o.get('position')}"))

    # captions / text
    applied = set(report.get("applied") or [])
    cap = ops.get("captions")
    if cap and cap.get("enabled", True):
        checks.append(_check("captions", "Captions", "pass" if "captions" in applied else "warn",
                             "burned in" if "captions" in applied else
                             "; ".join(w for w in report.get("warnings") or []
                                       if "caption" in w) or "not drawn"))
    if any(o.get("op") == "text_overlay" for o in (recipe or {}).get("ops") or []):
        checks.append(_check("text", "Text overlays",
                             "pass" if "text_overlay" in applied else "fail",
                             "drawn" if "text_overlay" in applied else "not drawn"))
    for k in ("intro", "outro", "end_card"):
        if ops.get(k):
            ok = k in applied
            checks.append(_check(k, k.replace("_", " ").capitalize(), "pass" if ok else "fail",
                                 "added" if ok else
                                 "; ".join(w for w in report.get("warnings") or [] if k in w)
                                 or "missing"))
    if report.get("skipped"):
        checks.append(_check("skipped", "Not rendered yet", "warn",
                             ", ".join(report["skipped"])))

    worst = ("fail" if any(c["status"] == "fail" for c in checks) else
             "warn" if any(c["status"] == "warn" for c in checks) else "pass")
    return {"status": worst, "checks": checks}


def check_deliverable(row, recipe, source_path=None) -> dict:
    """QA a deliverables row (its qa holds the compose report); returns the merged qa."""
    qa = dict(row.get("qa") or {})
    spec = None
    if (qa.get("clip") or {}) and (recipe or {}).get("clips"):
        spec = recipe["clips"]
    try:
        res = check_file(row["path"], recipe, row.get("platform") or "generic", qa,
                         source_path=source_path, clip_spec=spec)
    except Exception as e:
        log.exception("QA failed for %s", row.get("path"))
        res = {"status": "warn", "checks": [_check("qa", "QA", "warn", f"QA crashed: {e}")]}
    qa.update(compliance=res["status"], checks=res["checks"])
    return qa
