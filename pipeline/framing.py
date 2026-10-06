"""pipeline/framing.py

Face-aware vertical (9:16) framing for viral clips.

Replaces the old default (landscape video over a blurred background) with
configurable layout modes:

  auto           — decide per clip (default): 1 face -> speaker_crop,
                   2 faces wide shot -> stacked_split, uncertain -> branded_fit.
                   Camera cuts to a single close-up switch to speaker_crop.
  speaker_crop   — full-screen 9:16 crop following one tracked person.
  stacked_split  — top/bottom stacked crops, one person per panel.
  branded_fit    — full frame fitted on a solid branded background (no blur).

Techniques adapted from the MIT-licensed reference project
NaufalRizqullah/opensource-clipping (Copyright (c) 2026 Muhammad Naufal
Rizqullah) — see ATTRIBUTION below — reimplemented here against this repo's
FFmpeg config, checkpointing and subtitle pipeline:
  - clipping/studio/face_detection.py  -> MediaPipe FaceDetector singleton +
    auto model download, visual speaker-count scan.
  - clipping/studio/render_hybrid.py    -> STEP 0.25s / DEADZONE 0.15 /
    SMOOTH 0.30 / JITTER 5px / SNAP 0.25 smoothing core.
  - clipping/studio/render_split_screen.py -> top/bottom panels, diarization-
    guided face assignment, per-speaker frozen-frame fallback, scene-cut reset.
  - clipping/studio/render_camera_switch.py -> hold-duration switching,
    blurred-pillarbox avoidance (we use branded_fit instead of blur).
  - clipping/diarization.py             -> get_active_speaker(s) helpers;
    audio diarization ALONE never maps to a face (see map_speakers_to_faces).

v1 scope (per task): reliable face crops + stacked layouts. Active-speaker
switching is OPTIONAL and OFF by default (allow_active_speaker_switch=false):
  - fixed-seat podcasts may set framing.speaker_map {SPEAKER_00: left, ...}
    plus optional diarization segments; mapping must be explicit/reliable.
  - if mapping is uncertain -> stacked_split or branded_fit.
  - never present a frozen listener frame as live reaction footage: frozen
    panels are dimmed and labelled via layout_reason, and camera-switch
    full-screen follows only reliably tracked faces.

ATTRIBUTION: smoothing constants, deadzone/snap approach, panel divider and
scene-cut reset concept adapted from opensource-clipping (MIT). All code below
is a fresh implementation for this repository.
"""
import math
import os
import time
import urllib.request

from logger import get_logger

log = get_logger("pipeline.framing")

SUPPORTED_LAYOUTS = ("auto", "speaker_crop", "stacked_split", "branded_fit")

OUT_W, OUT_H = 1080, 1920

MEDIAPIPE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)

_detector_singleton = None
_haar_singleton = None


def validate_layout(value, fallback="auto"):
    """Return a supported layout string; fall back with a warning otherwise."""
    v = str(value or fallback).strip().lower()
    if v not in SUPPORTED_LAYOUTS:
        log.warning("unsupported layout '%s', falling back to '%s'", value, fallback)
        return fallback
    return v


# ── Face detection ────────────────────────────────────────────────────────────

def _model_path():
    return os.path.join("models", "blaze_face_short_range.tflite")


def get_detector(min_confidence=0.5):
    """MediaPipe FaceDetector singleton, or None when unavailable.

    Model download behaviour (documented for reproducibility): on first use,
    downloads blaze_face_short_range.tflite (~230KB) from
    storage.googleapis.com into models/. Set framing.face_model to a local
    path to skip the download. Requires `pip install mediapipe`. When
    mediapipe is missing/unusable, callers fall back to Haar then branded_fit.
    """
    global _detector_singleton
    if _detector_singleton is not None:
        return _detector_singleton
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except Exception:
        log.info("mediapipe not installed; face detection degraded (haar/branded_fit)")
        return None
    try:
        from config import get_config
        cfg = get_config().get("framing", {})
        model_path = cfg.get("face_model") or _model_path()
        url = cfg.get("face_model_url") or MEDIAPIPE_MODEL_URL
        conf = float(cfg.get("face_confidence", min_confidence))
    except Exception:
        model_path, url, conf = _model_path(), MEDIAPIPE_MODEL_URL, min_confidence
    try:
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
            log.info("downloading face model %s -> %s", url, model_path)
            urllib.request.urlretrieve(url, model_path)
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        base = mp_python.BaseOptions(model_asset_path=model_path)
        _detector_singleton = mp_vision.FaceDetector.create_from_options(
            mp_vision.FaceDetectorOptions(base_options=base,
                                           min_detection_confidence=conf))
        log.info("mediapipe face detector ready model=%s", model_path)
    except Exception:
        log.exception("mediapipe detector init failed; using fallback")
        _detector_singleton = None
    return _detector_singleton


def _haar():
    global _haar_singleton
    if _haar_singleton is not None:
        return _haar_singleton
    try:
        import cv2
        path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if os.path.exists(path):
            _haar_singleton = cv2.CascadeClassifier(path)
        else:
            _haar_singleton = None
    except Exception:
        _haar_singleton = None
    return _haar_singleton


def detect_faces_bgr(frame_bgr):
    """Return [{x1,y1,x2,y2,cx,cy}] in source pixels. Never raises."""
    try:
        det = get_detector()
        if det is not None:
            import cv2
            import mediapipe as _mp  # Image/ImageFormat live here in mp>=0.10
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb)
            res = det.detect(img)
            h, w = frame_bgr.shape[:2]
            out = []
            for d in res.detections or []:
                bb = d.bounding_box
                x1 = max(0.0, float(bb.origin_x)); y1 = max(0.0, float(bb.origin_y))
                x2 = min(float(w), x1 + float(bb.width)); y2 = min(float(h), float(bb.height))
                if x2 <= x1 or y2 <= y1:
                    continue
                out.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2})
            return out
    except Exception:
        log.debug("mediapipe detect failed, trying haar", exc_info=True)
    try:
        haar = _haar()
        if haar is not None:
            import cv2
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            rects = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                          minSize=(60, 60))
            return [{"x1": float(x), "y1": float(y),
                     "x2": float(x + w), "y2": float(y + h),
                     "cx": float(x + w / 2), "cy": float(y + h / 2)}
                    for (x, y, w, h) in rects]
    except Exception:
        pass
    return []


# ── Sampling / scene cuts / tracking ─────────────────────────────────────────

def _frame_diff(a_gray_small, b_gray_small):
    import numpy as np
    return float(abs(a_gray_small.astype(float) - b_gray_small.astype(float)).mean())


def sample_clip(video_path, cfg=None):
    """Sample faces every sample_interval seconds.

    Returns dict {samples, scenes, width, height, fps, duration} where each
    sample is {t, boxes, scene_cut}. Tracking is reset at scene cuts.
    Stable association (not largest-face-per-frame): greedy nearest-centre
    matching carries track ids across samples within a shot.
    """
    import cv2
    cfg = cfg or {}
    interval = float(cfg.get("sample_interval", 0.25))
    cut_thr = float(cfg.get("scene_cut_threshold", 18))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if math.isnan(fps) or fps <= 0:
        fps = 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = (n / fps) if n and fps else 0.0

    samples, scenes = [], [0.0]
    prev_small, prev_tracks = None, []
    next_id, t = 0, 0.0
    hold = int(cfg.get("layout_stability_frames", 6))
    _ = hold
    while True:
        if duration and t > duration:
            break
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64))
        scene_cut = False
        if prev_small is not None and _frame_diff(prev_small, small) > cut_thr:
            scene_cut = True
            scenes.append(round(t, 3))
            prev_tracks = []  # tracking reset at camera cuts
        prev_small = small
        boxes = detect_faces_bgr(frame)
        # stable association within shot
        tracks = []
        used = set()
        for pt in prev_tracks:
            best, best_d = None, 1e9
            for j, b in enumerate(boxes):
                if j in used:
                    continue
                d = abs(b["cx"] - pt["cx"]) + abs(b["cy"] - pt["cy"])
                if d < best_d:
                    best, best_d = j, d
            # match gate: within 12% of frame width (same person, not a jump)
            if best is not None and best_d < w * 0.12:
                used.add(best)
                tracks.append({"id": pt["id"], "cx": boxes[best]["cx"],
                               "cy": boxes[best]["cy"], "box": boxes[best]})
        for j, b in enumerate(boxes):
            if j not in used:
                tracks.append({"id": next_id, "cx": b["cx"], "cy": b["cy"], "box": b})
                next_id += 1
        prev_tracks = tracks
        samples.append({"t": round(t, 3), "boxes": boxes, "tracks": tracks,
                        "scene_cut": scene_cut, "n": len(boxes)})
        if duration and t + 1e-6 >= duration:
            break
        t += interval
        if t > 600:  # safety cap (10 min clips never happen; guards broken fps)
            break
    cap.release()
    return {"samples": samples, "scenes": scenes, "width": w, "height": h,
            "fps": fps, "duration": duration or (samples[-1]["t"] if samples else 0.0)}


def decide_layout(analysis, cfg=None, override="auto"):
    """Stable layout decision from sampled tracks.

    - 1 reliably detected face -> speaker_crop
    - 2 reliably detected faces in a wide shot -> stacked_split
    - camera cut to single close-up -> speaker_crop (handled per-segment)
    - missing/uncertain -> branded_fit (never blurred background)
    Stability: a briefly missed face must not flap layouts — require
    layout_stability_frames consecutive samples before switching.
    """
    cfg = cfg or {}
    override = validate_layout(override)
    if override != "auto":
        return {"layout": override, "reason": f"manual override {override}",
                "stable": True}
    samples = analysis.get("samples", [])
    if not samples:
        return {"layout": "branded_fit", "reason": "no samples (unreadable clip)", "stable": True}
    stab = max(2, int(cfg.get("layout_stability_frames", 6)))
    min_face_frac = float(cfg.get("min_face_fraction", 0.5))
    if not min_face_frac:
        min_face_frac = 0.5
    votes = {"speaker_crop": 0, "stacked_split": 0, "branded_fit": 0}
    # per-sample vote with hysteresis window
    window = []
    for s in samples:
        n = s.get("n", 0)
        if n == 1:
            v = "speaker_crop"
        elif n >= 2:
            v = "stacked_split"
        else:
            v = "branded_fit"
        window.append(v)
        if len(window) > stab:
            window.pop(0)
        # majority of the stability window
        maj = max(set(window), key=window.count)
        votes[maj] += 1
    total = sum(votes.values()) or 1
    # reliability: faces must appear in a good fraction of samples
    with_faces = sum(1 for s in samples if s.get("n", 0) > 0)
    if with_faces / max(1, len(samples)) < min_face_frac:
        return {"layout": "branded_fit",
                "reason": f"faces in only {with_faces}/{len(samples)} samples (< {min_face_frac:.0%}); branded fallback",
                "stable": True, "votes": votes}
    layout = max(votes, key=votes.get)
    conf = votes[layout] / total
    # camera-cut awareness: if scenes exist with different face counts, auto
    # rendering splits per scene (see render); the clip-level choice stays stable.
    n_scenes = len(analysis.get("scenes", [0.0]))
    reason = (f"stable majority {layout} ({votes[layout]}/{total} windows, "
              f"conf={conf:.2f}, scenes={n_scenes})")
    return {"layout": layout, "reason": reason, "stable": True, "votes": votes}


# ── Crop geometry ─────────────────────────────────────────────────────────────

def _crop_for_cx(w, h, cx, cy, cfg):
    """9:16 crop window (x, y, cw, ch) around (cx, cy) with headroom + limits."""
    target_ratio = OUT_W / OUT_H
    ch = h
    cw = int(h * target_ratio)
    if cw > w:  # very narrow source (rare): fit width instead
        cw, ch = w, int(w / target_ratio)
    max_zoom = float(cfg.get("max_zoom", 2.5)); min_zoom = float(cfg.get("min_zoom", 1.0))
    zoom = min(max(min_zoom, 1.0), max_zoom)
    if zoom > 1.0:
        cw = max(int(cw / zoom), int(OUT_W / 4)); ch = max(int(ch / zoom), int(OUT_H / 4))
    headroom = float(cfg.get("headroom_ratio", 0.35))
    # place face at headroom from top of crop
    y = int(min(max(cy - ch * headroom, 0), max(0, h - ch)))
    x = int(min(max(cx - cw / 2, 0), max(0, w - cw)))
    return x, y, cw, ch


def smooth_centres(analysis, cfg=None):
    """Deadzone + exponential smoothing + jitter gate + snap (no interp across cuts)."""
    cfg = cfg or {}
    samples = analysis.get("samples", [])
    w = analysis.get("width", 1280)
    dead = float(cfg.get("deadzone_ratio", 0.15))
    smooth = float(cfg.get("smooth_factor", 0.30))
    jitter = float(cfg.get("jitter_px", 5))
    snap = float(cfg.get("snap_ratio", 0.25))
    hold_s = float(cfg.get("hold_last_position_seconds", 1.0))
    interval = float(cfg.get("sample_interval", 0.25))
    hold_n = max(1, int(hold_s / max(interval, 1e-3)))

    track_w = int((analysis.get("height", 720)) * (OUT_W / OUT_H))
    dead_px, snap_px = track_w * dead, w * snap
    out, cam_cx, cam_cy, missing = [], None, None, 0
    for s in samples:
        if s.get("scene_cut"):
            missing = 0  # reset; next face re-anchors (snap below handles jump)
        if s.get("tracks"):
            # follow lowest-id stable track (not largest face)
            tr = sorted(s["tracks"], key=lambda t: t["id"])[0]
            fx, fy = tr["cx"], tr["cy"]
            if cam_cx is None:
                cam_cx, cam_cy = fx, fy
            elif abs(fx - cam_cx) > snap_px:
                cam_cx, cam_cy = fx, fy  # hard cut between speakers
            else:
                if abs(fx - cam_cx) > dead_px + jitter:
                    cam_cx += (fx - cam_cx) * smooth
                # vertical follows more loosely (subtitle-safe)
                if abs(fy - cam_cy) > jitter:
                    cam_cy += (fy - cam_cy) * smooth
            missing = 0
        else:
            missing += 1  # brief retention of last position, then fallback
        out.append({"t": s["t"], "cx": cam_cx, "cy": cam_cy,
                    "missing": missing, "lost": missing > hold_n,
                    "scene_cut": s.get("scene_cut", False)})
    return out


def _interp_centres(smoothed, t):
    if not smoothed:
        return None
    if t <= smoothed[0]["t"]:
        return smoothed[0]
    if t >= smoothed[-1]["t"]:
        return smoothed[-1]
    for a, b in zip(smoothed, smoothed[1:]):
        if a["t"] <= t <= b["t"]:
            if b.get("scene_cut"):
                return b  # never interpolate across a cut
            f = (t - a["t"]) / max(b["t"] - a["t"], 1e-6)
            if a["cx"] is None:
                return b
            if b["cx"] is None:
                return a
            return {"t": t, "cx": a["cx"] + (b["cx"] - a["cx"]) * f,
                    "cy": a["cy"] + (b["cy"] - a["cy"]) * f,
                    "missing": b["missing"], "lost": b["lost"],
                    "scene_cut": False}
    return smoothed[-1]


# ── Diarization (optional) ────────────────────────────────────────────────────

def get_active_speakers(diarization, t):
    """All speakers active at time t (adapted from reference diarization.py)."""
    if not diarization:
        return []
    return [s["speaker"] for s in diarization
            if s.get("start", 0) <= t <= s.get("end", 0)]


def map_speakers_to_faces(diarization, analysis, speaker_map=None):
    """Map diarized speakers to face tracks ONLY with explicit/reliable mapping.

    speaker_map: {SPEAKER_00: left|right} for fixed-seat podcasts.
    Returns {speaker: track_id} or {} when uncertain. Audio diarization alone
    NEVER identifies a face — without speaker_map (or a single-face track),
    this returns {} and the caller uses stacked/branded layouts.
    """
    if not diarization:
        return {}
    speaker_map = speaker_map or {}
    if not speaker_map:
        return {}  # uncertain -> stacked or branded_fit
    # left/right positional mapping for fixed seats
    tracks = {}
    for s in analysis.get("samples", []):
        for tr in s.get("tracks", []):
            tracks.setdefault(tr["id"], []).append(tr["cx"])
    if not tracks:
        return {}
    w = analysis.get("width", 1280)
    ids_by_x = sorted(tracks, key=lambda i: sum(tracks[i]) / len(tracks[i]))
    if len(ids_by_x) < 2:
        return {}
    side_id = {"left": ids_by_x[0], "right": ids_by_x[-1]}
    out = {}
    for spk, side in speaker_map.items():
        if side in side_id:
            out[spk] = side_id[side]
    return out


# ── Rendering ─────────────────────────────────────────────────────────────────

def _hex_to_bgr(hex_color):
    h = str(hex_color or "#0B0F1A").lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    try:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except ValueError:
        r, g, b = 11, 15, 26
    return (b, g, r)


def _ffmpeg_writer(output_path, fps):
    from config import ffmpeg_path
    import subprocess
    cmd = [ffmpeg_path(), "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{OUT_W}x{OUT_H}", "-r", f"{fps:.2f}", "-i", "-",
           "-i", "placeholder_audio",
           "-c:v", "libx264", "-preset", "fast", "-crf", "23",
           "-pix_fmt", "yuv420p", output_path]
    # audio is muxed in a second pass (preserves original audio exactly);
    # here we render video-only then mux. Build video-only command:
    cmd = [ffmpeg_path(), "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{OUT_W}x{OUT_H}", "-r", f"{fps:.2f}", "-i", "-",
           "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "23",
           "-pix_fmt", "yuv420p", output_path + ".video.mp4"]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc


def render_vertical(input_path, output_path, layout="auto", cfg=None,
                    analysis=None, diarization=None):
    """Render input_path (a cut clip) to 1080x1920 output_path.

    Returns {output_path, layout, layout_reason, width, height}.
    Audio is preserved from the input (stream copy via ffmpeg mux pass).
    """
    import cv2
    import subprocess
    from config import ffmpeg_path
    cfg = cfg or {}
    layout = validate_layout(layout)
    t0 = time.monotonic()
    if analysis is None:
        analysis = sample_clip(input_path, cfg)
    decision = decide_layout(analysis, cfg, layout)
    final_layout = decision["layout"]
    w, h = analysis["width"], analysis["height"]
    fps = analysis.get("fps", 25.0) or 25.0
    smoothed = smooth_centres(analysis, cfg)
    bg = _hex_to_bgr(cfg.get("branded_background", "#0B0F1A"))
    divider = int(cfg.get("split_divider_px", 6))

    cap = cv2.VideoCapture(input_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    tmp_video = output_path + ".video.mp4"
    proc = _ffmpeg_writer(output_path, fps)
    import numpy as np

    # stacked assignment: left/right stable tracks for the whole clip
    stack_ids = []
    if final_layout == "stacked_split":
        seen = {}
        for s in analysis.get("samples", []):
            for tr in s.get("tracks", []):
                seen.setdefault(tr["id"], []).append((tr["cx"], tr["cy"]))
        # two most persistent tracks
        ranked = sorted(seen, key=lambda i: len(seen[i]), reverse=True)[:2]
        # order left->right by mean cx
        ranked = sorted(ranked, key=lambda i: sum(c[0] for c in seen[i]) / len(seen[i]))
        stack_ids = ranked

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = i / fps
        i += 1
        pos = _interp_centres(smoothed, t)
        lost = (pos is None) or pos.get("lost", True) or pos.get("cx") is None

        if final_layout == "branded_fit" or lost:
            canvas = np.full((OUT_H, OUT_W, 3), bg, dtype=np.uint8)
            scale = min(OUT_W / w, OUT_H / h)
            nw, nh = max(2, int(w * scale) // 2 * 2), max(2, int(h * scale) // 2 * 2)
            small = cv2.resize(frame, (nw, nh))
            y0, x0 = (OUT_H - nh) // 2, (OUT_W - nw) // 2
            canvas[y0:y0 + nh, x0:x0 + nw] = small
            out = canvas
        elif final_layout == "stacked_split" and len(stack_ids) >= 2:
            # one crop per person, stacked top/bottom
            panel_h = (OUT_H - divider) // 2
            panels = []
            for tid in stack_ids[:2]:
                # mean position of this track near time t
                cx = pos["cx"] if tid == stack_ids[0] else None
                # find this track's latest known pos
                best = None
                for s in reversed(analysis.get("samples", [])):
                    if s["t"] > t:
                        continue
                    for tr in s.get("tracks", []):
                        if tr["id"] == tid:
                            best = tr
                            break
                    if best:
                        break
                if best is None:
                    best = {"cx": w * (0.3 if tid == stack_ids[0] else 0.7),
                            "cy": h * 0.4}
                x, y, cw, ch = _crop_for_cx(w, h, best["cx"], best["cy"], cfg)
                crop = frame[y:y + ch, x:x + cw]
                panels.append(cv2.resize(crop, (OUT_W, panel_h)))
            out = np.vstack([panels[0],
                             np.full((divider, OUT_W, 3), 20, dtype=np.uint8),
                             panels[1]])
            if out.shape[0] != OUT_H:  # odd divider guard
                out = cv2.resize(out, (OUT_W, OUT_H))
        else:  # speaker_crop
            cx = pos["cx"] if pos and pos.get("cx") is not None else w / 2
            cy = pos["cy"] if pos and pos.get("cy") is not None else h * 0.4
            x, y, cw, ch = _crop_for_cx(w, h, cx, cy, cfg)
            crop = frame[y:y + ch, x:x + cw]
            out = cv2.resize(crop, (OUT_W, OUT_H))
        try:
            proc.stdin.write(out.tobytes())
        except BrokenPipeError:
            break
    cap.release()
    try:
        proc.stdin.close()
        proc.wait(timeout=120)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    # mux original audio (speech fade-out disabled by default; see improvements)
    try:
        r = subprocess.run(
            [ffmpeg_path(), "-y", "-i", tmp_video, "-i", input_path,
             "-map", "0:v", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac",
             "-shortest", output_path],
            capture_output=True, text=True)
        if r.returncode != 0:
            log.error("framing mux failed rc=%d err=%.300s", r.returncode, (r.stderr or "")[-300:])
            # fall back to video-only file
            import shutil
            shutil.move(tmp_video, output_path)
        elif os.path.exists(tmp_video):
            os.remove(tmp_video)
    except Exception:
        log.exception("framing mux exception")
    dur = time.monotonic() - t0
    log.info("framing done in=%.60s out=%.60s layout=%s frames=%d elapsed=%.1fs",
             input_path, output_path, final_layout, i, dur)
    return {"output_path": output_path, "layout": final_layout,
            "layout_reason": decision.get("reason", ""), "width": OUT_W,
            "height": OUT_H, "frames": i, "analysis": analysis}
