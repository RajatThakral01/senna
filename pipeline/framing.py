"""pipeline/framing.py

Full-screen vertical (9:16) framing for viral clips.

HARD REQUIREMENT (framing.full_screen_vertical, default true): every output
shot fills the entire 1080x1920 frame. No letterboxing, padding, blurred
backgrounds, or fit-on-background fallbacks anywhere in the automatic paths.
Aspect ratio is always preserved (crop, never stretch); the accepted tradeoff
is loss of horizontal context.

Layout modes (SUPPORTED_LAYOUTS):

  auto           — decide per clip (default): 1 face -> speaker_crop,
                   2 faces wide shot -> stacked_split, uncertain -> center_crop.
                   Camera cuts to a single close-up switch to speaker_crop.
  speaker_crop   — full-screen 9:16 crop following one tracked person.
  stacked_split  — top/bottom stacked crops, one person per panel. Each panel
                   fills its area (no padding inside panels either).
  center_crop    — stable, static 9:16 centre crop (of the content region).
                   Final automatic fallback when no subject is tracked.
  branded_fit    — MANUAL ONLY, and only honoured when full_screen_vertical
                   is off. Under the default flag it is mapped to center_crop
                   at render time (recorded in layout_reason). Never selected
                   automatically and never rendered as bars when the flag is on.

Crop selection is planned per scene (split at scene cuts):
  - speaking person  -> face/upper-body crop following the stable track,
  - DIY/workbench    -> configured work_area ROI when set, else centre crop,
  - product close-up -> work_area ROI when set, else centre crop,
  - multiple people  -> full-screen stacked crops where appropriate.

Fallback order inside a shot (never back to a full horizontal picture):
  1. manual ROI from the edit plan (authoritative; detection never overrides),
  2. reliable subject track (face), retained briefly across detection loss,
  3. configured work_area crop for DIY footage,
  4. stable centre crop (final fallback).

Heuristic crops (3, 4) are marked needs_review in the decision, the scene
plan, and the edit plan — but they still render full-screen vertical.

Stability: deadzone + exponential smoothing + jitter gate + snap, tracking
resets at scene cuts, no interpolation across cuts, zoom bounded by
min_zoom/max_zoom with a resolution floor (never upscale from a postage
stamp: crop window never smaller than OUT_W/4 x OUT_H/4).

Embedded source bars (letterboxed uploads) are detected
(estimate_content_bounds) or taken from framing.content_bounds, and removed
BEFORE the 9:16 window is calculated, so the crop is computed on real
content, not on black bars.

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
  - if mapping is uncertain -> stacked_split or center_crop.
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

SUPPORTED_LAYOUTS = ("auto", "speaker_crop", "stacked_split", "center_crop",
                     "branded_fit")

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


def fullscreen(cfg=None):
    """True when every shot must fill 9:16 (no bars/pad/fit-fill).

    Hard requirement for this project; on by default. Only an explicit
    ``full_screen_vertical: false`` restores the legacy branded_fit path.
    """
    return bool((cfg or {}).get("full_screen_vertical", True))


# ── Face detection ────────────────────────────────────────────────────────────

def _model_path():
    return os.path.join("models", "blaze_face_short_range.tflite")


def get_detector(min_confidence=0.5):
    """MediaPipe FaceDetector singleton, or None when unavailable.

    Model download behaviour (documented for reproducibility): on first use,
    downloads blaze_face_short_range.tflite (~230KB) from
    storage.googleapis.com into models/. Set framing.face_model to a local
    path to skip the download. Requires `pip install mediapipe`. When
    mediapipe is missing/unusable, callers fall back to Haar then crop-only
    fallbacks (never letterbox).
    """
    global _detector_singleton
    if _detector_singleton is not None:
        return _detector_singleton
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except Exception:
        log.info("mediapipe not installed; face detection degraded (haar/crop fallbacks)")
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


# ── Source content bounds (embedded bars) ─────────────────────────────────────

def estimate_content_bounds(video_path, threshold=8, max_samples=8):
    """Detect the real content rectangle inside possibly-letterboxed source.

    Samples up to max_samples frames; rows/columns whose mean luma stays at
    or below `threshold` on every sample are treated as embedded bars.
    Returns {x, y, w, h} in source pixels (full frame when nothing found).
    Never raises. Set framing.content_bar_threshold: 0 to disable.
    """
    full = None
    try:
        import cv2
        import numpy as np
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        full = {"x": 0, "y": 0, "w": w, "h": h}
        if w <= 0 or h <= 0:
            cap.release()
            return full
        idxs = [int(n * (i + 1) / (max_samples + 1)) for i in range(max_samples)] if n > 0 else [0]
        row_dark = np.ones(h, dtype=bool)
        col_dark = np.ones(w, dtype=bool)
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            row_dark &= gray.mean(axis=1) <= threshold
            # downsample columns for speed on 4K
            small = gray if w <= 1920 else cv2.resize(gray, (1920, h))
            dark_small = small.mean(axis=0) <= threshold
            dark = cv2.resize(dark_small.astype(np.uint8) * 255, (w, 1),
                              interpolation=cv2.INTER_NEAREST)[0] > 0
            col_dark &= dark
        cap.release()
        rows = np.where(~row_dark)[0]
        cols = np.where(~col_dark)[0]
        if len(rows) == 0 or len(cols) == 0:
            return full
        y0, y1 = int(rows[0]), int(rows[-1])
        x0, x1 = int(cols[0]), int(cols[-1])
        # ignore slivers (flash frames / noise): need real content area
        if (x1 - x0) < w * 0.5 or (y1 - y0) < h * 0.5:
            return full
        bounds = {"x": x0, "y": y0, "w": x1 - x0 + 1, "h": y1 - y0 + 1}
        if bounds != full:
            log.info("content bounds x=%d y=%d w=%d h=%d (source %dx%d)",
                     x0, y0, bounds["w"], bounds["h"], w, h)
        return bounds
    except Exception:
        log.debug("content bounds estimation failed, using full frame", exc_info=True)
        return full or {"x": 0, "y": 0, "w": 0, "h": 0}


def resolve_content_bounds(video_path, cfg=None, analysis=None):
    """Content bounds for crop math: cfg override > estimate > full frame."""
    cfg = cfg or {}
    over = cfg.get("content_bounds")
    if isinstance(over, dict) and over.get("w") and over.get("h"):
        try:
            return {"x": int(over.get("x", 0)), "y": int(over.get("y", 0)),
                    "w": int(over["w"]), "h": int(over["h"])}
        except (TypeError, ValueError):
            pass
    if analysis and isinstance(analysis.get("content_bounds"), dict):
        return analysis["content_bounds"]
    thr = cfg.get("content_bar_threshold", 8)
    try:
        thr = int(thr)
    except (TypeError, ValueError):
        thr = 8
    if thr <= 0:
        w = (analysis or {}).get("width", 0)
        h = (analysis or {}).get("height", 0)
        return {"x": 0, "y": 0, "w": w, "h": h}
    return estimate_content_bounds(video_path, threshold=thr)


# ── Sampling / scene cuts / tracking ─────────────────────────────────────────

def _frame_diff(a_gray_small, b_gray_small):
    import numpy as np
    return float(abs(a_gray_small.astype(float) - b_gray_small.astype(float)).mean())


def sample_clip(video_path, cfg=None):
    """Sample faces every sample_interval seconds.

    Returns dict {samples, scenes, width, height, fps, duration,
    content_bounds} where each sample is {t, boxes, scene_cut}. Tracking is
    reset at scene cuts. Stable association (not largest-face-per-frame):
    greedy nearest-centre matching carries track ids across samples within
    a shot.
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
    bounds = resolve_content_bounds(video_path, cfg,
                                    {"width": w, "height": h})
    return {"samples": samples, "scenes": scenes, "width": w, "height": h,
            "fps": fps, "duration": duration or (samples[-1]["t"] if samples else 0.0),
            "content_bounds": bounds}


def _effective_layout(requested, cfg=None):
    """Map a requested layout through the fullscreen requirement.

    branded_fit is manual-only and only honoured with full_screen_vertical
    off; otherwise the request is treated as auto (best crop wins).
    Returns (layout, was_mapped).
    """
    if requested == "branded_fit" and fullscreen(cfg):
        return "auto", True
    return requested, False


def decide_layout(analysis, cfg=None, override="auto"):
    """Stable layout decision from sampled tracks.

    - 1 reliably detected face -> speaker_crop
    - 2 reliably detected faces in a wide shot -> stacked_split
    - camera cut to single close-up -> speaker_crop (handled per-segment)
    - missing/uncertain -> center_crop (full-screen; marked needs_review)
    Stability: a briefly missed face must not flap layouts — require
    layout_stability_frames consecutive samples before switching.
    """
    cfg = cfg or {}
    override = validate_layout(override)
    mapped, was_mapped = _effective_layout(override, cfg)
    note = ("branded_fit is fit-only and disabled by full_screen_vertical; "
            "auto crop selection used instead. " if was_mapped else "")
    if mapped != "auto":
        return {"layout": mapped, "reason": f"manual override {mapped}",
                "stable": True, "needs_review": False}
    samples = analysis.get("samples", [])
    if not samples:
        return {"layout": "center_crop",
                "reason": note + "no samples (unreadable clip); static centre crop",
                "stable": True, "needs_review": True}
    stab = max(2, int(cfg.get("layout_stability_frames", 6)))
    min_face_frac = float(cfg.get("min_face_fraction", 0.5))
    if not min_face_frac:
        min_face_frac = 0.5
    votes = {"speaker_crop": 0, "stacked_split": 0, "center_crop": 0}
    # per-sample vote with hysteresis window
    window = []
    for s in samples:
        n = s.get("n", 0)
        if n == 1:
            v = "speaker_crop"
        elif n >= 2:
            v = "stacked_split"
        else:
            v = "center_crop"
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
        return {"layout": "center_crop",
                "reason": note + f"faces in only {with_faces}/{len(samples)} samples "
                          f"(< {min_face_frac:.0%}); face-follow where tracked, "
                          f"static centre crop elsewhere (needs review)",
                "stable": True, "needs_review": True, "votes": votes}
    layout = max(votes, key=votes.get)
    conf = votes[layout] / total
    # camera-cut awareness: if scenes exist with different face counts, auto
    # rendering splits per scene (see render); the clip-level choice stays stable.
    n_scenes = len(analysis.get("scenes", [0.0]))
    reason = (note + f"stable majority {layout} ({votes[layout]}/{total} windows, "
              f"conf={conf:.2f}, scenes={n_scenes})")
    return {"layout": layout, "reason": reason, "stable": True,
            "needs_review": layout == "center_crop", "votes": votes}


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
    # resolution floor: never upscale from a postage stamp
    cw = max(cw, int(OUT_W / 4)); ch = max(ch, int(OUT_H / 4))
    headroom = float(cfg.get("headroom_ratio", 0.35))
    # place face at headroom from top of crop
    y = int(min(max(cy - ch * headroom, 0), max(0, h - ch)))
    x = int(min(max(cx - cw / 2, 0), max(0, w - cw)))
    return x, y, cw, ch


def _crop_in_bounds(bounds, cx, cy, cfg):
    """9:16 crop in full-frame coords, computed inside the content region."""
    bw, bh = max(1, bounds["w"]), max(1, bounds["h"])
    x, y, cw, ch = _crop_for_cx(bw, bh, cx - bounds["x"], cy - bounds["y"], cfg)
    return x + bounds["x"], y + bounds["y"], cw, ch


def _centre_of_bounds(bounds):
    return bounds["x"] + bounds["w"] / 2, bounds["y"] + bounds["h"] / 2


def validate_roi(roi, bounds=None):
    """Validate a manual 9:16 ROI {x,y,w,h} (+optional t_start/t_end).

    Returns a normalised dict or raises ValueError. Ratio must be 9:16
    within tolerance; bounds-clamping happens at render (source dims may
    legitimately be unknown here).
    """
    try:
        x, y, w, h = (float(roi["x"]), float(roi["y"]),
                      float(roi["w"]), float(roi["h"]))
    except (KeyError, TypeError, ValueError):
        raise ValueError("ROI needs numeric x/y/w/h")
    if w <= 0 or h <= 0:
        raise ValueError("ROI width/height must be positive")
    if abs((w / h) - (OUT_W / OUT_H)) > 0.03:
        raise ValueError(f"ROI must be 9:16 (got {w:.0f}x{h:.0f})")
    out = {"x": x, "y": y, "w": w, "h": h}
    for k in ("t_start", "t_end"):
        if roi.get(k) is not None:
            try:
                out[k] = float(roi[k])
            except (TypeError, ValueError):
                raise ValueError(f"ROI {k} must be seconds or omitted")
    if "t_start" in out and "t_end" in out and out["t_end"] <= out["t_start"]:
        raise ValueError("ROI requires t_start < t_end")
    if bounds:
        out["x"] = min(max(out["x"], 0.0), max(0.0, bounds.get("w", 0) - 1))
        out["y"] = min(max(out["y"], 0.0), max(0.0, bounds.get("h", 0) - 1))
    return out


def _work_area_roi(cfg, bounds):
    """Configured DIY fallback ROI ({x,y,w,h} source pixels), or None."""
    wa = (cfg or {}).get("work_area")
    if not isinstance(wa, dict):
        return None
    try:
        return validate_roi(wa)
    except ValueError as e:
        log.warning("ignoring invalid framing.work_area: %s", e)
        return None


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


# ── Per-scene crop plan ───────────────────────────────────────────────────────

def plan_scene_crops(analysis, cfg=None, manual_rois=None):
    """Plan one crop anchor per scene segment (split at scene cuts).

    Each entry: {t0, t1, kind, cx, cy, track_id, reason, needs_review} with
    kind in {face, retain, workarea, center, manual}. Manual ROIs are
    recorded separately (authoritative at render) but listed so the plan
    shows coverage. Fallback chain per scene: reliable track -> retained
    previous anchor -> configured work_area -> static centre. Never bars.
    """
    cfg = cfg or {}
    samples = analysis.get("samples", []) or []
    bounds = analysis.get("content_bounds") or {
        "x": 0, "y": 0,
        "w": analysis.get("width", 1280), "h": analysis.get("height", 720)}
    scenes = sorted(set([0.0] + [float(s) for s in (analysis.get("scenes", []) or [0.0])]))
    duration = analysis.get("duration", samples[-1]["t"] if samples else 0.0) or 0.0
    edges = scenes + [duration]
    wa = _work_area_roi(cfg, bounds)
    ccx, ccy = _centre_of_bounds(bounds)

    plan, prev_anchor = [], None
    for si in range(len(edges) - 1):
        t0, t1 = edges[si], edges[si + 1]
        if t1 <= t0:
            continue
        seg = [s for s in samples if t0 - 1e-6 <= s["t"] < t1 + 1e-6]
        # most persistent track = dominant subject of this scene
        counts = {}
        for s in seg:
            for tr in s.get("tracks", []):
                counts.setdefault(tr["id"], []).append((tr["cx"], tr["cy"]))
        if counts:
            tid = max(counts, key=lambda i: len(counts[i]))
            xs = sorted(c[0] for c in counts[tid])
            ys = sorted(c[1] for c in counts[tid])
            anchor = {"kind": "face", "cx": xs[len(xs) // 2], "cy": ys[len(ys) // 2],
                      "track_id": tid,
                      "reason": f"scene {si}: stable track {tid} "
                                f"({len(counts[tid])}/{max(1, len(seg))} samples)",
                      "needs_review": False}
        elif prev_anchor is not None:
            anchor = {"kind": "retain", "cx": prev_anchor["cx"], "cy": prev_anchor["cy"],
                      "track_id": prev_anchor.get("track_id"),
                      "reason": f"scene {si}: no face; retaining established crop",
                      "needs_review": True}
        elif wa is not None:
            anchor = {"kind": "workarea",
                      "cx": wa["x"] + wa["w"] / 2, "cy": wa["y"] + wa["h"] / 2,
                      "track_id": None,
                      "reason": f"scene {si}: no face; configured work_area crop",
                      "needs_review": True}
        else:
            anchor = {"kind": "center", "cx": ccx, "cy": ccy, "track_id": None,
                      "reason": f"scene {si}: no face; static centre crop",
                      "needs_review": True}
        prev_anchor = anchor
        plan.append({"t0": round(t0, 3), "t1": round(t1, 3), **anchor})
    log.info("scene plan scenes=%d faces=%s", len(plan),
             sum(1 for p in plan if p["kind"] == "face"))
    return plan


def _manual_roi_at(manual_rois, t):
    """Latest manual ROI covering time t (authoritative), or None."""
    hit = None
    for roi in manual_rois or []:
        try:
            t0 = float(roi.get("t_start", -1e9))
            t1 = float(roi.get("t_end", 1e9))
        except (TypeError, ValueError):
            continue
        if t0 <= t <= t1:
            hit = roi  # last-added wins on overlap
    return hit


def resolve_anchor(scene_plan, smoothed, t, cfg=None, manual_rois=None,
                   content_bounds=None, duration=0.0):
    """Resolve the crop anchor for one timestamp. Never returns bars.

    Order: manual ROI > scene face/retain anchor (with smoothed follow for
    face anchors) > work_area > static centre.
    """
    cfg = cfg or {}
    manual = _manual_roi_at(manual_rois, t)
    if manual is not None:
        return {"kind": "manual",
                "cx": manual["x"] + manual["w"] / 2,
                "cy": manual["y"] + manual["h"] / 2,
                "needs_review": False, "reason": "manual ROI (authoritative)"}
    seg = None
    for p in scene_plan or []:
        if p["t0"] - 1e-6 <= t <= p["t1"] + 1e-6:
            seg = p
            break
    if seg is not None and seg["kind"] == "face":
        pos = _interp_centres(smoothed, t)
        if pos is not None and pos.get("cx") is not None and not pos.get("lost"):
            return {"kind": "face", "cx": pos["cx"], "cy": pos["cy"],
                    "needs_review": False, "reason": seg["reason"]}
        # brief loss inside a face scene: hold the scene anchor, then chain
        if pos is not None and pos.get("cx") is not None:
            return {"kind": "retain", "cx": pos["cx"], "cy": pos["cy"],
                    "needs_review": True,
                    "reason": seg["reason"] + " + holding through dropout"}
        return {"kind": "retain", "cx": seg["cx"], "cy": seg["cy"],
                "needs_review": True, "reason": seg["reason"] + " (anchor held)"}
    if seg is not None and seg["kind"] in ("retain", "workarea", "center"):
        return {"kind": seg["kind"], "cx": seg["cx"], "cy": seg["cy"],
                "needs_review": seg.get("needs_review", True),
                "reason": seg["reason"]}
    bounds = content_bounds or {"x": 0, "y": 0, "w": 1280, "h": 720}
    wa = _work_area_roi(cfg, bounds)
    if wa is not None:
        return {"kind": "workarea", "cx": wa["x"] + wa["w"] / 2,
                "cy": wa["y"] + wa["h"] / 2, "needs_review": True,
                "reason": "no scene plan; configured work_area crop"}
    ccx, ccy = _centre_of_bounds(bounds)
    return {"kind": "center", "cx": ccx, "cy": ccy, "needs_review": True,
            "reason": "no scene plan; static centre crop"}


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
    this returns {} and the caller uses stacked/centre crops.
    """
    if not diarization:
        return {}
    speaker_map = speaker_map or {}
    if not speaker_map:
        return {}  # uncertain -> stacked or centre crop
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


def center_crop_rect(w, h, bounds=None):
    """Static 9:16 centre window (x, y, cw, ch) in full-frame coords."""
    b = bounds or {"x": 0, "y": 0, "w": w, "h": h}
    bw, bh = max(1, b["w"]), max(1, b["h"])
    target = OUT_W / OUT_H
    ch, cw = bh, int(bh * target)
    if cw > bw:
        cw, ch = bw, int(bw / target)
    x = b["x"] + (bw - cw) // 2
    y = b["y"] + (bh - ch) // 2
    return int(x), int(y), int(cw), int(ch)


def render_center_crop(input_path, output_path):
    """Dependency-free 9:16 centre crop (exception fallback). Never letterboxes.

    Used when face-aware framing raises: static centre window of the content
    region, aspect preserved, audio carried over. Returns output_path.
    """
    import cv2
    import subprocess
    from config import ffmpeg_path
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(input_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    bounds = resolve_content_bounds(input_path, {},
                                    {"width": w, "height": h})
    x, y, cw, ch = center_crop_rect(w, h, bounds)
    # even dims for yuv420p
    cw, ch = (cw // 2) * 2, (ch // 2) * 2
    vf = f"crop={cw}:{ch}:{x}:{y},scale={OUT_W}:{OUT_H},setsar=1"
    r = subprocess.run(
        [ffmpeg_path(), "-y", "-i", input_path, "-vf", vf,
         "-map", "0:v", "-map", "0:a?",
         "-c:v", "libx264", "-preset", "fast", "-crf", "23",
         "-c:a", "aac", "-shortest", output_path],
        capture_output=True, text=True)
    if r.returncode != 0:
        log.error("center-crop fallback failed rc=%d err=%.300s",
                  r.returncode, (r.stderr or "")[-300:])
        raise RuntimeError(f"center-crop fallback failed: {(r.stderr or '')[-300:]}")
    log.info("center-crop fallback done in=%.60s out=%.60s rect=%d,%d %dx%d",
             input_path, output_path, x, y, cw, ch)
    return output_path


def _stack_track_ids(analysis):
    """Two most persistent tracks, ordered left->right. [] when fewer exist."""
    seen = {}
    for s in analysis.get("samples", []):
        for tr in s.get("tracks", []):
            seen.setdefault(tr["id"], []).append((tr["cx"], tr["cy"]))
    ranked = sorted(seen, key=lambda i: len(seen[i]), reverse=True)[:2]
    if len(ranked) < 2:
        return []
    return sorted(ranked, key=lambda i: sum(c[0] for c in seen[i]) / len(seen[i]))


def render_vertical(input_path, output_path, layout="auto", cfg=None,
                    analysis=None, diarization=None, manual_rois=None):
    """Render input_path (a cut clip) to 1080x1920 output_path.

    With full_screen_vertical on (default), EVERY frame is a 9:16 crop —
    face follow, stacked panels, work-area, retained or centre crop — and
    manual_rois (edit-plan overrides) win over detection unconditionally.
    Returns {output_path, layout, layout_reason, width, height, frames,
    analysis, scene_plan, content_bounds, needs_review, manual_rois}.
    Audio is preserved from the input (stream copy via ffmpeg mux pass).
    """
    import cv2
    import subprocess
    from config import ffmpeg_path
    cfg = cfg or {}
    fs = fullscreen(cfg)
    layout = validate_layout(layout)
    if fs and layout == "branded_fit":
        # fit-only requested under the fullscreen requirement: auto wins
        layout = "auto"
        _mapped_note = ("branded_fit requested but full_screen_vertical is on; "
                        "auto crop selection used (no bars)")
    else:
        _mapped_note = ""
    t0 = time.monotonic()
    if analysis is None:
        analysis = sample_clip(input_path, cfg)
    bounds = analysis.get("content_bounds") or resolve_content_bounds(
        input_path, cfg, analysis)
    analysis["content_bounds"] = bounds
    decision = decide_layout(analysis, cfg, layout)
    final_layout = decision["layout"]
    if _mapped_note:
        decision = dict(decision, reason=_mapped_note + " | " + decision.get("reason", ""))
    w, h = analysis["width"], analysis["height"]
    fps = analysis.get("fps", 25.0) or 25.0
    smoothed = smooth_centres(analysis, cfg)
    scene_plan = plan_scene_crops(analysis, cfg, manual_rois)
    needs_review = bool(decision.get("needs_review")) or any(
        p.get("needs_review") for p in scene_plan)
    divider = int(cfg.get("split_divider_px", 6))

    cap = cv2.VideoCapture(input_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    tmp_video = output_path + ".video.mp4"
    proc = _ffmpeg_writer(output_path, fps)
    import numpy as np

    # stacked assignment: left/right stable tracks for the whole clip
    stack_ids = []
    if final_layout == "stacked_split":
        stack_ids = _stack_track_ids(analysis)
        if len(stack_ids) < 2:
            # not enough subjects: fall back to single full-screen crop chain
            final_layout = "speaker_crop"
            decision = dict(decision, layout=final_layout,
                            reason=decision.get("reason", "") +
                            " | <2 stable tracks; single crop instead of stacked")

    i = 0
    kinds = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = i / fps
        i += 1

        if not fs and final_layout == "branded_fit":
            # legacy path: only reachable with full_screen_vertical: false
            bg = _hex_to_bgr(cfg.get("branded_background", "#0B0F1A"))
            canvas = np.full((OUT_H, OUT_W, 3), bg, dtype=np.uint8)
            scale = min(OUT_W / w, OUT_H / h)
            nw, nh = max(2, int(w * scale) // 2 * 2), max(2, int(h * scale) // 2 * 2)
            small = cv2.resize(frame, (nw, nh))
            y0, x0 = (OUT_H - nh) // 2, (OUT_W - nw) // 2
            canvas[y0:y0 + nh, x0:x0 + nw] = small
            out = canvas
            kinds["branded_fit"] = kinds.get("branded_fit", 0) + 1
        elif final_layout == "stacked_split" and len(stack_ids) >= 2:
            # one crop per person, stacked top/bottom; each panel fills its area
            panel_h = (OUT_H - divider) // 2
            panels = []
            for tid in stack_ids[:2]:
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
                x, y, cw, ch = _crop_in_bounds(bounds, best["cx"], best["cy"], cfg)
                crop = frame[y:y + ch, x:x + cw]
                panels.append(cv2.resize(crop, (OUT_W, panel_h)))
            out = np.vstack([panels[0],
                             np.full((divider, OUT_W, 3), 20, dtype=np.uint8),
                             panels[1]])
            if out.shape[0] != OUT_H:  # odd divider guard
                out = cv2.resize(out, (OUT_W, OUT_H))
            kinds["stacked_split"] = kinds.get("stacked_split", 0) + 1
        else:  # speaker_crop / center_crop: always a full-frame 9:16 crop
            anchor = resolve_anchor(scene_plan, smoothed, t, cfg,
                                    manual_rois, bounds,
                                    analysis.get("duration", 0.0))
            kinds[anchor["kind"]] = kinds.get(anchor["kind"], 0) + 1
            if anchor["kind"] == "manual":
                roi = _manual_roi_at(manual_rois, t)
                x, y, cw, ch = (int(roi["x"]), int(roi["y"]),
                                int(roi["w"]), int(roi["h"]))
                # clamp into frame (source dims may have been unknown at edit)
                cw = min(cw, w); ch = min(ch, h)
                x = min(max(x, 0), max(0, w - cw))
                y = min(max(y, 0), max(0, h - ch))
            elif anchor["kind"] == "workarea":
                wa = _work_area_roi(cfg, bounds)
                if wa is not None:
                    x, y, cw, ch = (int(wa["x"]), int(wa["y"]),
                                    int(wa["w"]), int(wa["h"]))
                    cw = min(cw, w); ch = min(ch, h)
                    x = min(max(x, 0), max(0, w - cw))
                    y = min(max(y, 0), max(0, h - ch))
                else:
                    x, y, cw, ch = _crop_in_bounds(bounds, anchor["cx"], anchor["cy"], cfg)
            elif anchor["kind"] == "center" and final_layout == "center_crop":
                x, y, cw, ch = center_crop_rect(w, h, bounds)
            else:
                x, y, cw, ch = _crop_in_bounds(bounds, anchor["cx"], anchor["cy"], cfg)
            if ch <= 0 or cw <= 0:
                x, y, cw, ch = center_crop_rect(w, h, bounds)
            crop = frame[y:y + ch, x:x + cw]
            out = cv2.resize(crop, (OUT_W, OUT_H))
        try:
            proc.stdin.write(out.tobytes())
        except BrokenPipeError:
            break
    cap.release()
    # hard fail on truncation: a short write must never pass silently
    # (main.py falls back to a full-length static crop on any exception)
    expected = int(round((analysis.get("duration", 0.0) or 0.0) * fps))
    rc = None
    try:
        proc.stdin.close()
        rc = proc.wait(timeout=120)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    if rc not in (0, None):
        raise RuntimeError(f"framing encoder exited rc={rc} after {i} frames")
    if expected > 0 and i < int(expected * 0.95):
        raise RuntimeError(
            f"framing truncated: wrote {i}/{expected} frames "
            f"(source {analysis.get('width')}x{analysis.get('height')})")
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
    log.info("framing done in=%.60s out=%.60s layout=%s frames=%d kinds=%s elapsed=%.1fs",
             input_path, output_path, final_layout, i, kinds, dur)
    return {"output_path": output_path, "layout": final_layout,
            "layout_reason": decision.get("reason", ""), "width": OUT_W,
            "height": OUT_H, "frames": i, "analysis": analysis,
            "scene_plan": scene_plan, "content_bounds": bounds,
            "needs_review": needs_review, "manual_rois": manual_rois or [],
            "anchor_kinds": kinds}


# ── Crop debug previews ───────────────────────────────────────────────────────

def render_crop_debug(input_path, out_path, analysis=None, scene_plan=None,
                      manual_rois=None, cfg=None, at=None, n=6):
    """Draw the selected crop rectangle on source frames for visual review.

    Green rect = live anchor crop; amber = heuristic (needs review);
    blue = content bounds; magenta = manual ROI. Writes one JPG per
    timestamp (or n spread across duration) with `_t{t}s` suffixes.
    Returns the list of written paths.
    """
    import cv2
    cfg = cfg or {}
    if analysis is None:
        analysis = sample_clip(input_path, cfg)
    bounds = analysis.get("content_bounds") or resolve_content_bounds(
        input_path, cfg, analysis)
    if scene_plan is None:
        scene_plan = plan_scene_crops(analysis, cfg, manual_rois)
    smoothed = smooth_centres(analysis, cfg)
    duration = analysis.get("duration", 0.0) or 0.0
    if at is None:
        at = [round(duration * (i + 1) / (n + 1), 2) for i in range(n)]
    cap = cv2.VideoCapture(input_path)
    paths = []
    base, ext = (out_path.rsplit(".", 1) + ["jpg"])[:2]
    for t in at:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            continue
        dbg = frame.copy()
        bx, by, bw, bh = (int(bounds["x"]), int(bounds["y"]),
                          int(bounds["w"]), int(bounds["h"]))
        cv2.rectangle(dbg, (bx, by), (bx + bw, by + bh), (255, 0, 0), 3)
        anchor = resolve_anchor(scene_plan, smoothed, t, cfg, manual_rois,
                                bounds, duration)
        roi = _manual_roi_at(manual_rois, t)
        if roi is not None:
            cv2.rectangle(dbg, (int(roi["x"]), int(roi["y"])),
                          (int(roi["x"] + roi["w"]), int(roi["y"] + roi["h"])),
                          (255, 0, 255), 3)
        if anchor["kind"] == "manual" and roi is not None:
            x, y, cw, ch = (int(roi["x"]), int(roi["y"]),
                            int(roi["w"]), int(roi["h"]))
        elif anchor["kind"] == "center":
            x, y, cw, ch = center_crop_rect(frame.shape[1], frame.shape[0], bounds)
        else:
            x, y, cw, ch = _crop_in_bounds(bounds, anchor["cx"], anchor["cy"], cfg)
        color = (0, 255, 0) if not anchor.get("needs_review") else (0, 165, 255)
        cv2.rectangle(dbg, (x, y), (x + cw, y + ch), color, 3)
        cv2.putText(dbg, f"t={t}s {anchor['kind']}", (max(10, x), max(40, y - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        p = f"{base}_t{t}s.{ext}"
        cv2.imwrite(p, dbg)
        paths.append(p)
    cap.release()
    log.info("crop debug previews=%d out=%s", len(paths), out_path)
    return paths
