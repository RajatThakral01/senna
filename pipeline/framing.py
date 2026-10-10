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

Detection + tracking: YuNet (OpenCV, small/multi-face capable) > MediaPipe >
Haar, sampled by sequential decode (no inaccurate seeking) so sample times
line up exactly with rendered frames. A persistent tracker keeps identities
through short detection dropouts; one-off false positives are discarded.

Stability ("virtual camera"): the camera path is planned OFFLINE per scene
for the scene's subject track — median outlier rejection, deadzone (locked
shot while the subject stays put), zero-phase Gaussian smoothing at frame
rate, and a pan-speed limit — so motion eases in/out with no per-sample
stepping. Crops are sub-pixel (warpAffine), paths never cross scene cuts,
zoom bounded by min_zoom/max_zoom with a resolution floor (crop window
never smaller than OUT_W/4 x OUT_H/4).

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

Active speaker (framing.active_speaker, default on): in scenes with 2+
people and layout auto/speaker_crop, pipeline/active_speaker.py scores each
face's lip motion (gated by audio voicing) and a Viterbi director emits one
shot per speaking turn — hard cut to the speaker, min_shot_seconds hold,
stacked two-shot while two people talk at once. Weak/absent lip evidence
falls back to the stacked / single-subject behaviour. The legacy
diarization + explicit speaker_map adapter (allow_active_speaker_switch)
stays separate and off; audio diarization alone never maps to a face.

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

YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

_detector_singleton = None
_haar_singleton = None
_yunet_singleton = None
_yunet_failed = False


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


def get_yunet(cfg=None):
    """OpenCV YuNet face detector singleton, or None when unavailable.

    Preferred backend: handles small faces in wide multi-person shots far
    better than BlazeFace short-range (a selfie-distance model). Model
    (~230KB ONNX) is downloaded once from opencv_zoo into models/; set
    framing.yunet_model to a local path to skip the download.
    """
    global _yunet_singleton, _yunet_failed
    if _yunet_singleton is not None or _yunet_failed:
        return _yunet_singleton
    cfg = cfg or {}
    try:
        import cv2
        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError("cv2.FaceDetectorYN missing (opencv < 4.5.4)")
        path = cfg.get("yunet_model") or os.path.join(
            "models", "face_detection_yunet_2023mar.onnx")
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            log.info("downloading YuNet face model -> %s", path)
            urllib.request.urlretrieve(cfg.get("yunet_model_url") or YUNET_MODEL_URL, path)
        _yunet_singleton = cv2.FaceDetectorYN.create(
            path, "", (320, 320),
            float(cfg.get("face_confidence", 0.6)), 0.3, 50)
        log.info("yunet face detector ready model=%s", path)
    except Exception:
        log.warning("yunet unavailable; trying mediapipe/haar", exc_info=True)
        _yunet_failed = True
        _yunet_singleton = None
    return _yunet_singleton


def _box(x1, y1, x2, y2, score=1.0, landmarks=None):
    b = {"x1": x1, "y1": y1, "x2": x2, "y2": y2,
         "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2, "score": float(score)}
    if landmarks is not None:
        b["landmarks"] = landmarks
    return b


def _detect_yunet(det, frame_bgr, cfg):
    import cv2
    h, w = frame_bgr.shape[:2]
    dw = int(cfg.get("detect_width", 960) or w)
    scale = min(1.0, dw / max(1, w))
    img = frame_bgr if scale >= 1.0 else cv2.resize(
        frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    det.setInputSize((img.shape[1], img.shape[0]))
    _, faces = det.detect(img)
    out = []
    for f in (faces if faces is not None else []):
        x, y, fw, fh = (float(v) / scale for v in f[:4])
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = min(float(w), x + fw), min(float(h), y + fh)
        if x2 <= x1 or y2 <= y1:
            continue
        # 5 landmarks: right eye, left eye, nose, right/left mouth corner
        lm = [[float(f[4 + 2 * k]) / scale, float(f[5 + 2 * k]) / scale]
              for k in range(5)]
        out.append(_box(x1, y1, x2, y2, f[14], lm))
    return out


def _detect_mediapipe(det, frame_bgr):
    import cv2
    import mediapipe as _mp  # Image/ImageFormat live here in mp>=0.10
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = det.detect(_mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb))
    h, w = frame_bgr.shape[:2]
    out = []
    for d in res.detections or []:
        bb = d.bounding_box
        x1 = max(0.0, float(bb.origin_x)); y1 = max(0.0, float(bb.origin_y))
        x2 = min(float(w), float(bb.origin_x) + float(bb.width))
        y2 = min(float(h), float(bb.origin_y) + float(bb.height))
        if x2 <= x1 or y2 <= y1:
            continue
        score = d.categories[0].score if getattr(d, "categories", None) else 1.0
        out.append(_box(x1, y1, x2, y2, score))
    return out


def detect_faces_bgr(frame_bgr, cfg=None):
    """Return [{x1,y1,x2,y2,cx,cy,score[,landmarks]}] in source pixels.

    Backend (framing.face_backend): auto (yunet > mediapipe > haar) | yunet
    | mediapipe | haar. Faces narrower than min_face_ratio of the frame width
    are dropped (background noise). Never raises.
    """
    cfg = cfg or {}
    backend = str(cfg.get("face_backend", "auto")).lower()
    min_w = float(cfg.get("min_face_ratio", 0.02)) * frame_bgr.shape[1]

    def _keep(boxes):
        return [b for b in boxes if (b["x2"] - b["x1"]) >= min_w]

    if backend in ("auto", "yunet"):
        try:
            det = get_yunet(cfg)
            if det is not None:
                return _keep(_detect_yunet(det, frame_bgr, cfg))
        except Exception:
            log.debug("yunet detect failed, trying next backend", exc_info=True)
    if backend in ("auto", "mediapipe"):
        try:
            det = get_detector()
            if det is not None:
                return _keep(_detect_mediapipe(det, frame_bgr))
        except Exception:
            log.debug("mediapipe detect failed, trying haar", exc_info=True)
    try:
        haar = _haar()
        if haar is not None:
            import cv2
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            rects = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                          minSize=(60, 60))
            return _keep([_box(float(x), float(y), float(x + w), float(y + h))
                          for (x, y, w, h) in rects])
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

class _CutDetector:
    """Hard-cut detector on ADJACENT frames (colour content + histogram).

    A cut is a sudden, one-frame change of picture: HSV content delta >=
    scene_cut_content, colour-histogram correlation < scene_cut_corr, AND an
    onset >= scene_cut_onset x the previous frame's change. Whip pans and
    fast motion change pixels over many frames with the same colours, so the
    onset and histogram tests reject them (measured on fast-cut footage:
    real cuts concentrate 65-100% of the change in one frame, pans <= 45%).
    The old 0.1s-apart grey diff fired ~4x per second on handheld footage,
    resetting tracking constantly.
    """

    def __init__(self, cfg):
        self.content = float(cfg.get("scene_cut_content", 27.0))
        self.corr = float(cfg.get("scene_cut_corr", 0.85))
        self.onset = float(cfg.get("scene_cut_onset", 2.5))
        self.min_gap = float(cfg.get("min_scene_seconds", 0.3))
        self.prev, self.prev_jump, self.last_cut = None, None, -1e9

    def update(self, frame, t):
        import cv2
        import numpy as np
        hsv = cv2.cvtColor(cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        cut = False
        if self.prev is not None:
            jump = float(np.abs(hsv.astype(np.int16) - self.prev[1].astype(np.int16)).mean())
            corr = float(cv2.compareHist(hist, self.prev[0], cv2.HISTCMP_CORREL))
            onset = self.prev_jump is None or jump >= self.onset * max(self.prev_jump, 1.0)
            if (jump >= self.content and corr < self.corr and onset
                    and t - self.last_cut >= self.min_gap):
                cut, self.last_cut = True, t
            self.prev_jump = jump
        self.prev = (hist, hsv)
        return cut


_saliency_singleton = None


def salient_cx(frame_bgr, prev_gray=None, win_frac=None):
    """Horizontal centre of the most visually important region (source px), or None.

    Spectral-residual saliency on a 320px-wide thumbnail, weighted by motion
    against prev_gray (the adjacent frame) when given — the player mid-dunk
    beats static bright ceiling lights. With win_frac (crop width / frame
    width) the result is the centre of the crop-width window holding the most
    salience, not the centre of mass: with two clusters (kids left, strongman
    right) the centre of mass lands on the empty field between them. Top 18%
    and bottom 15% bands are ignored (burned-in titles / captions). Used
    where there is no face to follow.
    """
    global _saliency_singleton
    try:
        import cv2
        import numpy as np
        if _saliency_singleton is None:
            _saliency_singleton = cv2.saliency.StaticSaliencySpectralResidual_create()
        h, w = frame_bgr.shape[:2]
        sw, sh = 320, max(1, int(320 * h / max(1, w)))
        small = cv2.resize(frame_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
        ok, m = _saliency_singleton.computeSaliency(small)
        if not ok:
            return None
        m = m.astype(np.float32)
        if prev_gray is not None:
            g0 = cv2.resize(prev_gray, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
            g1 = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
            mot = cv2.GaussianBlur(np.abs(g1 - g0), (9, 9), 0)
            mx = float(mot.max())
            if mx > 4.0:  # real motion somewhere (not just sensor noise)
                # motion dominates: static bright things (lights, signs) carry
                # ~3x the raw saliency of a moving person
                m = m * (0.1 + 0.9 * mot / mx)
        m = m[int(0.18 * sh):int(0.85 * sh)]
        thr = np.percentile(m, 90)
        mask = np.where(m >= thr, m, 0.0)
        cols = mask.sum(axis=0)
        tot = float(cols.sum())
        if tot <= 0:
            return None
        if win_frac:
            win = max(3, int(round(win_frac * sw)))
            if win < sw:
                dens = np.convolve(cols, np.ones(win), mode="valid")
                c0 = int(np.argmax(dens))
                return (c0 + win / 2.0) * (w / float(sw))
        return float((cols * np.arange(len(cols))).sum() / tot) * (w / float(sw))
    except Exception:
        return None


class FaceTracker:
    """Persistent multi-face tracker (identity kept through short dropouts).

    Matching is greedy on normalised centre distance (distance / face width),
    gated so a match can't jump to another person. Tracks survive
    `max_missing` seconds without a detection (blinks, head turns, a missed
    frame) instead of being re-created with a new id — id churn was what made
    the old crop flip between people mid-shot.
    """

    def __init__(self, frame_w, max_missing=1.5):
        self.frame_w = float(frame_w)
        self.max_missing = float(max_missing)
        self.tracks = []  # {id, cx, cy, bw, last_t, vx}
        self.next_id = 0

    def reset(self):
        self.tracks = []  # scene cut: new shot, new identities

    def update(self, boxes, t):
        live = [tr for tr in self.tracks if t - tr["last_t"] <= self.max_missing]
        pairs = []
        for i, tr in enumerate(live):
            dt = t - tr["last_t"]
            px, py = tr["cx"] + tr["vx"] * dt, tr["cy"]  # constant-velocity guess
            for j, b in enumerate(boxes):
                size = max(tr["bw"], b["x2"] - b["x1"], 1.0)
                d = math.hypot(b["cx"] - px, b["cy"] - py)
                if d <= max(1.5 * size, 0.06 * self.frame_w):
                    pairs.append((d / size, i, j))
        pairs.sort()
        used_t, used_b, matched = set(), set(), []
        for _, i, j in pairs:
            if i in used_t or j in used_b:
                continue
            used_t.add(i); used_b.add(j)
            tr, b = live[i], boxes[j]
            dt = max(t - tr["last_t"], 1e-3)
            tr["vx"] = 0.5 * tr["vx"] + 0.5 * (b["cx"] - tr["cx"]) / dt
            tr.update(cx=b["cx"], cy=b["cy"], bw=b["x2"] - b["x1"], last_t=t)
            matched.append({"id": tr["id"], "cx": b["cx"], "cy": b["cy"], "box": b})
        for j, b in enumerate(boxes):
            if j in used_b:
                continue
            tr = {"id": self.next_id, "cx": b["cx"], "cy": b["cy"],
                  "bw": b["x2"] - b["x1"], "last_t": t, "vx": 0.0}
            self.next_id += 1
            live.append(tr)
            matched.append({"id": tr["id"], "cx": b["cx"], "cy": b["cy"], "box": b})
        self.tracks = live
        return matched


def sample_clip(video_path, cfg=None):
    """Detect + track faces every sample_interval seconds.

    Returns dict {samples, scenes, width, height, fps, duration,
    content_bounds}; each sample is {t, boxes, tracks, scene_cut, n}.
    Frames are decoded sequentially (grab() between samples) and sample time
    is frame_index / fps — exactly the clock render_vertical uses — so crop
    positions never drift against the picture (OpenCV seeking on long-GOP
    H.264 lands on the wrong frame). Tracking resets at scene cuts; tracks
    seen fewer than min_track_hits times are dropped as false positives and
    `n` counts confirmed faces only.
    """
    import cv2
    import numpy as np
    cfg = cfg or {}
    interval = float(cfg.get("sample_interval", 0.1))
    min_hits = int(cfg.get("min_track_hits", 3))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if math.isnan(fps) or fps <= 0:
        fps = 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, int(round(fps * interval)))
    tracker = FaceTracker(w, float(cfg.get("track_max_missing_seconds", 1.5)))

    want_mouth = bool(cfg.get("active_speaker", True))
    cuts = _CutDetector(cfg)
    samples, scenes = [], [0.0]
    prev_gray, pending_cut = None, False
    idx = 0
    max_frames = int(fps * 900)  # guard against broken fps metadata
    while idx < max_frames:
        # every frame is decoded anyway; cut detection runs on all of them
        # (adjacent frames), detection/tracking only on sample frames
        ret, frame = cap.read()
        if not ret:
            break
        t = idx / fps
        is_sample = idx % step == 0
        idx += 1
        if cuts.update(frame, t):
            if t > 0:
                scenes.append(round(t, 3))
            tracker.reset()
            pending_cut = True
            prev_gray = None  # no lip motion across a cut
        if not is_sample:
            # the frame right before a sample: lip motion is measured between
            # adjacent frames (~33ms), not 0.1s apart
            if want_mouth and idx % step == 0:
                prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = detect_faces_bgr(frame, cfg)
        tracks = tracker.update(boxes, t)
        if want_mouth:
            from pipeline.active_speaker import mouth_motion
            for tr in tracks:
                tr["mouth"] = mouth_motion(prev_gray, gray, tr["box"])
        if step == 1:
            prev_gray = gray
        samples.append({"t": round(t, 3), "boxes": boxes, "tracks": tracks,
                        "scene_cut": pending_cut,
                        "sal_cx": salient_cx(frame, prev_gray, (h * OUT_W / OUT_H) / max(1, w))
                        if cfg.get("saliency_fallback", True) else None})
        pending_cut = False
    cap.release()
    duration = idx / fps
    if want_mouth and samples:
        from pipeline.active_speaker import audio_envelope, voiced_at
        env = audio_envelope(video_path, hop=interval)
        st = [s["t"] for s in samples]
        voiced = voiced_at(env, st, float(cfg.get("speaker_voice_db", 10.0)))
        dbs = np.interp(st, env[0], env[1]) if env is not None else [None] * len(st)
        for s, v, d in zip(samples, voiced, dbs):
            s["voiced"] = bool(v)
            if d is not None:
                s["db"] = round(float(d), 2)

    hits = {}
    for s in samples:
        for tr in s["tracks"]:
            hits[tr["id"]] = hits.get(tr["id"], 0) + 1
    for s in samples:
        s["tracks"] = [tr for tr in s["tracks"] if hits[tr["id"]] >= min_hits]
        s["n"] = len(s["tracks"])
    bounds = resolve_content_bounds(video_path, cfg,
                                    {"width": w, "height": h})
    log.info("sampled faces samples=%d step=%d tracks=%d confirmed=%d",
             len(samples), step, len(hits),
             sum(1 for v in hits.values() if v >= min_hits))
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
    cw, ch = _crop_size(w, h, cfg)
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


# ── Virtual camera path (offline, per scene) ─────────────────────────────────

def _scene_edges(analysis):
    """[(t0, t1), ...] scene segments covering the clip (split at cuts)."""
    samples = analysis.get("samples", []) or []
    scenes = sorted(set([0.0] + [float(s) for s in (analysis.get("scenes") or [0.0])]))
    duration = analysis.get("duration") or (samples[-1]["t"] if samples else 0.0) or 0.0
    edges = scenes + [max(float(duration), scenes[-1])]
    return [(a, b) for a, b in zip(edges, edges[1:]) if b > a]


def _scene_samples(samples, t0, t1, last):
    return [s for s in samples
            if t0 - 1e-6 <= s["t"] < t1 - 1e-6 or (last and s["t"] >= t1 - 1e-6)]


def _box_w(tr, default=1.0):
    b = tr.get("box") or {}
    return float(b.get("x2", 0) - b.get("x1", 0)) or default


def _edge_clipped(tr, frame_w):
    b = tr.get("box") or {}
    return bool(b) and frame_w and (b.get("x1", 1.0) <= 1.0 or b.get("x2", 0.0) >= frame_w - 1.0)


def _prominence(seg, frame_w=None):
    """{track_id: score} = sum over samples of face width x centrality.

    The camera operator frames the subject: a big face near the middle of the
    source frame beats a tiny bystander, and a face cut off by the frame edge
    counts half. Without frame_w this is plain presence x size.
    """
    score = {}
    for s in seg:
        for tr in s.get("tracks", []):
            c = 1.0
            if frame_w:
                off = abs(tr["cx"] - frame_w / 2.0) / (frame_w / 2.0)
                c = 1.0 - 0.7 * min(1.0, off)
                if _edge_clipped(tr, frame_w):
                    c *= 0.5
            score[tr["id"]] = score.get(tr["id"], 0.0) + _box_w(tr) * c
    return score


def _scene_subject(seg, frame_w=None):
    """Dominant subject of a scene: the most prominent face (see _prominence).

    One subject per scene keeps the crop from flipping between people; the
    active-speaker director overrides it when someone is clearly talking.
    """
    score = _prominence(seg, frame_w)
    return max(score, key=score.get) if score else None


def _crop_size(bw, bh, cfg):
    """(cw, ch) of the single 9:16 crop inside a bw x bh content region."""
    target_ratio = OUT_W / OUT_H
    ch = bh
    cw = int(bh * target_ratio)
    if cw > bw:  # very narrow source (rare): fit width instead
        cw, ch = bw, int(bw / target_ratio)
    max_zoom = float(cfg.get("max_zoom", 2.5)); min_zoom = float(cfg.get("min_zoom", 1.0))
    zoom = min(max(min_zoom, 1.0), max_zoom)
    if zoom > 1.0:
        cw = max(int(cw / zoom), int(OUT_W / 4)); ch = max(int(ch / zoom), int(OUT_H / 4))
    # resolution floor: never upscale from a postage stamp
    return max(cw, int(OUT_W / 4)), max(ch, int(OUT_H / 4))


def _median_filter(v, k):
    import numpy as np
    if k < 3 or len(v) < 3:
        return v
    r = k // 2
    p = np.pad(v, r, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(p, 2 * r + 1), axis=1)


def _gauss_smooth(v, sigma):
    """Zero-phase Gaussian smoothing (no lag), edge-padded."""
    import numpy as np
    if sigma < 0.5 or len(v) < 3:
        return v
    r = int(math.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(v, r, mode="edge"), k, mode="valid")


def _limit_speed(v, max_step):
    """Cap per-step motion; forward+backward average keeps it lag-free."""
    if max_step <= 0 or len(v) < 2:
        return v
    f = v.copy(); b = v.copy()
    for i in range(1, len(v)):
        f[i] = min(max(f[i], f[i - 1] - max_step), f[i - 1] + max_step)
    for i in range(len(v) - 2, -1, -1):
        b[i] = min(max(b[i], b[i + 1] - max_step), b[i + 1] + max_step)
    return (f + b) / 2.0


def plan_camera_axis(ts, vals, out_ts, cfg, span, lo=None, hi=None, frame_w=1280):
    """Plan one axis of the virtual camera for a scene.

    ts/vals: subject positions at sample times (None = not detected).
    out_ts: times to evaluate (render frames or samples). span: crop size on
    this axis (deadzone is a fraction of it). Returns a numpy array or None
    when the subject was never seen.

      1. fill gaps (hold/interpolate), clamp to the reachable range,
      2. median filter -> drops single-sample detection outliers,
      3. deadzone: the camera stays LOCKED while the subject moves inside
         deadzone_ratio * span; once it leaves, the target re-centres on it,
      4. zero-phase Gaussian (path_smooth_seconds) at output rate -> moves
         ease in/out and start slightly ahead of the subject,
      5. pan-speed limit (max_pan_speed source widths/s) + light re-smooth.
    """
    import numpy as np
    ts = np.asarray(ts, dtype=float)
    v = np.array([np.nan if x is None else float(x) for x in vals], dtype=float)
    ok = ~np.isnan(v)
    if not ok.any():
        return None
    v = np.interp(ts, ts[ok], v[ok])
    if lo is not None and hi is not None:
        v = np.clip(v, lo, max(lo, hi))
    v = _median_filter(v, int(cfg.get("median_window", 5)))
    dead = float(cfg.get("deadzone_ratio", 0.15)) * span
    cam = np.empty_like(v)
    c = float(np.median(v[:max(1, min(len(v), 10))]))
    for i, x in enumerate(v):
        if abs(x - c) > dead:
            c = x
        cam[i] = c
    out_ts = np.asarray(out_ts, dtype=float)
    idx = np.clip(np.searchsorted(ts, out_ts, side="right") - 1, 0, len(ts) - 1)
    path = cam[idx]
    dt = float(np.median(np.diff(out_ts))) if len(out_ts) > 1 else 0.04
    dt = max(dt, 1e-3)
    sigma = float(cfg.get("path_smooth_seconds", 0.45)) / dt
    path = _gauss_smooth(path, sigma)
    path = _limit_speed(path, float(cfg.get("max_pan_speed", 0.6)) * frame_w * dt)
    path = _gauss_smooth(path, sigma / 3.0)
    if lo is not None and hi is not None:
        path = np.clip(path, lo, max(lo, hi))
    return path


def _track_series(seg, tid):
    xs, ys = [], []
    for s in seg:
        tr = next((t for t in s.get("tracks", []) if t["id"] == tid), None)
        xs.append(tr["cx"] if tr else None)
        ys.append(tr["cy"] if tr else None)
    return [s["t"] for s in seg], xs, ys


def _overlap_frac(b, left, right):
    fw = max(1.0, b["x2"] - b["x1"])
    return max(0.0, min(right, b["x2"]) - max(left, b["x1"])) / fw


def _sliced_weight(boxes, left, right):
    """Size-weighted count of faces cut by a crop edge (partly in, partly out)."""
    tot = 0.0
    for b in boxes:
        f = _overlap_frac(b, left, right)
        if 0.05 < f < 0.95:
            tot += max(1.0, b["x2"] - b["x1"]) ** 0.7
    return tot


def _deslice_target(cx, cw, subj, others, lo, hi, max_shift):
    """Shift a crop centre so neighbouring faces are fully in or fully out.

    A face sliced by the crop edge is what makes a 9:16 frame look "odd".
    Candidate shifts bring each sliced face fully inside or push it fully
    outside; the subject must stay inside with a 5% margin and the shift is
    capped at max_shift. Fewest sliced faces wins, then the smallest shift.
    """
    if not others:
        return cx
    left = cx - cw / 2.0
    if _sliced_weight(others, left, left + cw) == 0:
        return cx
    shifts = {0.0}
    for b in others:
        fw = b["x2"] - b["x1"]
        m = 0.08 * fw
        shifts.update([(b["x2"] + m) - (left + cw),   # include: right edge past it
                       (b["x1"] - m) - left,          # include: left edge before it
                       (b["x1"] - m) - (left + cw),   # exclude: right edge before it
                       (b["x2"] + m) - left])         # exclude: left edge past it
    margin = 0.05 * cw
    best = None
    for sh in shifts:
        if abs(sh) > max_shift:
            continue
        c = min(max(cx + sh, lo), max(lo, hi))
        l2, r2 = c - cw / 2.0, c + cw / 2.0
        if subj and (subj["x2"] - subj["x1"]) < cw - 2 * margin:
            if subj["x1"] < l2 + margin or subj["x2"] > r2 - margin:
                continue
        key = (_sliced_weight(others, l2, r2), abs(c - cx))
        if best is None or key < best[0]:
            best = (key, c)
    return best[1] if best else cx


def track_path(seg, tid, out_ts, cfg, bounds, cw, ch, frame_w):
    """(cx_arr, cy_arr) camera centre path following track tid, or None."""
    ts, xs, ys = _track_series(seg, tid)
    if not ts:
        return None
    lo = bounds["x"] + cw / 2.0
    hi = bounds["x"] + bounds["w"] - cw / 2.0
    if all(x is None for x in xs):
        return None  # subject never seen here: caller falls back to the scene subject
    if cfg.get("saliency_fallback", True):
        # face lost for longer than hold_last_position_seconds (turned away,
        # left the frame, a faceless insert inside the shot): follow the
        # salient content instead of holding a stale position
        hold_n = max(1, int(float(cfg.get("hold_last_position_seconds", 1.0))
                            / max(float(cfg.get("sample_interval", 0.1)), 1e-3)))
        k = 0
        while k < len(xs):
            if xs[k] is not None:
                k += 1
                continue
            j = k
            while j < len(xs) and xs[j] is None:
                j += 1
            if j - k > hold_n:
                for q in range(k + hold_n, j):
                    xs[q] = seg[q].get("sal_cx")
            k = j
    if cfg.get("deslice_faces", True):
        max_shift = float(cfg.get("deslice_max_shift", 0.25)) * cw
        for k, s in enumerate(seg):
            if xs[k] is None:
                continue
            subj = next((t.get("box") for t in s.get("tracks", []) if t["id"] == tid), None)
            others = [t["box"] for t in s.get("tracks", [])
                      if t["id"] != tid and t.get("box")]
            xs[k] = _deslice_target(xs[k], cw, subj, others, lo, hi, max_shift)
    px = plan_camera_axis(ts, xs, out_ts, cfg, cw, lo, hi, frame_w)
    if px is None:
        return None
    py = plan_camera_axis(ts, ys, out_ts, cfg, ch, frame_w=frame_w)
    if py is None:  # defensive: hold the vertical centre of the content
        import numpy as np
        py = np.full(len(px), bounds["y"] + bounds["h"] / 2.0)
    return px, py


def smooth_centres(analysis, cfg=None):
    """Per-sample camera centres from the planned virtual-camera path.

    Each scene follows its dominant subject (see _scene_subject); paths never
    cross scene cuts. Returns [{t, cx, cy, missing, lost, scene_cut}] where
    `missing` counts consecutive samples without the subject and `lost` is
    set once that exceeds hold_last_position_seconds (render then holds /
    falls back through the crop chain, never bars).
    """
    cfg = cfg or {}
    samples = analysis.get("samples", []) or []
    w = analysis.get("width", 1280) or 1280
    h = analysis.get("height", 720) or 720
    bounds = analysis.get("content_bounds") or {"x": 0, "y": 0, "w": w, "h": h}
    cw, ch = _crop_size(max(1, bounds["w"]), max(1, bounds["h"]), cfg)
    hold_s = float(cfg.get("hold_last_position_seconds", 1.0))
    interval = float(cfg.get("sample_interval", 0.1))
    hold_n = max(1, int(hold_s / max(interval, 1e-3)))
    edges = _scene_edges(analysis)
    out = []
    for si, (t0, t1) in enumerate(edges):
        seg = _scene_samples(samples, t0, t1, si == len(edges) - 1)
        if not seg:
            continue
        tid = _scene_subject(seg, w)
        path = (track_path(seg, tid, [s["t"] for s in seg], cfg, bounds, cw, ch, w)
                if tid is not None else None)
        missing = 0
        for k, s in enumerate(seg):
            if any(tr["id"] == tid for tr in s.get("tracks", [])):
                missing = 0
            else:
                missing += 1
            out.append({"t": s["t"],
                        "cx": float(path[0][k]) if path else None,
                        "cy": float(path[1][k]) if path else None,
                        "missing": missing,
                        "lost": path is None or missing > hold_n,
                        "scene_cut": s.get("scene_cut", False),
                        "track_id": tid})
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
    edges = _scene_edges(analysis)
    wa = _work_area_roi(cfg, bounds)
    ccx, ccy = _centre_of_bounds(bounds)

    plan, prev_anchor = [], None
    for si, (t0, t1) in enumerate(edges):
        seg = _scene_samples(samples, t0, t1, si == len(edges) - 1)
        # dominant subject of this scene (persistence x face size)
        counts = {}
        for s in seg:
            for tr in s.get("tracks", []):
                counts.setdefault(tr["id"], []).append((tr["cx"], tr["cy"]))
        if counts:
            tid = _scene_subject(seg, analysis.get("width"))
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


def _stack_track_ids(analysis, seg=None, min_presence=0.0):
    """Two most persistent tracks, ordered left->right. [] when fewer exist.

    seg restricts the count to one scene's samples; tracks present in fewer
    than min_presence of those samples don't qualify (passers-by, posters).
    """
    seg = analysis.get("samples", []) if seg is None else seg
    seen = {}
    for s in seg:
        for tr in s.get("tracks", []):
            seen.setdefault(tr["id"], []).append((tr["cx"], tr["cy"]))
    need = min_presence * max(1, len(seg))
    ranked = [i for i in sorted(seen, key=lambda i: len(seen[i]), reverse=True)
              if len(seen[i]) >= need][:2]
    if len(ranked) < 2:
        return []
    return sorted(ranked, key=lambda i: sum(c[0] for c in seen[i]) / len(seen[i]))


def _warp_crop(frame, x, y, cw, ch, ow, oh, interp):
    """Crop (x, y, cw, ch) — floats allowed — and scale to ow x oh in one
    sub-pixel warp. Integer crops make slow pans step in visible
    (OUT_W / cw)-pixel jumps; this keeps motion continuous."""
    import cv2
    import numpy as np
    sx, sy = ow / float(cw), oh / float(ch)
    m = np.float32([[sx, 0, -x * sx], [0, sy, -y * sy]])
    return cv2.warpAffine(frame, m, (ow, oh), flags=interp,
                          borderMode=cv2.BORDER_REPLICATE)


def _place(bounds, cx, cy, cw, ch, headroom):
    """Top-left of a cw x ch window centred on cx with the face at
    `headroom` of the window height, clamped inside the content region."""
    bx, by, bw, bh = bounds["x"], bounds["y"], bounds["w"], bounds["h"]
    x = min(max(cx - cw / 2.0, bx), bx + max(0, bw - cw))
    y = min(max(cy - ch * headroom, by), by + max(0, bh - ch))
    return x, y


def _shot_crop_size(bw, bh, cfg, face_w):
    """Single-crop size for one shot, framed by the subject's face size.

    With auto_zoom on, the crop is sized so the face spans target_face_ratio
    of the output width (a real close-up instead of the full-height slice
    that also shows half the other guest). Bounded by the full-height crop
    (never wider) and by max_upscale (never blurrier than that). One size per
    shot — zoom never "breathes" with detection noise.
    """
    cw0, ch0 = _crop_size(bw, bh, cfg)
    if not cfg.get("auto_zoom", True) or not face_w or face_w <= 0:
        return cw0, ch0
    cw = face_w / max(0.05, float(cfg.get("target_face_ratio", 0.33)))
    cw_min = max(OUT_W / max(1.0, float(cfg.get("max_upscale", 2.4))), OUT_W / 4.0)
    cw = max(cw, cw_min)
    if cw >= cw0:
        return cw0, ch0
    return cw, cw * OUT_H / OUT_W


def _smoothstep(x):
    import numpy as np
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def emphasis_zoom(seg, out_ts, cfg):
    """Per-frame zoom factors (>= 1) punching in on emphatic moments.

    An emphasis is a voiced loudness peak: >= the shot's 95th-percentile
    level, >= emphasis_db above the local 2s median, and a local maximum.
    Peaks are taken loudest-first with emphasis_min_gap seconds between
    them and never within 0.6s of the shot's edges (no zoom across a cut).
    Each punch eases in over 0.2s, holds, and eases out.
    """
    import numpy as np
    out_ts = np.asarray(out_ts, dtype=float)
    z = np.ones(len(out_ts))
    amount = float(cfg.get("emphasis_zoom", 0.06))
    if amount <= 0 or len(out_ts) < 2 or len(seg) < 10:
        return z
    db = np.array([s.get("db", np.nan) for s in seg], dtype=float)
    ts = np.array([s["t"] for s in seg], dtype=float)
    voiced = np.array([bool(s.get("voiced", False)) for s in seg])
    ok = voiced & ~np.isnan(db)
    if ok.sum() < 10:
        return z
    interval = float(cfg.get("sample_interval", 0.1))
    r = max(1, int(round(1.0 / interval)))
    pad = np.pad(np.where(np.isnan(db), np.nanmedian(db), db), r, mode="edge")
    local = np.median(np.lib.stride_tricks.sliding_window_view(pad, 2 * r + 1), axis=1)
    p95 = float(np.percentile(db[ok], 95))
    rise = float(cfg.get("emphasis_db", 4.0))
    cand = [k for k in range(len(seg))
            if ok[k] and db[k] >= p95 and db[k] - local[k] >= rise
            and db[k] >= np.nanmax(db[max(0, k - 3):k + 4])
            and out_ts[0] + 0.6 <= ts[k] <= out_ts[-1] - 0.6]
    gap = float(cfg.get("emphasis_min_gap", 4.0))
    hold = float(cfg.get("emphasis_hold", 1.2))
    chosen = []
    for k in sorted(cand, key=lambda k: -db[k]):
        if all(abs(ts[k] - ts[c]) >= gap for c in chosen):
            chosen.append(k)
    for k in chosen:
        t0 = ts[k] - 0.15
        up = _smoothstep((out_ts - (t0 - 0.2)) / 0.2)
        down = 1.0 - _smoothstep((out_ts - (t0 + hold)) / 0.6)
        z = np.maximum(z, 1.0 + amount * np.minimum(up, down))
    return z


def _frame_rects(p, k, bounds, manual, w, h):
    """Source windows for frame k of render plan p: ([(x, y, cw, ch, out_h)], kind).

    Shared by the renderer and the debug video so the overlay shows exactly
    what was rendered.
    """
    if manual is not None:
        cw = min(float(manual["w"]), w); ch = min(float(manual["h"]), h)
        x = min(max(float(manual["x"]), 0.0), max(0.0, w - cw))
        y = min(max(float(manual["y"]), 0.0), max(0.0, h - ch))
        return [(x, y, cw, ch, OUT_H)], "manual"
    if p["mode"] == "stacked":
        rects = []
        for (px, py), ph in zip(p["panels"], p["heights"]):
            pch = p["pcw"] * ph / OUT_W
            x, y = _place(bounds, px[k], py[k], p["pcw"], pch, p["headroom"])
            rects.append((x, y, p["pcw"], pch, ph))
        return rects, "stacked_split"
    if p["kind"] in ("face", "group", "salient"):
        z = float(p["zoom"][k]) if p.get("zoom") is not None else 1.0
        cw, ch = p["cw"] / z, p["ch"] / z
        x, y = _place(bounds, p["cx"][k], p["cy"][k], cw, ch, p["headroom"])
        if p["kind"] in ("group", "salient"):
            return [(x, y, cw, ch, OUT_H)], p["kind"]
        return [(x, y, cw, ch, OUT_H)], ("retain" if p["lost"][k] else "face")
    x, y, cw, ch = p["rect"]
    if cw <= 0 or ch <= 0:
        x, y, cw, ch = center_crop_rect(w, h, bounds)
    return [(x, y, cw, ch, OUT_H)], p["kind"]


class _DebugWriter:
    """Side-by-side review video: source with overlays | rendered vertical.

    Overlays: every tracked face (id, lip-activity bar; the shot's subject in
    green), the crop window(s) actually rendered, and a status line (time,
    shot state, layout kind, voiced/dB, zoom). Audio is muxed so speaker
    decisions can be judged by ear. Failures never affect the real render.
    """

    def __init__(self, path, w, h, fps, analysis, width=960):
        import subprocess
        from config import ffmpeg_path
        self.path, self.fps, self.ok = path, fps, False
        self.s = width / float(max(1, w))
        self.sw, self.sh = width, int(round(h * self.s)) // 2 * 2
        self.rw = int(round(self.sh * OUT_W / OUT_H)) // 2 * 2
        self.samples = analysis.get("samples", []) or []
        self.si = 0
        try:
            self.proc = subprocess.Popen(
                [ffmpeg_path(), "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{self.sw + self.rw}x{self.sh}", "-r", f"{fps:.3f}", "-i", "-",
                 "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                 "-pix_fmt", "yuv420p", path + ".video.mp4"],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
        except Exception:
            log.warning("debug video disabled (encoder failed to start)", exc_info=True)
            self.proc = None

    def write(self, frame, out, rects, p, k, kind, t):
        if self.proc is None:
            return
        try:
            import cv2
            import numpy as np
            while (self.si + 1 < len(self.samples)
                   and self.samples[self.si + 1]["t"] <= t + 1e-6):
                self.si += 1
            smp = self.samples[self.si] if self.samples else {}
            s = self.s
            src = cv2.resize(frame, (self.sw, self.sh), interpolation=cv2.INTER_AREA)
            shot = p.get("shot") or {}
            subj = shot.get("state", p.get("track_id"))
            for tr in smp.get("tracks", []):
                b = tr.get("box") or {}
                if not b:
                    continue
                col = (0, 220, 0) if tr["id"] == subj else (0, 200, 255)
                p1 = (int(b["x1"] * s), int(b["y1"] * s))
                p2 = (int(b["x2"] * s), int(b["y2"] * s))
                cv2.rectangle(src, p1, p2, col, 2)
                m = tr.get("mouth")
                label = f"#{tr['id']}" + (f" lip {m:.2f}" if m is not None else "")
                cv2.putText(src, label, (p1[0], max(12, p1[1] - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
                if m is not None:
                    bar = int(min(1.0, m / 0.3) * (p2[0] - p1[0]))
                    cv2.rectangle(src, (p1[0], p2[1] + 3), (p1[0] + bar, p2[1] + 8), col, -1)
            for x, y, cw, ch, _ in rects:
                cv2.rectangle(src, (int(x * s), int(y * s)),
                              (int((x + cw) * s), int((y + ch) * s)), (255, 80, 255), 2)
            z = float(p["zoom"][k]) if p.get("zoom") is not None else 1.0
            st = shot.get("state", "-")
            info = (f"t={t:6.2f}s  {kind}  shot={st}  "
                    f"{'VOICED' if smp.get('voiced') else 'quiet '} "
                    f"{smp.get('db', float('nan')):5.1f}dB  zoom={z:.2f}")
            cv2.rectangle(src, (0, 0), (self.sw, 22), (0, 0, 0), -1)
            cv2.putText(src, info, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            right = cv2.resize(out, (self.rw, self.sh), interpolation=cv2.INTER_AREA)
            self.proc.stdin.write(np.ascontiguousarray(np.hstack([src, right])).tobytes())
        except Exception:
            log.warning("debug video write failed; disabling", exc_info=True)
            self._kill()

    def _kill(self):
        try:
            self.proc.kill()
        except Exception:
            pass
        self.proc = None

    def close(self, audio_src):
        import subprocess
        from config import ffmpeg_path
        if self.proc is None:
            return
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=120)
            tmp = self.path + ".video.mp4"
            r = subprocess.run(
                [ffmpeg_path(), "-y", "-i", tmp, "-i", audio_src, "-map", "0:v",
                 "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-shortest", self.path],
                capture_output=True, text=True)
            if r.returncode == 0:
                os.remove(tmp)
            else:
                os.replace(tmp, self.path)
            self.ok = True
            log.info("framing debug video %s", self.path)
        except Exception:
            log.warning("debug video finalize failed", exc_info=True)


def _eligible_people(seg, min_presence, frame_w=None, min_face_ratio=0.0):
    """Track ids visible in >= min_presence of the scene's samples, most
    prominent first.

    With frame_w + min_face_ratio, a person must also have a median face
    width >= min_face_ratio x frame width and not be cut off by the frame
    edge most of the time: lip motion on a 40px face is compression noise,
    and those faces were driving wrong-speaker picks in crowd shots.
    """
    import numpy as np
    seen, widths, edge = {}, {}, {}
    for s in seg:
        for tr in s.get("tracks", []):
            i = tr["id"]
            seen[i] = seen.get(i, 0) + 1
            widths.setdefault(i, []).append(_box_w(tr, 0.0))
            edge[i] = edge.get(i, 0) + (1 if _edge_clipped(tr, frame_w) else 0)
    need = min_presence * max(1, len(seg))
    ok = []
    for i, n in seen.items():
        if n < need:
            continue
        if frame_w and min_face_ratio > 0:
            ws = [x for x in widths[i] if x > 0]
            if ws and float(np.median(ws)) < min_face_ratio * frame_w:
                continue
            if edge[i] > 0.5 * n:
                continue
        ok.append(i)
    prom = _prominence(seg, frame_w)
    return sorted(ok, key=lambda i: -prom.get(i, 0.0))


def _group_target(boxes, cw, lo, hi, frame_w=None):
    """Crop centre that frames the most whole faces (crowd shots).

    Candidates centre on each face, align a crop edge just outside each face,
    or sit on the weighted centroid. Score = whole faces minus 1.5x sliced
    faces, each face weighted by size and (with frame_w) by closeness to the
    centre of the source frame: the camera operator points at the action, so
    a cluster of sideline faces must not outvote the subject in the middle.
    Centrality falls off as (1 - offset)^1.5, so edge columns of bystanders
    barely count, and faces cut off by the source frame edge count 20%.
    (Saliency is deliberately NOT used here: on crowd footage it locks onto
    jerseys, field lines and titles and dragged the crop off the subject.)
    """
    if not boxes:
        return None
    wts = []
    for b in boxes:
        wt = max(1.0, b["x2"] - b["x1"]) ** 0.7
        if frame_w:
            off = min(1.0, abs(b["cx"] - frame_w / 2.0) / (frame_w / 2.0))
            wt *= max(0.03, 1.0 - off) ** 1.5      # strong centre preference in crowds
            if b["x1"] <= 1.0 or b["x2"] >= frame_w - 1.0:
                wt *= 0.2
        wts.append(max(wt, 1e-3))
    centroid = sum(b["cx"] * w for b, w in zip(boxes, wts)) / sum(wts)
    cands = {centroid}
    for b in boxes:
        m = 0.15 * (b["x2"] - b["x1"])
        cands.update([b["cx"], b["x1"] - m + cw / 2.0, b["x2"] + m - cw / 2.0])
    best = None
    for c in cands:
        c = min(max(c, lo), max(lo, hi))
        left, right = c - cw / 2.0, c + cw / 2.0
        whole = sum(w for b, w in zip(boxes, wts) if _overlap_frac(b, left, right) >= 0.95)
        score = whole - 1.5 * _sliced_weight(boxes, left, right) - 1e-4 * abs(c - centroid)
        if best is None or score > best[0]:
            best = (score, c)
    return best[1]


def _render_plan(analysis, scene_plan, cfg, fps, n_frames, layout, bounds,
                 requested=None):
    """Per-shot render instructions with per-frame camera paths.

    Returns [{f0, f1, mode, ...}] where mode is
      single  — one full-screen crop: cx/cy arrays (+ lost flags) or a
                fixed rect for retain/workarea/centre scenes,
      stacked — two panels, each with its own smoothed path and a crop of
                the PANEL's aspect ratio (no squashing).
    Scenes with 2+ people and requested layout auto/speaker_crop go through
    the active-speaker director (pipeline.active_speaker): one shot per
    speaking turn, hard cuts between speakers, stacked panels while two
    people talk over each other. Without reliable lip evidence the scene
    keeps the single-subject / stacked behaviour.
    """
    import numpy as np
    samples = analysis.get("samples", []) or []
    w = analysis.get("width", 1280)
    bw, bh = max(1, bounds["w"]), max(1, bounds["h"])
    cw, ch = _crop_size(bw, bh, cfg)
    headroom = float(cfg.get("headroom_ratio", 0.35))
    hold_s = float(cfg.get("hold_last_position_seconds", 1.0))
    divider = int(cfg.get("split_divider_px", 6))
    top_h = (OUT_H - divider) // 2
    bot_h = OUT_H - divider - top_h
    requested = requested or layout
    use_director = (bool(cfg.get("active_speaker", True))
                    and requested in ("auto", "speaker_crop"))

    def frames(t0, t1, last):
        f0 = int(round(t0 * fps))
        return f0, (n_frames if last else max(f0 + 1, int(round(t1 * fps))))

    def stacked(seg, stack, f0, f1):
        out_ts = np.arange(f0, f1) / fps
        # two panels need two people on screen AT THE SAME TIME: a second face
        # that only shows up for the last 0.5s of a close-up once produced a
        # split of the same face twice (its panel held a position on top of
        # the first person until it appeared)
        both = sum(1 for s in seg
                   if set(stack) <= {tr["id"] for tr in s.get("tracks", [])})
        if both < float(cfg.get("stack_min_concurrent", 0.6)) * max(1, len(seg)):
            return None
        fws = [_box_w(tr, 0.0) for s in seg for tr in s.get("tracks", [])
               if tr["id"] in stack]
        fws = [f for f in fws if f > 0]
        fw = float(np.median(fws)) if fws else bw * 0.08
        aspect = OUT_W / float(top_h)
        pcw = fw * float(cfg.get("panel_face_scale", 3.2))
        pcw = min(max(pcw, OUT_W / 4.0), bw * 0.55, bh * aspect)
        pch = pcw / aspect
        panels = [track_path(seg, tid, out_ts, cfg, bounds, pcw, pch, w) for tid in stack]
        if any(p is None for p in panels):
            return None
        # panels that overlap would show the same picture twice
        if float(np.median(np.abs(panels[0][0] - panels[1][0]))) < 0.8 * pcw:
            return None
        return {"f0": f0, "f1": f1, "mode": "stacked", "kind": "stacked_split",
                "panels": panels, "pcw": pcw, "pch": pch, "heights": (top_h, bot_h),
                "headroom": float(cfg.get("panel_headroom_ratio", 0.4)),
                "track_ids": stack}

    def single(seg, tid, f0, f1):
        out_ts = np.arange(f0, f1) / fps
        fws = [_box_w(tr, 0.0) for s in seg for tr in s.get("tracks", [])
               if tr["id"] == tid]
        fws = [f for f in fws if f > 0]
        scw, sch = _shot_crop_size(bw, bh, cfg, float(np.median(fws)) if fws else 0.0)
        p = track_path(seg, tid, out_ts, cfg, bounds, scw, sch, w)
        if p is None:
            return None
        # frames more than hold_s from any subject detection = "retain"
        seen = np.array([s["t"] for s in seg
                         if any(tr["id"] == tid for tr in s.get("tracks", []))])
        if len(seen):
            j = np.clip(np.searchsorted(seen, out_ts), 1, len(seen)) - 1
            gap = np.minimum(np.abs(out_ts - seen[j]),
                             np.abs(out_ts - seen[np.minimum(j + 1, len(seen) - 1)]))
            lost = gap > hold_s
        else:
            lost = np.ones(len(out_ts), dtype=bool)
        return {"f0": f0, "f1": f1, "mode": "single", "kind": "face",
                "cx": p[0], "cy": p[1], "lost": lost, "cw": scw, "ch": sch,
                "headroom": headroom, "track_id": tid,
                "zoom": emphasis_zoom(seg, out_ts, cfg)}

    def group(seg, f0, f1):
        """Crowd shot: the crop follows whichever position frames the most
        whole faces (no single speaker is reliable)."""
        out_ts = np.arange(f0, f1) / fps
        lo, hi = bounds["x"] + cw / 2.0, bounds["x"] + bounds["w"] - cw / 2.0
        ts = [s["t"] for s in seg]
        xs = [_group_target([tr["box"] for tr in s.get("tracks", []) if tr.get("box")],
                            cw, lo, hi, w) for s in seg]
        if cfg.get("saliency_fallback", True):
            # no faces for longer than hold: follow the salient action instead
            # of holding the crowd position (a dunk framed as ceiling lights)
            hold_n = max(1, int(hold_s / max(float(cfg.get("sample_interval", 0.1)), 1e-3)))
            k = 0
            while k < len(xs):
                if xs[k] is not None:
                    k += 1
                    continue
                j = k
                while j < len(xs) and xs[j] is None:
                    j += 1
                if j - k > hold_n:
                    for q in range(k + hold_n, j):
                        xs[q] = seg[q].get("sal_cx")
                k = j
        px = plan_camera_axis(ts, xs, out_ts, cfg, cw, lo, hi, w)
        if px is None:
            return None
        cys = [tr["cy"] for s in seg for tr in s.get("tracks", [])]
        cy = float(np.median(cys)) if cys else bounds["y"] + bh / 2.0
        return {"f0": f0, "f1": f1, "mode": "single", "kind": "group",
                "cx": px, "cy": np.full(len(out_ts), cy),
                "lost": np.zeros(len(out_ts), dtype=bool), "cw": cw, "ch": ch,
                "headroom": headroom, "track_id": None}

    def by_x(ids, seg):
        xs = {i: [tr["cx"] for s in seg for tr in s.get("tracks", []) if tr["id"] == i]
              for i in ids}
        return sorted(ids, key=lambda i: sum(xs[i]) / max(1, len(xs[i])))

    edges = _scene_edges(analysis) or [(0.0, n_frames / fps)]
    plans = []
    for si, (t0, t1) in enumerate(edges):
        last = si == len(edges) - 1
        f0, f1 = frames(t0, t1, last)
        seg = _scene_samples(samples, t0, t1, last)
        sp = next((p for p in scene_plan or []
                   if abs(p["t0"] - round(t0, 3)) < 1e-3), None)

        presence = float(cfg.get("speaker_min_presence", 0.2))
        # everyone visible vs. people whose faces are big enough to read lips
        present = _eligible_people(seg, presence, w) if seg else []
        people = (_eligible_people(seg, presence, w,
                                   float(cfg.get("speaker_min_face_ratio", 0.045)))
                  if seg else [])
        prom = _prominence(seg, w) if seg else {}
        ranked = sorted(prom.values(), reverse=True)
        dominant = (len(ranked) == 1 or
                    (len(ranked) >= 2 and ranked[0] >= float(cfg.get("crowd_dominance", 2.5))
                     * max(ranked[1], 1e-6)))
        crowd = len(present) >= int(cfg.get("crowd_min_people", 3)) and not dominant

        if use_director and len(people) >= 2:
            from pipeline.active_speaker import direct_shots, GROUP
            cand = people[:int(cfg.get("speaker_max_people", 4))]
            top = max(prom.get(i, 0.0) for i in cand) or 1.0
            prior = {i: prom.get(i, 0.0) / top for i in cand}
            try:
                shots = direct_shots(seg, cand, cfg, prior=prior)
            except Exception:
                # one scene's director failure must not drop the whole clip
                # to a static centre crop: this scene uses Phase-A framing
                log.exception("active speaker director failed scene=%d; "
                              "single-subject/stacked framing for this scene", si)
                shots = None
            if shots and crowd:
                # In a crowd, a single-person close-up must be earned: a weak
                # winner (score = how much more active than the rest) is a
                # guess, and guessing one face out of a group reads as
                # "wrong person". Frame the group instead.
                min_sc = float(cfg.get("crowd_speaker_min_score", 0.25))
                if min(sh["score"] for sh in shots) < min_sc:
                    log.info("scene %d crowd: speaker evidence weak (min score %.2f < %.2f); "
                             "group framing", si, min(sh["score"] for sh in shots), min_sc)
                    shots = None
            if shots:
                ents = []
                for k, sh in enumerate(shots):
                    s_last = last and k == len(shots) - 1
                    st1 = sh["t1"] if sh["t1"] is not None else t1
                    g0, g1 = frames(sh["t0"], st1, s_last)
                    if k == 0:
                        g0 = f0
                    if k == len(shots) - 1:
                        g1 = f1
                    sseg = _scene_samples(seg, sh["t0"], st1, sh["t1"] is None)
                    if sh["state"] == GROUP:
                        ent = stacked(sseg, by_x(people[:2], seg), g0, g1)
                    else:
                        ent = single(sseg, sh["state"], g0, g1)
                    if ent is None:  # subject unseen in this shot: scene subject
                        ent = single(seg, people[0], g0, g1)
                    ent["shot"] = sh
                    ents.append(ent)
                if all(e is not None for e in ents):
                    plans.extend(ents)
                    continue

        # No reliable speaker. Explicit stacked_split keeps its two panels;
        # automatic stacking only for exactly two prominent people (never a
        # bystander pair picked out of a crowd).
        if layout == "stacked_split" and seg:
            if requested == "stacked_split":
                stack = _stack_track_ids(analysis, seg,
                                         float(cfg.get("stack_min_presence", 0.3)))
            else:
                stack = by_x(people[:2], seg) if (len(people) == 2 and not crowd) else []
            if len(stack) == 2:
                ent = stacked(seg, stack, f0, f1)
                if ent is not None:
                    plans.append(ent)
                    continue

        if crowd and seg:
            ent = group(seg, f0, f1)
            if ent is not None:
                plans.append(ent)
                continue

        if sp is not None and sp["kind"] == "face" and seg:
            ent = single(seg, sp.get("track_id"), f0, f1)
            if ent is not None:
                plans.append(ent)
                continue

        # faceless scene (b-roll, product insert): follow the salient content
        if (seg and cfg.get("saliency_fallback", True)
                and (sp is None or sp["kind"] in ("center", "retain"))):
            sal = [s.get("sal_cx") for s in seg]
            if any(v is not None for v in sal):
                out_ts = np.arange(f0, f1) / fps
                lo, hi = bounds["x"] + cw / 2.0, bounds["x"] + bounds["w"] - cw / 2.0
                px = plan_camera_axis([s["t"] for s in seg], sal, out_ts, cfg, cw, lo, hi, w)
                if px is not None:
                    plans.append({"f0": f0, "f1": f1, "mode": "single", "kind": "salient",
                                  "cx": px, "cy": np.full(len(out_ts), bounds["y"] + bh / 2.0),
                                  "lost": np.zeros(len(out_ts), dtype=bool), "cw": cw, "ch": ch,
                                  "headroom": 0.5, "track_id": None})
                    continue

        # static scene: retained anchor, work area, or centre crop
        entry = {"f0": f0, "f1": f1, "mode": "single"}
        kind = sp["kind"] if sp is not None else "center"
        if kind == "workarea" and _work_area_roi(cfg, bounds) is not None:
            wa = _work_area_roi(cfg, bounds)
            rect = (wa["x"], wa["y"], min(wa["w"], w), min(wa["h"], analysis.get("height", bh)))
        elif kind == "center" or sp is None:
            kind = "center"
            rect = center_crop_rect(w, analysis.get("height", bh), bounds)
        else:
            x, y = _place(bounds, sp["cx"], sp["cy"], cw, ch, headroom)
            rect = (x, y, cw, ch)
        entry.update(kind=kind, rect=rect)
        plans.append(entry)
    return plans


def render_vertical(input_path, output_path, layout="auto", cfg=None,
                    analysis=None, diarization=None, manual_rois=None,
                    debug_path=None):
    """Render input_path (a cut clip) to 1080x1920 output_path.

    With full_screen_vertical on (default), EVERY frame is a 9:16 crop —
    face follow, stacked panels, work-area, retained or centre crop — and
    manual_rois (edit-plan overrides) win over detection unconditionally.
    Camera paths are planned offline per scene (see plan_camera_axis) and
    stacked layouts are decided per scene (a single-person scene inside a
    two-person clip renders as one full-screen crop).
    Returns {output_path, layout, layout_reason, width, height, frames,
    analysis, scene_plan, content_bounds, needs_review, manual_rois}.
    Audio is preserved from the input (stream copy via ffmpeg mux pass).
    debug_path (or framing.debug_video: true -> <output>_debug.mp4) also
    writes a side-by-side review video (see _DebugWriter).
    """
    import cv2
    import subprocess
    import numpy as np
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
    scene_plan = plan_scene_crops(analysis, cfg, manual_rois)
    needs_review = bool(decision.get("needs_review")) or any(
        p.get("needs_review") for p in scene_plan)
    divider = int(cfg.get("split_divider_px", 6))
    interp = {"linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC,
              "lanczos": cv2.INTER_LANCZOS4}.get(
        str(cfg.get("interpolation", "cubic")).lower(), cv2.INTER_CUBIC)

    cap = cv2.VideoCapture(input_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    n_est = max(n_frames, int(round((analysis.get("duration", 0.0) or 0.0) * fps))) + 2
    tmp_video = output_path + ".video.mp4"
    plans = _render_plan(analysis, scene_plan, cfg, fps, n_est, final_layout, bounds,
                         requested=layout)
    shots = [dict(p["shot"], mode=p["mode"]) for p in plans if "shot" in p]
    if shots:
        n_group = sum(1 for s in shots if s["mode"] == "stacked")
        final_layout = "speaker_crop"
        decision = dict(decision, layout=final_layout,
                        reason=decision.get("reason", "") +
                        f" | active-speaker director: {len(shots)} shots"
                        f" ({n_group} two-shot), cuts on speaker change")
    elif final_layout == "stacked_split" and not any(p["mode"] == "stacked" for p in plans):
        final_layout = "speaker_crop"
        decision = dict(decision, layout=final_layout,
                        reason=decision.get("reason", "") +
                        " | <2 stable tracks in any scene; single crop instead of stacked")
    proc = _ffmpeg_writer(output_path, fps)

    dbg_path = debug_path or (output_path.rsplit(".", 1)[0] + "_debug.mp4"
                              if cfg.get("debug_video") else None)
    dbg = _DebugWriter(dbg_path, w, h, fps, analysis) if dbg_path else None
    i, pi = 0, 0
    kinds = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = i / fps
        while pi < len(plans) - 1 and i >= plans[pi]["f1"]:
            pi += 1
        p = plans[pi]
        k = min(max(i - p["f0"], 0), max(0, p["f1"] - p["f0"] - 1))
        i += 1
        manual = _manual_roi_at(manual_rois, t)

        if not fs and final_layout == "branded_fit":
            # legacy path: only reachable with full_screen_vertical: false
            bg = _hex_to_bgr(cfg.get("branded_background", "#0B0F1A"))
            canvas = np.full((OUT_H, OUT_W, 3), bg, dtype=np.uint8)
            scale = min(OUT_W / w, OUT_H / h)
            nw, nh = max(2, int(w * scale) // 2 * 2), max(2, int(h * scale) // 2 * 2)
            small = cv2.resize(frame, (nw, nh))
            y0, x0 = (OUT_H - nh) // 2, (OUT_W - nw) // 2
            canvas[y0:y0 + nh, x0:x0 + nw] = small
            out, rects, kind = canvas, [], "branded_fit"
        else:
            rects, kind = _frame_rects(p, k, bounds, manual, w, h)
            tiles = [_warp_crop(frame, x, y, cw, ch, OUT_W, oh, interp)
                     for x, y, cw, ch, oh in rects]
            if len(tiles) == 2:
                out = np.vstack([tiles[0],
                                 np.full((divider, OUT_W, 3), 20, dtype=np.uint8),
                                 tiles[1]])
            else:
                out = tiles[0]
        kinds[kind] = kinds.get(kind, 0) + 1
        try:
            proc.stdin.write(np.ascontiguousarray(out).tobytes())
        except BrokenPipeError:
            break
        if dbg is not None:
            dbg.write(frame, out, rects, p, k, kind, t)
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
    if dbg is not None:
        dbg.close(input_path)
    dur = time.monotonic() - t0
    log.info("framing done in=%.60s out=%.60s layout=%s frames=%d kinds=%s elapsed=%.1fs",
             input_path, output_path, final_layout, i, kinds, dur)
    return {"output_path": output_path, "layout": final_layout,
            "layout_reason": decision.get("reason", ""), "width": OUT_W,
            "height": OUT_H, "frames": i, "analysis": analysis,
            "scene_plan": scene_plan, "content_bounds": bounds,
            "needs_review": needs_review, "manual_rois": manual_rois or [],
            "anchor_kinds": kinds, "shots": shots,
            "debug_path": dbg_path if dbg is not None and dbg.ok else None}


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
