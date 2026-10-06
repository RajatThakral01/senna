"""
pipeline/visual.py

Optional candidate-focused visual analysis interface (Phase 7).

Samples frames at scene changes + around proposed moments (dense only where
promising, never every source frame) and exposes an adapter for labels like
action / demonstration / reaction / on-screen text.

Backends: "none" (default, disabled with explicit status) or a configured
LLM-vision provider. No vision-capable model is available on the current
provider key, so the groq_vision backend reports unavailable instead of
fake outputs. Results cache per (video, range) in transcripts/visual.json;
max_frames + enabled flag bound cost/resources.
"""
import json
import os
import time

from logger import get_logger

log = get_logger("pipeline.visual")

CACHE_PATH = os.path.join("transcripts", "visual.json")


def backend_status(cfg=None):
    cfg = (cfg or {}).get("visual", {}) if isinstance(cfg, dict) else {}
    backend = str(cfg.get("backend", "none")).lower()
    if backend == "none" or not cfg.get("enabled", False):
        return {"backend": backend, "status": "disabled",
                "reason": "visual analysis not enabled"}
    if backend == "groq_vision":
        return {"backend": backend, "status": "unavailable",
                "reason": "no vision-capable model on the configured key"}
    return {"backend": backend, "status": "disabled",
            "reason": f"unknown visual backend {backend!r}"}


def sample_keyframes(video_path, ranges, framing_analysis=None, cfg=None,
                     max_frames=12):
    """Extract keyframe paths at scene changes + moment midpoints.

    Returns list of {t, path, kind}. Frames land in clips/ (gitignored).
    Candidate-focused: scene cuts inside ranges + range midpoints only.
    """
    import subprocess
    from config import ffmpeg_path
    cfg = (cfg or {}).get("visual", {}) if isinstance(cfg, dict) else {}
    max_frames = int(cfg.get("max_frames", max_frames) or max_frames)
    scenes = []
    try:
        scenes = [float(s) for s in (framing_analysis or {}).get("scenes", [])]
    except (TypeError, ValueError):
        scenes = []
    targets = []
    for a, b in ranges:
        cuts = [t for t in scenes if a <= t <= b]
        mids = [(a + b) / 2]
        for t in sorted(set(cuts + mids)):
            targets.append((round(t, 2), "cut" if t in cuts else "mid"))
    targets = targets[:max_frames]
    os.makedirs("clips", exist_ok=True)
    out = []
    for i, (t, kind) in enumerate(targets):
        path = f"clips/visual_{i}_{t}.jpg"
        r = subprocess.run([ffmpeg_path(), "-y", "-ss", str(t),
                            "-i", video_path, "-frames:v", "1",
                            "-q:v", "4", path],
                           capture_output=True)
        if r.returncode == 0 and os.path.exists(path):
            out.append({"t": t, "path": path, "kind": kind})
    log.info("visual keyframes=%d ranges=%d", len(out), len(ranges))
    return out


def _load_cache():
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_cache(cache):
    try:
        os.makedirs(os.path.dirname(CACHE_PATH) or ".", exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except OSError:
        pass


def analyze_moments(video_id, ranges, keyframes, cfg=None):
    """Run the visual backend over keyframes (cached). Returns (labels, status).

    labels: {range_idx: {tags: [...], note: ...}}. With backend disabled,
    returns ({}, status) — never fake labels.
    """
    st = backend_status(cfg)
    if st["status"] != "ready":
        # groq_vision/unavailable also land here: explicit, no fakes
        if st["status"] != "disabled":
            log.info("visual backend %s", st.get("reason"))
        return {}, st
    # Backend-specific analysis would run here when a ready backend exists.
    return {}, dict(st, status="disabled",
                    reason="ready backend has no implementation yet")


def cache_key(video_id, ranges):
    return f"{video_id}:" + ",".join(f"{a:.1f}-{b:.1f}" for a, b in ranges)
