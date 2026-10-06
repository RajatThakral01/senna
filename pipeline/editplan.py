"""
pipeline/editplan.py

Structured, versioned edit plans shared by preview and final export (Phase 5).

A plan reuses existing structures (clip source_ranges/timeline from
boundaries+clipper, caption cues from captions.py, layout+scenes from
framing analysis) instead of duplicating them, and adds:
  - audio settings (normalization, ducking, denoise, fades)
  - branding/export settings (logo, music, codec, crf, fps, SAR)
  - model + configuration versions + input config fingerprint

Crop paths are NOT baked per-pixel in v1: preview and export both run the
same deterministic framing functions over the same cut clip with the same
config fingerprint, so the same plan yields the same pixels. (Full keyframe
driving is explicit future work, recorded in plan["limits"].)

Plans persist to the edit_plans table (versioned per clip) and to
clips/clip_N_plan.json. config_fingerprint lets preview/export refuse
stale plans when settings changed.
"""
import hashlib
import json
import time

from logger import get_logger

log = get_logger("pipeline.editplan")

PLAN_VERSION = 1


def plan_fingerprint(settings: dict) -> str:
    raw = json.dumps(settings or {}, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def build_edit_plan(clip, campaign_config=None, config=None,
                    framing_analysis=None, caption_data=None,
                    source_info=None):
    """Assemble the edit plan dict for one clip. Pure (no IO)."""
    config = config or {}
    campaign_config = campaign_config or {}
    framing_cfg = config.get("framing", {})
    cap_cfg = config.get("captions", {})
    audio_cfg = config.get("audio", {})
    exp_cfg = config.get("export", {})
    fades_cfg = config.get("fades", {})

    ranges = [[float(a), float(b)] for a, b in
              (clip.get("source_ranges") or
               [[clip.get("start_time", 0), clip.get("end_time", 0)]])]
    timeline = clip.get("timeline") or []
    out_dur = round(sum(b - a for a, b in ranges), 3)

    scenes = []
    if framing_analysis:
        scenes = [{"t": round(float(s), 3)} for s in
                  framing_analysis.get("scenes", [])]
    face_frac = None
    try:
        samples = (framing_analysis or {}).get("samples", [])
        if samples:
            with_faces = sum(1 for s in samples if s.get("n", 0) > 0)
            face_frac = round(with_faces / len(samples), 3)
    except Exception:
        pass

    settings = {
        "layout": clip.get("layout", "auto"),
        "captions": {k: cap_cfg.get(k) for k in
                     ("enabled", "renderer", "max_words", "max_chars",
                      "max_duration", "font", "font_size", "placement")},
        "framing": {k: framing_cfg.get(k) for k in
                    ("sample_interval", "deadzone_ratio", "smooth_factor",
                     "headroom_ratio", "max_zoom", "branded_background")},
        "audio": audio_cfg,
        "export": exp_cfg,
        "fades": fades_cfg,
        "logo": campaign_config.get("logo"),
        "logo_position": campaign_config.get("logo_position", "top-right"),
        "music": campaign_config.get("music"),
        "music_volume": campaign_config.get("music_volume", 0.1),
        "fade": campaign_config.get("fade", True),
    }
    src = source_info or {}
    cap_style_keys = ("font", "font_size", "primary_color",
                      "highlight_color", "outline_color", "outline_width",
                      "shadow", "margin_v", "margin_v_top", "max_lines")
    plan = {
        "plan_version": PLAN_VERSION,
        "clip_number": clip.get("clip_number"),
        "clip_id": str(clip.get("id", "")),
        "source_ranges": ranges,
        "timeline": timeline,
        "output_duration": out_dur,
        "captions": {
            "cues": (caption_data or {}).get("cues", []),
            "placement": (caption_data or {}).get("placement", "bottom"),
            "cue_count": len((caption_data or {}).get("cues", [])),
            "style": {k: cap_cfg.get(k) for k in cap_style_keys},
        },
        "scenes": scenes,
        "layout": clip.get("layout", "auto"),
        "layout_reason": clip.get("layout_reason", ""),
        "face_fraction": face_frac,
        "audio": {
            "normalize_loudness": audio_cfg.get("normalize_loudness", True),
            "peak_limit_db": audio_cfg.get("peak_limit_db", -1.5),
            "duck_music": audio_cfg.get("duck_music", True),
            "denoise": audio_cfg.get("denoise", False),
            "audio_fade": fades_cfg.get("audio", False),
            "fade_duration": fades_cfg.get("duration", 0.5),
        },
        "branding": {
            "logo": settings["logo"], "logo_position": settings["logo_position"],
            "music": settings["music"], "music_volume": settings["music_volume"],
            "fade": settings["fade"],
        },
        "export": {
            "width": 1080, "height": 1920,
            "codec": exp_cfg.get("codec", "libx264"),
            "preset": exp_cfg.get("preset", "fast"),
            "crf": exp_cfg.get("crf", 23),
            "audio_bitrate": exp_cfg.get("audio_bitrate", "128k"),
            "fps": exp_cfg.get("fps"),  # null = keep source fps
            "setsar": True,
            "upscaled_from": f"{src.get('width', '?')}x{src.get('height', '?')}",
            "detail_note": ("Exporting at 1080x1920 does NOT restore detail "
                            "lost from a narrow low-resolution crop."),
        },
        "versions": {
            "whisper_model": (config.get("transcription", {}) or {}).get("model"),
            "embed_model": (config.get("embeddings", {}) or {}).get("model"),
            "llm_model": (config.get("ai", {}) or {}).get("llm_model"),
            "face_model": "blaze_face_short_range",
        },
        "config_fingerprint": plan_fingerprint(settings),
        "limits": ["crop paths recomputed deterministically (not baked); "
                   "same inputs + fingerprint => same pixels"],
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return plan


def save_plan_file(clip_number, plan):
    import os
    os.makedirs("clips", exist_ok=True)
    path = f"clips/clip_{clip_number}_plan.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    return path


def persist_plan(video_id, clip_id, clip_number, plan):
    """Save to DB (versioned) + clips/clip_N_plan.json. Returns (id, path)."""
    from db.repositories import editplan_repo
    pid, version = editplan_repo.save_plan(video_id, clip_id, plan)
    plan["version"] = version
    path = save_plan_file(clip_number, plan)
    log.info("edit plan saved clip=%s version=%s path=%s",
             clip_number, plan.get("version"), path)
    return pid, path
