"""
pipeline/fingerprints.py

Config/source fingerprints for checkpoint invalidation.

A stage is re-run when its fingerprint changes (source, model, prompt, or
relevant config changed). Fingerprints are short hex digests over the inputs
that actually affect each stage — never secrets.
"""
import hashlib
import json


def compute_fingerprint(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# Bump these when a prompt template changes.
PROMPT_VERSIONS = {
    "outline_section": "v1",
    "outline_combine": "v1",
    "discovery": "v1",
    "fusion_rank": "v1",
}


def outline_fingerprint(transcript_sig: str, llm_model: str, cfg: dict) -> str:
    o = cfg.get("outline", {})
    return compute_fingerprint({
        "stage": "outline",
        "transcript": transcript_sig,
        "model": llm_model,
        "prompt": PROMPT_VERSIONS["outline_section"] + "+" + PROMPT_VERSIONS["outline_combine"],
        "window_sentences": o.get("window_sentences", 40),
    })


def discovery_fingerprint(transcript_sig: str, outline_sig: str, llm_model: str,
                          cfg: dict) -> str:
    d = cfg.get("discovery", {})
    return compute_fingerprint({
        "stage": "discovery",
        "transcript": transcript_sig,
        "outline": outline_sig,
        "model": llm_model,
        "prompt": PROMPT_VERSIONS["discovery"],
        "target_clips": d.get("target_clips", 8),
        "max_candidates": d.get("max_candidates", 24),
        "min_span": d.get("min_span_seconds", 15),
        "max_span": d.get("max_span_seconds", 90),
        "allow_noncontiguous": d.get("allow_noncontiguous", False),
    })


def audio_events_fingerprint(audio_sig: str, cfg: dict) -> str:
    a = cfg.get("audio_events", {})
    return compute_fingerprint({
        "stage": "audio_events",
        "audio": audio_sig,
        "win_ms": a.get("window_ms", 200),
        "baseline_sec": a.get("baseline_seconds", 10),
        "min_duration": a.get("min_duration_seconds", 0.4),
        "max_event": a.get("max_event_seconds", 8.0),
        "threshold_db": a.get("threshold_db", 8.0),
        "merge_gap": a.get("merge_gap_seconds", 1.0),
        "pre": a.get("pre_seconds", 25),
        "post": a.get("post_seconds", 10),
        "min_cov": a.get("min_speech_coverage", 0.3),
        "classifier": (a.get("classifier", {}) or {}).get("backend", "none"),
    })


def transcript_signature(transcript_path: str) -> str:
    """Stable id for transcript content (mtime + size; content hash if small)."""
    import os
    try:
        st = os.stat(transcript_path)
        return f"{int(st.st_mtime)}:{st.st_size}"
    except OSError:
        return "missing"
