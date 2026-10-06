"""pipeline/diarization.py

Optional speaker diarization wrapper (v1: explicit mapping only).

Active-speaker switching is OFF by default. This module provides:
  - run_diarization(): pyannote-backed when HF_TOKEN + pyannote are present,
    otherwise raises a clear, actionable error (never silent wrong mapping).
  - get_active_speaker(s): pure helpers adapted from the MIT-licensed
    reference (NaufalRizqullah/opensource-clipping, clipping/diarization.py).
  - load_segments_for_clip(): clip diarization segments to a time window.

Rule enforced by callers: audio diarization ALONE never identifies the
corresponding face. framing.map_speakers_to_faces() requires an explicit
speaker_map (config framing.speaker_map, e.g. fixed-seat podcasts) or falls
back to stacked_split / branded_fit.
"""
from logger import get_logger

log = get_logger("pipeline.diarization")


def get_active_speaker(diarization_data, timestamp):
    if not diarization_data:
        return None
    for seg in diarization_data:
        if seg.get("start", 0) <= timestamp <= seg.get("end", 0):
            return seg.get("speaker")
    return None


def get_active_speakers(diarization_data, timestamp):
    if not diarization_data:
        return []
    active = []
    for seg in diarization_data:
        if seg.get("start", 0) <= timestamp <= seg.get("end", 0):
            if seg.get("speaker") not in active:
                active.append(seg.get("speaker"))
    return active


def load_segments_for_clip(diarization_data, start, end):
    if not diarization_data:
        return []
    out = []
    for seg in diarization_data:
        s, e = max(seg.get("start", 0), start), min(seg.get("end", 0), end)
        if e > s:
            out.append({"speaker": seg.get("speaker"), "start": s, "end": e})
    return out


def run_diarization(audio_path, num_speakers="auto"):
    """Run pyannote diarization if available; else raise with instructions."""
    import os
    token = os.getenv("HF_TOKEN", "")
    if not token:
        raise RuntimeError("Diarization requires HF_TOKEN in .env (pyannote model agreement). "
                           "Falling back to visual-only stacked/branded layouts.")
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        raise RuntimeError("Diarization requires `pip install pyannote.audio`. "
                           "Falling back to visual-only stacked/branded layouts.")
    log.info("diarization start audio=%.60s speakers=%s", audio_path, num_speakers)
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                            token=token)
    except TypeError:
        from pyannote.audio import Pipeline as _P
        pipeline = _P.from_pretrained("pyannote/speaker-diarization-3.1",
                                      use_auth_token=token)
    try:
        import torch
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
    except Exception:
        pass
    diar = pipeline(audio_path,
                    num_speakers=None if str(num_speakers).lower() == "auto"
                    else int(num_speakers))
    if hasattr(diar, "speaker_diarization"):
        diar = diar.speaker_diarization
    elif hasattr(diar, "annotation"):
        diar = diar.annotation
    segs = []
    for turn, _, speaker in diar.itertracks(yield_label=True):
        segs.append({"speaker": speaker, "start": round(turn.start, 3),
                     "end": round(turn.end, 3)})
    # merge adjacent same-speaker (gap < 0.5s, per reference)
    merged = []
    for s in segs:
        if merged and merged[-1]["speaker"] == s["speaker"] \
                and s["start"] - merged[-1]["end"] < 0.5:
            merged[-1]["end"] = s["end"]
        else:
            merged.append(dict(s))
    log.info("diarization done segments=%d speakers=%d", len(merged),
             len(set(s["speaker"] for s in merged)))
    return merged
