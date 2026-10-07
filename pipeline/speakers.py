"""
pipeline/speakers.py

Active-speaker switching adapter (Phase 7, OPTIONAL, off by default).

Rules (never violated):
  - diarization + explicit/reliable voice-to-face mapping required;
    audio diarization alone never identifies a face.
  - manual speaker_map ({SPEAKER_00: left, ...} for fixed seats) supported.
  - the largest face is NOT assumed to be the speaker.
  - a single visible face during speech does NOT imply mapping.
  - switches only on stable speaking turns (>= min_turn_seconds).
  - uncertain -> no switches (caller keeps stacked_split / branded_fit).
  - frozen listener frames are never presented as live reaction.

Automatic voice-to-face matching is NOT implemented: no multi-person footage
was available to evaluate it, and an unevaluated matcher would produce
untrustworthy switches. auto_match therefore reports status
"unavailable_unevaluated" instead of fake mappings. The adapter surface
(match_voices_to_faces) is stable so a future evaluated matcher plugs in.
"""
from logger import get_logger

log = get_logger("pipeline.speakers")


def stable_turns(diarization, min_turn_seconds=1.5):
    """Merge micro-turns; keep turns >= min length. Pure function."""
    if not diarization:
        return []
    # sort + merge same-speaker overlaps/gaps < 0.5s
    segs = sorted(diarization, key=lambda s: (s.get("start", 0), s.get("end", 0)))
    merged = []
    for s in segs:
        if (merged and merged[-1].get("speaker") == s.get("speaker")
                and s.get("start", 0) - merged[-1].get("end", 0) < 0.5):
            merged[-1]["end"] = max(merged[-1]["end"], s.get("end", 0))
        else:
            merged.append(dict(s))
    return [s for s in merged
            if s.get("end", 0) - s.get("start", 0) >= min_turn_seconds]


def match_voices_to_faces(diarization, framing_analysis, speaker_map=None,
                          cfg=None):
    """Voice-to-face mapping behind a configurable adapter.

    Returns (mapping, status) where mapping = {speaker: track_id} (possibly
    {}) and status explains the outcome:
      - mapped_explicit: manual speaker_map applied positionally
      - unavailable_no_diarization / unavailable_no_faces
      - unavailable_unevaluated: auto matching requested but no evaluated
        matcher exists (never guess)
      - disabled: adapter off
    """
    cfg = cfg or {}
    s_cfg = cfg.get("speakers", {}) if isinstance(cfg, dict) else {}
    if not s_cfg.get("enabled", False):
        return {}, {"status": "disabled",
                    "reason": "speakers.enabled is false"}
    if not diarization:
        return {}, {"status": "unavailable_no_diarization",
                    "reason": "no diarization segments for this clip"}
    tracks = set()
    for s in (framing_analysis or {}).get("samples", []):
        for tr in s.get("tracks", []):
            tracks.add(tr.get("id"))
    if len(tracks) < 2:
        return {}, {"status": "unavailable_no_faces",
                    "reason": f"need >=2 face tracks, saw {len(tracks)} "
                              "(single visible face never implies mapping)"}
    if speaker_map:
        from pipeline.framing import map_speakers_to_faces
        mapping = map_speakers_to_faces(diarization, framing_analysis,
                                        speaker_map)
        if mapping:
            return mapping, {"status": "mapped_explicit",
                             "reason": f"manual speaker_map {speaker_map}"}
        return {}, {"status": "unavailable_uncertain_map",
                    "reason": "explicit map given but tracks unclear"}
    if s_cfg.get("auto_match", False):
        return {}, {"status": "unavailable_unevaluated",
                    "reason": "auto voice-to-face matching has no evaluated "
                              "implementation; not guessing"}
    return {}, {"status": "unavailable_no_map",
                "reason": "no speaker_map configured; "
                          "use stacked_split/branded_fit"}


def plan_switches(diarization, mapping, start=0.0, end=None,
                  min_turn_seconds=1.5, hold_seconds=2.0):
    """Switch timeline from stable turns + mapping. Returns [] when uncertain.

    Each switch: {t, speaker, track_id, confidence}. Holds the current
    speaker for hold_seconds to avoid flicker; overlapping speech keeps the
    current speaker (no simultaneous-switch).
    """
    if not mapping or not diarization:
        return []
    turns = stable_turns(diarization, min_turn_seconds)
    turns = [t for t in turns if t.get("end", 0) > start
             and (end is None or t.get("start", 0) < end)]
    switches, current, last_t = [], None, start - hold_seconds - 1
    for t in turns:
        spk = t.get("speaker")
        if spk not in mapping:
            continue
        if spk == current:
            continue
        # overlapping multi-speaker: hold current
        overlapping = [u for u in turns
                       if u is not t and u.get("speaker") != spk
                       and u.get("start", 0) < t.get("end", 0)
                       and u.get("end", 0) > t.get("start", 0)]
        if overlapping and current is not None:
            continue
        if t.get("start", 0) - last_t < hold_seconds and current is not None:
            continue
        switches.append({"t": round(max(start, t["start"]), 3), "speaker": spk,
                         "track_id": mapping[spk], "confidence": 0.8})
        current, last_t = spk, t["start"]
    return switches
