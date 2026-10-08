"""
pipeline/audio_events.py

Checkpointed audio-event discovery (energy spikes before any output processing).

Runs on the source audio (downloads/audio.wav as extracted pre-normalization)
and NEVER shifts timestamps through later edits — all events keep source time.

Detector (method=rms v1, no heavy deps):
  - short-window RMS/log-energy (window_ms, default 200ms)
  - rolling LOCAL baseline (median over +/- baseline_seconds, robust to
    recording differences) with a floor for silence/near-zero energy
  - threshold_db above baseline + min_duration_seconds to trigger
  - merge nearby detections (merge_gap_seconds), suppress isolated clicks
    (< min_duration) and cap sustained high-energy regions (max_event_seconds,
    truncated with lowered confidence)
  - bounded by source duration; returns [] (never raises) on unreadable audio

Energy alone never proves highlight quality: each event only PROMPTS
inspection of surrounding transcript (pre_seconds before for setup, e.g. the
joke before laughter; post_seconds after). A candidate is created only when
speech coverage supports content, with transcript-aligned (sentence) bounds —
never spike bounds.

Optional classifier adapter (pipeline/audio_classifier.py): labels like
laughter/applause/music via a documented local model when installed;
disabled by default with explicit status, uncertainty preserved.
"""
import time

from logger import get_logger

log = get_logger("pipeline.audio_events")


def ensure_source_audio(raw_video_path="input/raw_video.mp4",
                        audio_path="downloads/audio.wav"):
    """Return path to pre-normalization source audio, extracting if needed."""
    import os
    import subprocess
    if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
        return audio_path
    from config import ffmpeg_path
    os.makedirs(os.path.dirname(audio_path) or ".", exist_ok=True)
    r = subprocess.run([ffmpeg_path(), "-y", "-i", raw_video_path,
                        "-vn", "-acodec", "pcm_s16le", "-ar", "16000",
                        "-ac", "1", audio_path],
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(audio_path):
        raise RuntimeError(f"audio extraction failed: {(r.stderr or '')[-300:]}")
    return audio_path


def detect_events(audio, sr, cfg=None):
    """Pure detection over mono samples. Returns event dicts (source seconds).

    audio: 1-D numpy float array. sr: sample rate. All thresholds from cfg.
    """
    import numpy as np
    cfg = cfg or {}
    win_ms = float(cfg.get("window_ms", 200) or 200)
    baseline_sec = float(cfg.get("baseline_seconds", 10) or 10)
    threshold_db = float(cfg.get("threshold_db", 8.0))
    min_dur = float(cfg.get("min_duration_seconds", 0.4) or 0.4)
    max_dur = float(cfg.get("max_event_seconds", 8.0) or 8.0)
    merge_gap = float(cfg.get("merge_gap_seconds", 1.0) or 1.0)
    floor = float(cfg.get("silence_floor", 1e-4))

    if audio is None or len(audio) == 0:
        return []
    audio = np.asarray(audio, dtype=np.float64).ravel()
    win = max(1, int(sr * win_ms / 1000))
    n = len(audio) // win
    if n == 0:
        return []
    frames = audio[:n * win].reshape(n, win)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(np.maximum(rms, floor))
    # rolling median baseline over +/- baseline_sec
    rad = max(1, int(baseline_sec * 1000 / win_ms))
    base = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - rad), min(n, i + rad + 1)
        base[i] = np.median(db[lo:hi])
    # silence safety: where baseline sits at the floor, require absolute level
    active = (db - base) >= threshold_db
    active &= db > (20 * np.log10(floor) + 3.0)

    # runs of active frames -> raw detections
    runs, i = [], 0
    while i < n:
        if active[i]:
            j = i
            while j + 1 < n and active[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    dur = len(audio) / sr
    events = []
    for a, b in runs:
        s, e = a * win / sr, min((b + 1) * win / sr, dur)
        if e - s < min_dur:
            continue  # isolated click / blip
        peak_rel = int(a + np.argmax(rms[a:b + 1]))
        peak_t = min((peak_rel + 0.5) * win / sr, dur)
        increase = float(db[peak_rel] - base[peak_rel])
        conf = min(1.0, max(0.05, (increase - threshold_db + 6.0) / 12.0))
        truncated = False
        if e - s > max_dur:
            e = s + max_dur  # cap sustained regions; lower confidence
            conf *= 0.6
            truncated = True
        events.append({"start_time": round(s, 3), "end_time": round(e, 3),
                       "peak_time": round(peak_t, 3),
                       "energy_increase": round(increase, 2),
                       "confidence": round(conf, 3),
                       "method": "rms", "version": "v1",
                       "truncated": truncated})
    # merge nearby
    merged = []
    for ev in events:
        if merged and ev["start_time"] - merged[-1]["end_time"] <= merge_gap:
            last = merged[-1]
            last["end_time"] = ev["end_time"]
            if ev["peak_time"] and (ev["energy_increase"] > last["energy_increase"]):
                last["peak_time"] = ev["peak_time"]
                last["energy_increase"] = ev["energy_increase"]
                last["confidence"] = max(last["confidence"], ev["confidence"])
        else:
            merged.append(dict(ev))
    for ev in merged:
        ev["start_time"] = max(0.0, ev["start_time"])
        ev["end_time"] = min(dur, ev["end_time"])
    log.debug("audio events detected=%d (runs=%d)", len(merged), len(runs))
    return merged


def detect_audio_events(audio_path, cfg=None):
    """Load audio + detect. Returns (events, fingerprint). Never raises."""
    import librosa
    from pipeline.fingerprints import audio_events_fingerprint
    cfg = cfg or {}
    t0 = time.monotonic()
    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        duration = len(y) / sr
    except Exception:
        log.exception("audio load failed path=%s", audio_path)
        return [], audio_events_fingerprint("unreadable", cfg)
    audio_sig = f"{audio_path}:{round(duration, 1)}:{len(y)}"
    fp = audio_events_fingerprint(audio_sig, cfg)
    events = detect_events(y, sr, cfg)
    for ev in events:
        ev["config_fingerprint"] = fp
    log.info("audio_events done path=%.60s events=%d elapsed=%.1fs",
             audio_path, len(events), time.monotonic() - t0)
    return events, fp


def speech_coverage(start, end, words):
    """Fraction of [start, end] covered by word times (content-support gate)."""
    if end <= start:
        return 0.0
    covered = 0.0
    for w in words:
        try:
            ws, we = float(w["start"]), float(w["end"])
        except (KeyError, TypeError, ValueError):
            continue
        lo, hi = max(ws, start), min(we, end)
        if hi > lo:
            covered += hi - lo
    return covered / (end - start)


def propose_from_events(events, words, sentences, cfg=None):
    """Deterministic event-cued candidates (transcript-aligned, gated).

    For each event: window [peak - pre, peak + post] snapped to sentence
    bounds; kept only when speech coverage >= min_speech_coverage.
    Returns candidate dicts (source=audio_event) WITHOUT LLM text; an
    optional batched LLM pass (enrich_event_candidates) adds hook/idea/payoff.
    """
    cfg = cfg or {}
    a_cfg = cfg.get("audio_events", {})
    pre = float(a_cfg.get("pre_seconds", 25) or 25)
    post = float(a_cfg.get("post_seconds", 10) or 10)
    min_cov = float(a_cfg.get("min_speech_coverage", 0.3))
    min_span = float(cfg.get("discovery", {}).get("min_span_seconds", 15) or 0)
    max_span = float(cfg.get("discovery", {}).get("max_span_seconds", 90) or 0)
    by_id = {s["id"]: s for s in sentences}
    out = []
    for ev in events:
        peak = ev.get("peak_time", (ev["start_time"] + ev["end_time"]) / 2)
        ws, we = peak - pre, peak + post
        # snap to sentence bounds (transcript-aligned, not spike bounds)
        sids = [s["id"] for s in sentences
                if s["end_time"] >= ws and s["start_time"] <= we]
        if not sids:
            continue
        a = min(by_id[i]["start_time"] for i in sids)
        b = max(by_id[i]["end_time"] for i in sids)
        cov = speech_coverage(a, b, words)
        if cov < min_cov:
            log.debug("event cue dropped: speech coverage %.2f < %.2f",
                      cov, min_cov)
            continue
        span = b - a
        if (min_span and span < min_span) or (max_span and span > max_span):
            continue
        out.append({
            "source": "audio_event",
            "sentence_ids": sorted(sids),
            "source_ranges": [[round(a, 3), round(b, 3)]],
            "hook": "", "main_idea": "", "payoff": "",
            "required_context": "",
            "rationale": (f"audio spike +{ev.get('energy_increase')}dB @ {peak:.1f}s; "
                          f"speech coverage {cov:.2f}"),
            "uncertainty": round(1.0 - 0.5 * float(ev.get("confidence", 0.5)), 3),
            "joins": [],
            "provenance": {"event_peak": peak,
                           "energy_increase": ev.get("energy_increase"),
                           "label": ev.get("label")},
        })
    log.info("event cues proposed=%d/%d", len(out), len(events))
    return out


def enrich_event_candidates(candidates, cfg=None):
    """One batched LLM call adding hook/main_idea/payoff to event cues.

    No-op (with clear log) when no key or no candidates. Never raises.
    """
    if not candidates:
        return candidates
    try:
        from config import LLM_API_KEY, LLM_API_URL, LLM_MODEL
    except Exception:
        return candidates
    from pipeline.llm_client import pool_usable as _pool_usable
    if not _pool_usable():
        log.info("event enrich skipped (no LLM key)")
        return candidates
    import json as _json
    import re
    from pipeline.llm_client import post_chat, SLOT_FOR_STAGE
    items = []
    for i, c in enumerate(candidates):
        items.append(f"[{i}] range {c['source_ranges'][0]} "
                     f"peak +{c['provenance'].get('energy_increase')}dB "
                     f"label={c['provenance'].get('label')} rationale={c['rationale']}")
    prompt = (
        "For each audio-spike-cued video passage below, write hook/main_idea/payoff "
        "as JSON. The spike marks a reaction; describe the LIKELY setup+payoff shape "
        "a clip here must capture (do not invent transcript text).\n"
        f"{chr(10).join(items)}\n\n"
        'Return ONLY JSON: {"0": {"hook": "...", "main_idea": "...", "payoff": "..."}, ...}'
    )
    try:
        r = post_chat({"model": LLM_MODEL, "temperature": 0.3,
                       "max_tokens": 2000,
                       "messages": [{"role": "user", "content": prompt}]},
                      timeout=120, slot=SLOT_FOR_STAGE["enrich"],
                      purpose="audio-enrich")
        raw = r.json()["choices"][0]["message"]["content"].strip()
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        data = _json.loads(m.group(0)) if m else {}
        for i, c in enumerate(candidates):
            d = data.get(str(i), {})
            for k in ("hook", "main_idea", "payoff"):
                if d.get(k):
                    c[k] = str(d[k])[:400]
        log.info("event enrich done n=%d", len(candidates))
    except Exception:
        log.exception("event enrich failed")
    return candidates


def persist_events(video_id, events):
    from db.repositories import event_repo
    event_repo.clear_events(video_id)
    for ev in events:
        event_repo.insert_event(
            video_id, ev["start_time"], ev["end_time"],
            peak_time=ev.get("peak_time"),
            energy_increase=ev.get("energy_increase"),
            confidence=ev.get("confidence"), method=ev.get("method", "rms"),
            version=ev.get("version", "v1"), label=ev.get("label"),
            label_confidence=ev.get("label_confidence"),
            config_fingerprint=ev.get("config_fingerprint", ""))
    log.info("events persisted video=%.8s count=%d", video_id, len(events))
