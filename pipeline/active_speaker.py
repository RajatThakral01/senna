"""pipeline/active_speaker.py

Visual active-speaker detection + shot director for multi-person framing.

Who is talking is decided from the picture, gated by the audio:
  - mouth_motion(): per-face lip activity between two adjacent frames —
    motion inside the mouth ROI minus motion in the eye band, so head nods,
    turns and lighting changes don't read as speech. Uses YuNet mouth/nose
    landmarks when present, else the lower face box.
  - audio_envelope()/voiced: clip loudness at the sampling rate; speaker
    evidence only counts while someone is actually talking (pauses hold the
    current shot instead of cutting to whoever happens to move).
  - direct_shots(): Viterbi over states {each visible person, "group"} with
    a switch penalty, then a minimum shot length — a shot list of hard cuts
    between speakers (OpusClip style), "group" (stacked panels) when two
    people talk over each other.

Returns None whenever the evidence is too weak to trust (no visible mouth
motion, <2 people) so callers keep the Phase-A single-subject/stacked
behaviour instead of cutting on noise.
"""
import math
import subprocess

from logger import get_logger

log = get_logger("pipeline.active_speaker")

GROUP = "group"


# ── Lip motion ────────────────────────────────────────────────────────────────

def _rois(box):
    """(mouth_roi, eye_roi) as (x0, y0, x1, y1) in source pixels."""
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    fw, fh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    lm = box.get("landmarks")
    if lm and len(lm) >= 5:
        (rmx, rmy), (lmx, lmy), (_, ny) = lm[3], lm[4], lm[2]
        mw = abs(lmx - rmx) or fw * 0.35
        mcx, my = (rmx + lmx) / 2.0, (rmy + lmy) / 2.0
        mouth = (mcx - mw * 0.8, ny + 0.3 * (my - ny), mcx + mw * 0.8, my + mw * 0.6)
    else:
        mouth = (x1 + fw * 0.25, y1 + fh * 0.62, x2 - fw * 0.25, y1 + fh * 0.98)
    eyes = (x1 + fw * 0.15, y1 + fh * 0.18, x2 - fw * 0.15, y1 + fh * 0.45)
    return mouth, eyes


def _patch(gray, roi, size=(32, 24)):
    import cv2
    h, w = gray.shape[:2]
    x0, y0 = max(0, int(roi[0])), max(0, int(roi[1]))
    x1, y1 = min(w, int(math.ceil(roi[2]))), min(h, int(math.ceil(roi[3])))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return cv2.resize(gray[y0:y1, x0:x1], size, interpolation=cv2.INTER_AREA).astype("float32")


def _change(a, b):
    import numpy as np
    return float(np.abs(a - b).mean() / (0.5 * (a.std() + b.std()) + 4.0))


def mouth_motion(prev_gray, gray, box):
    """Lip activity of one face between two adjacent grayscale frames.

    >= 0; ~0 for a still or silently nodding face. None when the ROI is
    unusable (face at the frame edge, too small).
    """
    if prev_gray is None:
        return None
    mouth, eyes = _rois(box)
    m0, m1 = _patch(prev_gray, mouth), _patch(gray, mouth)
    if m0 is None or m1 is None:
        return None
    e0, e1 = _patch(prev_gray, eyes), _patch(gray, eyes)
    head = _change(e0, e1) if e0 is not None and e1 is not None else 0.0
    return max(0.0, _change(m0, m1) - 0.8 * head)


# ── Audio ─────────────────────────────────────────────────────────────────────

def audio_envelope(path, hop=0.1, sr=16000):
    """(times, rms_db) of the clip's audio at `hop` seconds, or None."""
    import numpy as np
    try:
        from config import ffmpeg_path
        r = subprocess.run(
            [ffmpeg_path(), "-v", "error", "-i", path, "-vn", "-ac", "1",
             "-ar", str(sr), "-f", "s16le", "-"],
            capture_output=True, timeout=300)
        if r.returncode != 0 or not r.stdout:
            return None
        pcm = np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception:
        log.debug("audio envelope failed", exc_info=True)
        return None
    n = max(1, int(sr * hop))
    frames = len(pcm) // n
    if frames < 2:
        return None
    rms = np.sqrt((pcm[:frames * n].reshape(frames, n) ** 2).mean(axis=1) + 1e-12)
    return (np.arange(frames) + 0.5) * hop, 20 * np.log10(rms + 1e-9)


def voiced_at(env, times, voice_db=10.0):
    """Bool per time: loudness at least voice_db above the clip's noise floor.

    No audio -> everything voiced (visual-only decision).
    """
    import numpy as np
    times = np.asarray(times, dtype=float)
    if env is None:
        return np.ones(len(times), dtype=bool)
    et, db = env
    floor = float(np.percentile(db, 10))
    if float(np.percentile(db, 90)) - floor < voice_db:  # flat: constant talk or silence
        return np.full(len(times), float(np.percentile(db, 90)) > -50.0)
    return np.interp(times, et, db) > floor + voice_db


# ── Director ──────────────────────────────────────────────────────────────────

def _nan_smooth(v, k):
    import numpy as np
    # np.convolve(mode="same") returns max(len(v), k) values: a scene shorter
    # than the window (fast-cut footage) must not grow the series
    k = min(int(k), len(v))
    if k < 2:
        return v
    ok = ~np.isnan(v)
    ker = np.ones(k)
    num = np.convolve(np.where(ok, v, 0.0), ker, mode="same")
    den = np.convolve(ok.astype(float), ker, mode="same")
    out = np.full(len(v), np.nan)
    out[den > 0] = num[den > 0] / den[den > 0]
    out[~ok & (den < k / 2)] = np.nan  # mostly-missing windows stay unknown
    return out


def speaker_scores(seg, track_ids, cfg=None):
    """Matrix [len(seg), len(track_ids)] of smoothed lip activity (NaN = face
    not visible), plus the per-sample voiced flags."""
    import numpy as np
    cfg = cfg or {}
    a = np.full((len(seg), len(track_ids)), np.nan)
    col = {tid: j for j, tid in enumerate(track_ids)}
    for k, s in enumerate(seg):
        for tr in s.get("tracks", []):
            j = col.get(tr["id"])
            if j is not None and tr.get("mouth") is not None:
                a[k, j] = tr["mouth"]
    interval = float(cfg.get("sample_interval", 0.1))
    win = max(1, int(round(float(cfg.get("speaker_window_seconds", 0.5)) / interval)))
    for j in range(a.shape[1]):
        a[:, j] = _nan_smooth(a[:, j], win)
    voiced = np.array([bool(s.get("voiced", True)) for s in seg])
    return a, voiced


def _viterbi(cost, switch):
    import numpy as np
    n, m = cost.shape
    acc = cost[0].copy()
    back = np.zeros((n, m), dtype=int)
    for k in range(1, n):
        best = int(np.argmin(acc))
        stay = acc
        move = acc[best] + switch
        back[k] = np.where(stay <= move, np.arange(m), best)
        acc = np.minimum(stay, move) + cost[k]
    path = np.empty(n, dtype=int)
    path[-1] = int(np.argmin(acc))
    for k in range(n - 1, 0, -1):
        path[k - 1] = back[k, path[k]]
    return path


def _runs(path):
    runs, start = [], 0
    for k in range(1, len(path) + 1):
        if k == len(path) or path[k] != path[start]:
            runs.append([start, k, int(path[start])])
            start = k
    return runs


def _enforce_min_shot(runs, cost, min_len):
    """Merge shots shorter than min_len samples into the better neighbour."""
    while len(runs) > 1:
        lens = [r[1] - r[0] for r in runs]
        i = min(range(len(runs)), key=lambda j: lens[j])
        if lens[i] >= min_len:
            break
        a, b, _ = runs[i]
        cands = [j for j in (i - 1, i + 1) if 0 <= j < len(runs)]
        j = min(cands, key=lambda j: cost[a:b, runs[j][2]].sum())
        runs[i][2] = runs[j][2]
        merged = []
        for r in runs:
            if merged and merged[-1][2] == r[2]:
                merged[-1][1] = r[1]
            else:
                merged.append(list(r))
        runs = merged
    return runs


def _snap_cuts(runs, seg, window):
    """Move each cut to the quietest sample within +-window samples.

    Editors cut on the breath between sentences, not mid-word; lip evidence
    alone places a cut up to a few hundred ms off. Needs per-sample "db";
    a cut never moves past the middle of its neighbouring shots.
    """
    import numpy as np
    win = int(round(window))
    if win < 1 or len(runs) < 2:
        return runs
    db = np.array([s.get("db", np.nan) for s in seg], dtype=float)
    if np.isnan(db).all():
        return runs
    for r in range(1, len(runs)):
        b = runs[r][0]
        lo = max(b - win, (runs[r - 1][0] + b) // 2 + 1)
        hi = min(b + win, (b + runs[r][1]) // 2)
        if hi <= lo:
            continue
        cand = db[lo:hi + 1]
        if np.isnan(cand).all():
            continue
        nb = lo + int(np.nanargmin(cand))
        runs[r - 1][1] = nb
        runs[r][0] = nb
    return runs


def direct_shots(seg, track_ids, cfg=None, prior=None):
    """Shot list for one scene, or None when speaker evidence is unreliable.

    seg: scene samples (t, tracks[with 'mouth'], voiced). track_ids: the
    people eligible for a shot (>=2; the caller filters out faces too small
    to read lips). prior: optional {track_id: 0..1} prominence; during pauses
    and weak evidence the director leans to the most prominent person.
    Returns [{t0, t1, state, score}] where state is a track id or GROUP;
    t0/t1 are sample times (t1 exclusive, the last shot ends at the scene end
    supplied by the caller).

    Evidence rules (Phase D): lip activity below speaker_min_activity counts
    as "not talking" (absolute floor, not just relative to the loudest face);
    the scene needs at least speaker_min_clear_seconds where one person
    clearly out-talks the rest, otherwise None (off-screen narration, chewing,
    crowds -> the caller frames the prominent subject or the group); the
    two-shot state exists only when exactly two people are eligible.
    """
    import numpy as np
    cfg = cfg or {}
    if len(track_ids) < 2 or len(seg) < 3:
        return None
    a, voiced = speaker_scores(seg, track_ids, cfg)
    vals = a[~np.isnan(a) & voiced[:, None]] if voiced.any() else a[~np.isnan(a)]
    if vals.size == 0:
        return None
    ref = float(np.percentile(vals, 90))
    floor = float(cfg.get("speaker_min_activity", 0.08))
    if ref < floor:
        log.info("active speaker: lip activity too weak (p90=%.3f); no director", ref)
        return None
    s = np.clip(a / ref, 0.0, 1.5)
    s = np.where(a < floor, 0.0, s)                 # below the absolute floor: not talking
    s_f = np.where(np.isnan(s), -0.25, s)          # off-screen/unknown: weak evidence against
    srt = np.sort(s_f, axis=1)
    clear = voiced & (srt[:, -1] >= 0.6) & (srt[:, -1] - srt[:, -2] >= 0.3)
    interval = float(cfg.get("sample_interval", 0.1))
    if clear.sum() * interval < float(cfg.get("speaker_min_clear_seconds", 0.3)):
        log.info("active speaker: no clear speaker in scene (%.1fs clear); no director",
                 clear.sum() * interval)
        return None
    allow_group = bool(cfg.get("speaker_allow_group", True)) and len(track_ids) == 2
    m = len(track_ids) + (1 if allow_group else 0)
    cost = np.zeros((len(seg), m))
    # evidence relative to the other people: the most active mouth wins
    for j in range(len(track_ids)):
        others = np.delete(s_f, j, axis=1).max(axis=1)
        cost[:, j] = -(s_f[:, j] - others)
    if allow_group:
        top2 = np.sort(s_f, axis=1)[:, -2:]
        both = top2.min(axis=1)                     # second-most active mouth
        # two-shot wins once the SECOND mouth is clearly active too; below
        # the threshold it is penalised (a nodding listener stays a listener)
        thr = float(cfg.get("speaker_group_threshold", 0.5))
        cost[:, -1] = float(cfg.get("speaker_group_bias", 0.2)) - 3.0 * (both - thr)
    cost[~voiced] = 0.0                              # pauses: no evidence, hold the shot
    if allow_group:
        cost[~voiced, -1] = 0.05
    if prior:
        pw = float(cfg.get("speaker_prior_weight", 0.1))
        for j, tid in enumerate(track_ids):
            cost[:, j] += pw * (1.0 - float(prior.get(tid, 0.0)))
    path = _viterbi(cost, float(cfg.get("speaker_switch_cost", 3.0)))
    min_len = max(1, int(round(float(cfg.get("min_shot_seconds", 1.6)) / interval)))
    runs = _enforce_min_shot(_runs(path), cost, min_len)
    runs = _snap_cuts(runs, seg, float(cfg.get("cut_snap_seconds", 0.4)) / interval)
    states = list(track_ids) + ([GROUP] if allow_group else [])
    shots = []
    for a0, b0, st in runs:
        shots.append({"t0": seg[a0]["t"],
                      "t1": seg[b0]["t"] if b0 < len(seg) else None,
                      "state": states[st],
                      "score": round(float(-cost[a0:b0, st].mean()), 3)})
    log.info("active speaker: %d shots %s", len(shots),
             [(round(x["t0"], 1), x["state"]) for x in shots])
    return shots
