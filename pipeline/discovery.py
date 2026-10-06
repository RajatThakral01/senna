"""
pipeline/discovery.py

Contextual candidate discovery, pass 2 of the two-pass process.

Uses the video outline (pipeline/outline.py) plus the ORIGINAL transcript to
generate candidate passages. Each candidate cites source sentence IDs; all
timestamps are derived from word data and every reference is validated:

  - unknown sentence IDs are dropped; candidates left empty are rejected
  - start/end IDs are ordered; ranges derive from sentence word times
  - adjacent context is inspected for a complete passage (pronoun/answer/
    payoff checks in the prompt); the refine stage re-validates boundaries
  - semantic-search hits are recorded as related context only — a distant
    passage is never treated as chronological continuation
  - non-contiguous assembly is OFF by default (allow_noncontiguous=false);
    when enabled every join is validated for coherence and meaning

Transcript-only discovery always runs: quiet insights stay eligible.
"""
import json
import time

from logger import get_logger

log = get_logger("pipeline.discovery")


def _candidate_prompt(outline, sentences_by_id, window_ids, target_clips,
                      allow_noncontiguous):
    secs = []
    for s in outline.get("sections", []):
        secs.append(f'- "{s.get("title", "")}" ids {s.get("start_id")}-{s.get("end_id")}: '
                    f'{s.get("summary", "")[:220]}')
    lines = []
    for sid in window_ids:
        s = sentences_by_id.get(sid)
        if s:
            lines.append(f'[{sid}] ({s["start_time"]:.1f}s) {s["text"]}')
    join_rule = ("You may propose at most ONE non-contiguous assembly per candidate "
                 "as join_ids=[...] (a second short passage that directly completes "
                 "the first); every join is coherence-checked afterwards."
                 if allow_noncontiguous else
                 "Propose ONLY continuous excerpts (single start_id..end_id span). "
                 "Never join distant passages.")
    schema = ('{"candidates": [{"start_id": <int>, "end_id": <int>, '
                '"hook": "<opening spoken words>", "main_idea": "...", '
                '"payoff": "...", '
                '"required_context": "<what the viewer must already know, or empty>", '
                '"rationale": "...", "uncertainty": <0..1>'
                + (', "join_ids": [<int>, ...]' if allow_noncontiguous else '')
                + "}]}")
    return (
        "You select viral short-form clip candidates from a video transcript.\n"
        f"Video outline:\n{chr(10).join(secs) if secs else '(no outline sections)'}\n\n"
        "Transcript window (ID, start-seconds, text):\n"
        f"{chr(10).join(lines)}\n\n"
        f"Propose up to {target_clips} candidates as JSON: {schema}\n"
        f"Rules: cite ONLY IDs listed above. {join_rule} "
        "Keep each candidate's source span roughly 20-90 seconds long "
        "(use the start-seconds shown to judge length). "
        "Prefer passages that open strong, stand alone (no unexplained pronouns, "
        "no answer without its question), and end AFTER the story payoff or punchline. "
        "Quiet insightful moments are eligible alongside loud ones. "
        "Set uncertainty higher when context is ambiguous."
    )


def _llm_discover(prompt, cfg):
    try:
        from config import LLM_API_KEY, LLM_API_URL, LLM_MODEL
    except Exception:
        return None
    if not LLM_API_KEY:
        return None
    try:
        import re
        import requests
        r = requests.post(LLM_API_URL,
                          headers={"Authorization": f"Bearer {LLM_API_KEY}",
                                   "Content-Type": "application/json"},
                          json={"model": LLM_MODEL, "temperature": 0.3,
                                "max_tokens": 3000,
                                "messages": [{"role": "user", "content": prompt}]},
                          timeout=180)
        if r.status_code == 429:
            log.warning("discovery LLM rate-limited")
            return None
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"].strip()
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        return json.loads(m.group(0)) if m else None
    except Exception:
        log.exception("discovery LLM call failed")
        return None


def _derive_range(sids, sentences_by_id, pad_b=0.0, pad_a=0.0):
    """Timestamps for sentence IDs from word data. Returns (start, end) or None."""
    sents = [sentences_by_id[i] for i in sids if i in sentences_by_id]
    if not sents:
        return None
    return (min(s["start_time"] for s in sents) - pad_b,
            max(s["end_time"] for s in sents) + pad_a)


def validate_candidate(raw, sentences_by_id, allow_noncontiguous=False,
                       min_span_seconds=15.0, max_span_seconds=90.0):
    """Validate + normalize one raw LLM candidate. Returns dict or None (reject).

    Rejection reasons are explicit strings for the candidates table.
    Spans outside [min_span_seconds, max_span_seconds] are rejected — the
    model is instructed to propose clip-sized passages, and this enforces it.
    """
    try:
        start_id, end_id = int(raw["start_id"]), int(raw["end_id"])
    except (KeyError, TypeError, ValueError):
        return None, "missing/invalid start_id/end_id"
    if start_id > end_id:
        start_id, end_id = end_id, start_id
    sids = [i for i in range(start_id, end_id + 1) if i in sentences_by_id]
    if not sids:
        return None, "no known sentence IDs in span"
    # cap span length to avoid whole-video candidates (refine enforces duration too)
    if len(sids) > 120:
        return None, "span too long (>120 sentences)"
    joins = []
    if allow_noncontiguous and raw.get("join_ids"):
        try:
            jids = [int(i) for i in raw["join_ids"]]
        except (TypeError, ValueError):
            return None, "invalid join_ids"
        jids = [i for i in jids if i in sentences_by_id and i not in sids]
        if not jids:
            return None, "join_ids reference unknown sentences"
        # joins must be short and near enough to stay coherent (<=60s gap)
        jr = _derive_range(jids, sentences_by_id)
        mr = _derive_range(sids, sentences_by_id)
        if jr and mr and min(abs(jr[0] - mr[1]), abs(mr[0] - jr[1])) > 60:
            return None, "join too distant (>60s gap)"
        joins = sorted(set(jids))
    rng = _derive_range(sids, sentences_by_id)
    if not rng or rng[1] <= rng[0]:
        return None, "degenerate time range"
    span = rng[1] - rng[0]
    if span < min_span_seconds:
        return None, f"span too short ({span:.1f}s < {min_span_seconds:.0f}s)"
    if span > max_span_seconds:
        return None, f"span too long ({span:.1f}s > {max_span_seconds:.0f}s)"
    ranges = [[round(rng[0], 3), round(rng[1], 3)]]
    all_ids = list(sids)
    if joins:
        jr = _derive_range(joins, sentences_by_id)
        ranges.append([round(jr[0], 3), round(jr[1], 3)])
        all_ids += joins
    try:
        uncertainty = float(raw.get("uncertainty", 0.5))
    except (TypeError, ValueError):
        uncertainty = 0.5
    cand = {
        "sentence_ids": all_ids,
        "source_ranges": sorted(ranges),
        "hook": str(raw.get("hook", ""))[:300],
        "main_idea": str(raw.get("main_idea", ""))[:600],
        "payoff": str(raw.get("payoff", ""))[:600],
        "required_context": str(raw.get("required_context", ""))[:600],
        "rationale": str(raw.get("rationale", ""))[:600],
        "uncertainty": min(1.0, max(0.0, uncertainty)),
        "joins": joins,
    }
    return cand, ""


def generate_candidates(outline, words, sentences, cfg=None, events=None):
    """Two-pass candidate generation. Returns list of validated candidate dicts.

    One LLM call over the whole transcript when it fits, else per outline
    section windows. Every candidate is ID-validated; invalid ones are dropped
    with a logged reason. Never raises on LLM failure (returns []).
    """
    cfg = cfg or {}
    d_cfg = cfg.get("discovery", {}) if isinstance(cfg, dict) else {}
    target = int(d_cfg.get("target_clips", 8) or 8)
    max_cand = int(d_cfg.get("max_candidates", 24) or 24)
    allow_nc = bool(d_cfg.get("allow_noncontiguous", False))
    per_call = int(d_cfg.get("sentences_per_call", 120) or 120)
    min_span = float(d_cfg.get("min_span_seconds", 15) or 0)
    max_span = float(d_cfg.get("max_span_seconds", 90) or 0)
    t0 = time.monotonic()
    sentences_by_id = {s["id"]: s for s in sentences}
    if not sentences:
        return []

    # window the transcript (by outline sections when available)
    windows = []
    if outline.get("sections"):
        for sec in outline["sections"]:
            ids = list(range(sec["start_id"], sec["end_id"] + 1))
            windows.append([i for i in ids if i in sentences_by_id])
    else:
        ids = sorted(sentences_by_id)
        windows = [ids[i:i + per_call] for i in range(0, len(ids), per_call)]
    # split oversized windows
    split = []
    for w in windows:
        split.extend([w[i:i + per_call] for i in range(0, len(w), per_call)] or [[]])
    windows = [w for w in split if w]
    log.info("discovery start windows=%d target=%d", len(windows), target)

    raw_cands, seen = [], 0
    for wi, window_ids in enumerate(windows):
        prompt = _candidate_prompt(outline, sentences_by_id, window_ids,
                                   target_clips=max(2, target // max(1, len(windows)) + 1),
                                   allow_noncontiguous=allow_nc)
        data = _llm_discover(prompt, cfg)
        if not data:
            continue
        for raw in data.get("candidates", []) or []:
            cand, reason = validate_candidate(
                raw, sentences_by_id, allow_nc, min_span, max_span)
            if cand is None:
                log.debug("discovery drop: %s raw=%s", reason, str(raw)[:160])
                continue
            cand["provenance"] = {"window": wi, "outline_method": outline.get("method")}
            raw_cands.append(cand)
            if len(raw_cands) >= max_cand:
                break
        time.sleep(2)  # polite delay for rate-limited tiers
        if len(raw_cands) >= max_cand:
            break
    log.info("discovery done candidates=%d elapsed=%.1fs",
             len(raw_cands), time.monotonic() - t0)
    for c in raw_cands:
        c.setdefault("source", "outline")
    return raw_cands


def persist_candidates(video_id, candidates, source="outline"):
    """Write validated candidates (status=proposed), replacing this source only."""
    from db.repositories import candidate_repo
    candidate_repo.clear_candidates(video_id, source=source)
    ids = []
    for c in candidates:
        cid = candidate_repo.insert_candidate(
            video_id, source, c["sentence_ids"], c["source_ranges"],
            hook=c.get("hook"), main_idea=c.get("main_idea"),
            payoff=c.get("payoff"), required_context=c.get("required_context"),
            rationale=c.get("rationale"), uncertainty=c.get("uncertainty"),
            scores={"provenance": c.get("provenance", {})},
            status="proposed")
        ids.append(cid)
    log.info("candidates persisted video=%.8s count=%d", video_id, len(ids))
    return ids
