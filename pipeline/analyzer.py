"""
pipeline/analyzer.py

Analyzes transcript chunks to identify viral short-form video clips.
"""

import json
import time
import requests
from logger import get_logger

from config import LLM_API_KEY, LLM_MODEL, LLM_API_URL
from db.repositories import chunk_repo, clip_repo
from pipeline.embedder import embed_text

log = get_logger("pipeline.analyzer")

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def to_sec(t):
    if len(t.split(':')) == 2: t = '00:' + t
    p = t.split(':')
    return int(p[0])*3600 + int(p[1])*60 + float(p[2])

def extract_json_from_response(raw_text):
    import re

    outside = re.sub(r'ground.*?ground', '', raw_text, flags=re.DOTALL).strip()
    outside = re.sub(r'<think>.*?</think>', '', outside, flags=re.DOTALL).strip()

    cleaned = outside
    if '```' in cleaned:
        parts = cleaned.split('```')
        for part in parts:
            part = part.strip()
            if part.startswith('json'):
                part = part[4:]
            part = part.strip()
            if part.startswith('[') or part.startswith('{'):
                cleaned = part
                break

    start = cleaned.find('[')
    end = cleaned.rfind(']')
    if start != -1 and end != -1 and end > start:
        return cleaned[start:end+1]

    start = cleaned.find('{')
    end = cleaned.rfind('}')
    if start != -1 and end != -1 and end > start:
        return '[' + cleaned[start:end+1] + ']'

    think_match = re.search(r'<think>(.*?)</think>', raw_text, re.DOTALL)
    if think_match:
        inside = think_match.group(1)

        if '```' in inside:
            parts = inside.split('```')
            for part in parts:
                part = part.strip()
                if part.startswith('json'):
                    part = part[4:]
                part = part.strip()
                if part.startswith('[') or part.startswith('{'):
                    inside = part
                    break

        start = inside.rfind('[')
        end = inside.rfind(']')
        if start != -1 and end != -1 and end > start:
            log.debug("JSON found inside think block")
            return inside[start:end+1]

    log.error("JSON extraction failed raw_len=%d raw=%.500s", len(raw_text), raw_text)
    raise ValueError("JSON extraction failed")


def _extract_clips_from_chunk(chunk: dict, config: dict) -> list[dict]:
    """
    Send a single chunk's text to the LLM to extract 0-2 viral moments.
    """
    prompt = f"""You are a viral short-form video expert for
TikTok, Instagram Reels, and YouTube Shorts.

Here is a segment of a video transcript from {format_time(chunk['start_time'])} to {format_time(chunk['end_time'])}:
{chunk['text']}

Find the 0 to 2 best moments for viral short clips in this segment.
If there are no strong moments, return an empty array [].
Only include a clip if the moment is genuinely strong.

WHAT MAKES A STRONG CLIP:
- Something unexpected or surprising happens
- A peak emotional moment (triumph, failure, fear, joy)
- A question gets answered dramatically
- Something physical and visual happens
- A relatable human moment anyone can connect with

CLIP RULES:
- Each clip MUST be between 20 and 90 seconds long.
- CRITICAL: Clips must not start or end abruptly.
- Must start at a strong hook sentence (natural starting point).
- Must end at a completed sentence or natural stopping point.
- No overlapping clips

GOOD hook examples:
- "Wait, hold on. See something?" (curiosity)
- "It floats!" (triumph)
- "Oh no. The paddle broke." (crisis)
- "I can't go anymore." (emotional breaking point)
- "We're in the open ocean. It wants to take us out." (danger)

BAD hooks (never start a clip here):
- "So...", "And...", "Well...", "You know..."
- Generic: "The moment of truth", "Here we go"
- Narration: "We spent hours...", "We were working..."

Return ONLY valid JSON array, no explanation:
[
    {{
        "start_time": "HH:MM:SS",
        "end_time": "HH:MM:SS",
        "duration_seconds": 60,
        "hook": "exact first spoken words",
        "reason": "why this will go viral",
        "suggested_title": "punchy caption",
        "suggested_hashtags": "#tag1 #tag2 #tag3"
    }}
]"""

    import time

    log.debug("extract chunk_idx=%s range=%.1f-%.1f chars=%d",
              chunk.get("chunk_index"), chunk.get("start_time", 0),
              chunk.get("end_time", 0), len(chunk.get("text", "")))
    max_retries = 3
    response = None
    for attempt in range(max_retries):
        try:
            from pipeline.llm_client import post_chat, SLOT_FOR_STAGE
            response = post_chat(
                {
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2000,
                    "temperature": 0.3
                },
                timeout=120, slot=SLOT_FOR_STAGE["analyzer"],
                purpose="analyzer")
            break  # Success
        except requests.exceptions.RequestException as e:
            resp = getattr(e, "response", None)
            if resp is not None and resp.status_code in (400, 401, 403, 404):
                log.error("analyzer API %s (key/model dead), skipping chunk",
                          resp.status_code)
                return []
            log.warning("analyzer API failed attempt %d/%d: %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                log.error("analyzer API exhausted retries, returning []")
                return []

    raw = response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
    if not raw:
        log.warning("analyzer empty LLM response")
        return []

    try:
        raw = extract_json_from_response(raw)
        clips = json.loads(raw) if raw != '[]' else []
    except ValueError:
        log.exception("analyzer JSON parse failed")
        return []
    log.debug("analyzer LLM returned %d raw clips", len(clips))

    MIN_DURATION = 15

    valid_clips = []
    for clip in clips:
        if 'start_time' not in clip or 'end_time' not in clip: continue
        try:
            clip['start_time'] = str(clip['start_time'])
            clip['end_time'] = str(clip['end_time'])
            
            dur = to_sec(clip['end_time']) - to_sec(clip['start_time'])
            if dur >= MIN_DURATION:
                clip['duration_seconds'] = dur
                valid_clips.append(clip)
            else:
                log.debug("skip short clip %.0fs < %ds hook=%.40s", dur, MIN_DURATION, clip.get("hook", ""))
        except Exception as e:
            log.warning("skip malformed clip timestamp: %s clip=%s", e, {k: clip.get(k) for k in ("start_time", "end_time")})

    log.debug("extract done valid=%d/%d", len(valid_clips), len(clips))
    return valid_clips


def analyze(video_id: str, chunks: list[dict], config: dict,
            campaign_text: str = "") -> list[dict]:
    """Two-pass contextual analysis.

    Pass 1 (outline): whole-video structure from the transcript (cached in
    the outlines table + <workspace>/outline.json).
    Pass 2 (discovery): candidates with source sentence IDs from the outline
    + original transcript; timestamps derived from word data and validated.

    Falls back to legacy per-chunk extraction when discovery yields nothing
    and discovery.legacy_fallback is enabled (default true).
    Returns the list of inserted clip dicts.
    """
    from pipeline.boundaries import load_words, build_sentences
    from pipeline.outline import build_outline, persist_outline
    from pipeline.discovery import generate_candidates, persist_candidates
    from pipeline.fingerprints import (outline_fingerprint, discovery_fingerprint,
                                       transcript_signature)
    from db.repositories import outline_repo, candidate_repo, run_repo
    from config import get_config as _get_config, LLM_MODEL as _LLM_MODEL

    t0 = time.monotonic()
    cfg = config or _get_config()
    d_cfg = cfg.get("discovery", {})
    target = int(d_cfg.get("target_clips", 8) or 8)
    log.info("analyze start video=%.8s chunks=%d target=%d", video_id, len(chunks), target)

    try:
        from pipeline.workspace import current as _ws
        words = load_words(_ws().transcript)
        sentences = build_sentences(words)
    except Exception:
        log.exception("analyze transcript load failed, legacy fallback")
        words, sentences = [], []

    clips = []
    outline = None
    if words and sentences:
        tsig = transcript_signature(_ws().transcript)
        # Pass 1: reuse cached outline when the fingerprint matches
        outline = None
        try:
            ofp = outline_fingerprint(tsig, _LLM_MODEL, cfg)
            if run_repo.fingerprint_matches(video_id, "outline", ofp):
                secs = outline_repo.get_outlines(video_id, "section")
                if secs:
                    outline = {"method": "cached", "sections": [
                        {"title": s.get("title"), "start_id": s["content"].get("start_id"),
                         "end_id": s["content"].get("end_id"),
                         "start_time": s.get("start_time"), "end_time": s.get("end_time"),
                         "summary": s["content"].get("summary", ""),
                         "topics": s["content"].get("topics", []),
                         "qa": s["content"].get("qa", []),
                         "stories": s["content"].get("stories", []),
                         "key_ids": s["content"].get("key_ids", [])}
                        for s in secs], "video": {}}
                    log.info("analyze reusing cached outline sections=%d", len(secs))
        except Exception:
            log.exception("outline cache read failed")
            outline = None
        if outline is None:
            outline = build_outline(words, sentences, chunks, cfg=cfg)
            try:
                persist_outline(video_id, outline)
                run_repo.set_fingerprint(video_id, "outline",
                                         outline_fingerprint(tsig, _LLM_MODEL, cfg))
            except Exception:
                log.exception("outline persist failed")
        # Pass 2: candidates
        try:
            dfp = discovery_fingerprint(
                tsig,
                outline_fingerprint(tsig, _LLM_MODEL, cfg), _LLM_MODEL, cfg)
            candidates = generate_candidates(outline, words, sentences, cfg)
            ids = persist_candidates(video_id, candidates, source="outline")
            for c, i in zip(candidates, ids):
                c["db_id"] = i
            run_repo.set_fingerprint(video_id, "discovery", dfp)
        except Exception:
            log.exception("discovery failed")
            candidates = []
        clips = _insert_clips_from_candidates(
            video_id, _fusion_pool(video_id, candidates), chunks, target,
            campaign_text=campaign_text, cfg=cfg)
        log.info("analyze two-pass done video=%.8s clips=%d", video_id, len(clips))

    if not clips and d_cfg.get("legacy_fallback", True):
        log.info("analyze falling back to legacy per-chunk extraction")
        clips = _analyze_legacy(video_id, chunks, config)
    if not clips:
        log.warning("analyze: LLM unreachable, deterministic section fallback")
        clips = _deterministic_section_clips(
            video_id, outline, words, sentences, chunks, target,
            campaign_text=campaign_text, cfg=cfg)
    elif len(clips) < target and (outline or {}).get("method") != "llm":
        # LLM was unreachable (non-llm outline) and two-pass fell short:
        # top up with guaranteed section spans so the run still yields a
        # useful set. Existing clips win overlaps (excluded below).
        log.warning("analyze: topping up %d -> %d with deterministic spans",
                    len(clips), target)
        extra = _deterministic_section_clips(
            video_id, outline, words, sentences, chunks,
            target - len(clips),
            campaign_text=campaign_text, cfg=cfg,
            exclude_ranges=[(c["start_time"], c["end_time"]) for c in clips
                            if c.get("start_time") is not None],
            start_clip_number=len(clips) + 1)
        clips.extend(extra)
    _write_clips_analysis_json(clips)
    return clips


def _fusion_pool(video_id, outline_candidates):
    """Full fusion pool: fresh outline candidates + proposed audio-event cues.

    Audio cues were persisted by the audio_events stage; outline candidates
    were just persisted above. Both carry sentence_ids/source_ranges, so the
    same validation + scoring applies. DB rows expose db_id for status
    updates.
    """
    from db.repositories import candidate_repo
    pool = list(outline_candidates or [])
    try:
        for row in candidate_repo.get_candidates(video_id, status="proposed"):
            if row.get("source") == "audio_event":
                row["db_id"] = row["id"]
                pool.append(row)
    except Exception:
        log.debug("audio cue pool read skipped", exc_info=True)
    log.info("fusion pool outline=%d total=%d",
             len(outline_candidates or []), len(pool))
    return pool


def _split_ids_by_time(sids, sentences_by_id, target_span=55.0,
                       max_span=90.0, min_span=15.0):
    """Greedy sentence-ID groups bounded by time. Pure function (testable).

    Accumulates until target_span (or max_span / 120-sentence cap), merges a
    short tail into the previous group when it fits, drops spans below
    min_span. Returns a list of ID lists.
    """
    def _span(ids):
        return (max(sentences_by_id[i]["end_time"] for i in ids)
                - min(sentences_by_id[i]["start_time"] for i in ids))

    groups, cur = [], []
    for sid in sids:
        if sid not in sentences_by_id:
            continue
        cur.append(sid)
        if _span(cur) >= target_span or len(cur) >= 120 or _span(cur) >= max_span:
            groups.append(cur)
            cur = []
    if cur:
        if groups:
            merged = groups[-1] + cur
            if _span(merged) <= max_span:
                groups[-1] = merged
            elif _span(cur) >= min_span:
                groups.append(cur)
            # else: short tail that cannot merge — dropped
        else:
            groups.append(cur)
    return [g for g in groups if _span(g) >= min_span]


def _deterministic_section_clips(video_id, outline, words, sentences, chunks,
                                 target, campaign_text="", cfg=None,
                                 exclude_ranges=None, start_clip_number=1):
    """Last-resort picker when the LLM is unreachable (dead key / offline).

    One candidate per outline-section sub-span (hook = opening words, idea =
    section title/summary). Scored + inserted through the normal fusion path
    (local embeddings only — no LLM). Spans overlapping exclude_ranges
    (IoU > 0.5) are skipped so existing clips win. Clip numbering starts at
    start_clip_number. Returns inserted clips (may be fewer than target).
    """
    from pipeline.discovery import persist_candidates
    if not outline or not sentences:
        return []
    cfg = cfg or {}
    d_cfg = cfg.get("discovery", {})
    min_span = float(d_cfg.get("min_span_seconds", 15) or 0)
    max_span = float(d_cfg.get("max_span_seconds", 90) or 0)
    max_cand = int(d_cfg.get("max_candidates", 24) or 24)
    by_id = {s["id"]: s for s in sentences}
    excluded = [(float(a), float(b)) for a, b in (exclude_ranges or [])]

    def _excluded(rng):
        a, b = rng
        for x, y in excluded:
            inter = max(0.0, min(b, y) - max(a, x))
            union = (b - a) + (y - x) - inter
            if union > 0 and inter / union > 0.5:
                return True
        return False

    cands = []
    for sec in (outline.get("sections") or []):
        try:
            lo, hi = int(sec.get("start_id", 0)), int(sec.get("end_id", -1))
        except (TypeError, ValueError):
            continue
        sids = [i for i in range(lo, hi + 1) if i in by_id]
        for grp in _split_ids_by_time(sids, by_id, max_span=max_span or 90.0,
                                      min_span=min_span or 0):
            rng = (min(by_id[i]["start_time"] for i in grp),
                   max(by_id[i]["end_time"] for i in grp))
            if _excluded(rng):
                continue
            text = " ".join(by_id[i]["text"] for i in grp)
            cands.append({
                "sentence_ids": grp,
                "source_ranges": [[round(rng[0], 3), round(rng[1], 3)]],
                "hook": " ".join(text.split()[:12])[:300],
                "main_idea": str(sec.get("title", "")
                                 or sec.get("summary", ""))[:600],
                "payoff": "",
                "required_context": "",
                "rationale": ("deterministic section span "
                              "(LLM unreachable; unvalidated)"),
                "uncertainty": 0.7,
                "joins": [],
                "source": "outline_deterministic",
                "provenance": {"section": str(sec.get("title", ""))[:120]},
            })
            if len(cands) >= max_cand:
                break
        if len(cands) >= max_cand:
            break
    if not cands:
        return []
    ids = persist_candidates(video_id, cands, source="outline_deterministic")
    for c, i in zip(cands, ids):
        c["db_id"] = i
    pool = list(cands) + [r for r in _fusion_pool(video_id, [])
                          if r.get("source") == "audio_event"
                          and r.get("id") not in {c.get("db_id") for c in cands}]
    log.info("deterministic fallback candidates=%d pool=%d",
             len(cands), len(pool))
    inserted = _insert_clips_from_candidates(video_id, pool, chunks, target,
                                             campaign_text=campaign_text, cfg=cfg)
    # renumber when topping up an existing set
    if start_clip_number != 1:
        _renumber_clips(inserted, start_clip_number)
    return inserted


def _renumber_clips(clips, start):
    """Shift clip_number (DB + in-memory) to follow an existing set."""
    from db.connection import get_conn, release_conn
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            for i, c in enumerate(clips):
                cur.execute("UPDATE clips SET clip_number=%s WHERE id=%s",
                            (start + i, c["id"]))
            conn.commit()
    except Exception:
        log.exception("clip renumber failed")
        try:
            conn.rollback()
        except Exception:
            pass
    finally:
        release_conn(conn)
    for i, c in enumerate(clips):
        c["clip_number"] = start + i


def _insert_clips_from_candidates(video_id, candidates, chunks, target,
                                    campaign_text="", cfg=None):
    """Fuse + rank candidates (Phase 3), insert winners as clips.

    Preference comes from fusion scores; component scores and skip reasons
    are recorded on the candidates table. target is a maximum: fewer clips
    when quality is insufficient.
    """
    from db.repositories import candidate_repo
    from pipeline.fusion import rank_and_select

    selected, report = rank_and_select(candidates, campaign_text, cfg=cfg)
    # record scores + fusion decisions for the whole pool
    for cand in candidates:
        if not cand.get("db_id"):
            continue
        try:
            candidate_repo.update_candidate_status(
                cand["db_id"],
                "shortlisted" if cand in selected else cand.get("fusion_status", "proposed"),
                cand.get("fusion_reason", ""),
                scores=cand.get("scores"))
        except Exception:
            pass
    clips = []
    clip_number = 1
    for cand in selected:
        a, b = cand["source_ranges"][0]
        dur = b - a
        if dur < 15:
            candidate_repo.update_candidate_status(
                cand.get("db_id", ""), "rejected", "shorter than 15s") \
                if cand.get("db_id") else None
            continue
        hook = cand.get("hook") or ""
        reason_bits = [cand.get("rationale", "")]
        if cand.get("main_idea"):
            reason_bits.append("idea: " + cand["main_idea"][:200])
        reason = " ".join(x for x in reason_bits if x)[:600]
        # working title until editorial titles exist (Phase 3/6)
        title = (hook or cand.get("main_idea") or "")[:80] or None
        clip_text = f"{hook} {reason}"
        try:
            embedding = embed_text(clip_text, prompt_name="query")
        except Exception:
            log.exception("clip embed failed (two-pass)")
            raise
        # source chunks overlapped by the candidate range
        src_ids = [ch["id"] for ch in (chunks or [])
                   if ch.get("id") and not (ch.get("end_time", 0) < a or ch.get("start_time", 0) > b)]
        clip_id = clip_repo.insert_clip(
            video_id=video_id,
            clip_number=clip_number,
            start_time=a,
            end_time=b,
            duration_seconds=dur,
            hook=hook or None,
            reason=reason or None,
            suggested_title=title,
            suggested_hashtags=[],
            source_chunk_ids=src_ids,
            embedding=embedding,
        )
        try:
            from db.connection import get_conn, release_conn
            import json as _json
            conn = get_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute("UPDATE clips SET provenance=%s::jsonb WHERE id=%s",
                                (_json.dumps({"source": cand.get("source", "outline"),
                                              "sentence_ids": cand["sentence_ids"],
                                              "uncertainty": cand.get("uncertainty"),
                                              "fusion_score": (cand.get("scores") or {}).get("total")}), clip_id))
                    conn.commit()
            finally:
                release_conn(conn)
        except Exception:
            log.debug("provenance persist skipped", exc_info=True)
        clip = {"id": clip_id, "clip_number": clip_number,
                "start_time": a, "end_time": b, "duration_seconds": dur,
                "hook": hook, "reason": reason, "suggested_title": title,
                "source_chunk_ids": src_ids, "embedding": embedding}
        clips.append(clip)
        if cand.get("db_id"):
            try:
                candidate_repo.update_candidate_status(
                    cand["db_id"], "selected",
                    cand.get("fusion_reason", ""),
                    clip_id=clip_id,
                    scores=cand.get("scores"))
            except Exception:
                pass
        clip_number += 1
    return clips


def _analyze_legacy(video_id: str, chunks: list[dict], config: dict) -> list[dict]:
    """Legacy per-chunk extraction (fallback only)."""
    all_clips = []
    clip_number = 1
    t0 = time.monotonic()
    log.info("legacy analyze start video=%.8s chunks=%d", video_id, len(chunks))

    for chunk in chunks:
        log.info("analyzing chunk %s (%.1fs → %.1fs)", chunk.get('chunk_index'),
                 chunk.get('start_time', 0), chunk.get('end_time', 0))
        try:
            chunk_clips = _extract_clips_from_chunk(chunk, config)
        except Exception:
            log.exception("chunk extract crashed idx=%s", chunk.get("chunk_index"))
            continue
        log.info("chunk %s yielded %d clips", chunk.get("chunk_index"), len(chunk_clips))

        for clip in chunk_clips:
            clip["clip_number"] = clip_number
            clip["source_chunk_ids"] = [chunk["id"]]

            # Embed the clip's hook immediately (query prompt: clips are used
            # as similarity queries against stored chunk/document embeddings)
            clip_text = f"{clip.get('hook', '')} {clip.get('reason', '')}"
            try:
                clip["embedding"] = embed_text(clip_text, prompt_name="query")
            except Exception:
                log.exception("clip embed failed clip_no=%d", clip["clip_number"])
                raise

            # Write to DB
            clip_id = clip_repo.insert_clip(
                video_id=video_id, 
                clip_number=clip["clip_number"],
                start_time=to_sec(clip["start_time"]),
                end_time=to_sec(clip["end_time"]),
                duration_seconds=clip.get("duration_seconds"),
                hook=clip.get("hook"),
                reason=clip.get("reason"),
                suggested_title=clip.get("suggested_title"),
                suggested_hashtags=clip.get("suggested_hashtags"),
                source_chunk_ids=clip["source_chunk_ids"],
                embedding=clip["embedding"]
            )
            clip["id"] = clip_id

            all_clips.append(clip)
            clip_number += 1

    # Deduplicate clips with overlapping timestamps (same moment found in two chunks)
    before = len(all_clips)
    all_clips = _deduplicate_clips(all_clips, overlap_threshold_seconds=5.0)

    log.info("analyze done video=%.8s clips=%d (dedup %d->%d) elapsed=%.1fs",
             video_id, len(all_clips), before, len(all_clips), time.monotonic() - t0)

    _write_clips_analysis_json(all_clips)
    return all_clips


def _write_clips_analysis_json(clips: list[dict]) -> None:
    """Save clips to <workspace>/clips_analysis.json (embeddings stripped)."""
    import os
    from pipeline.workspace import current
    out_dir = current().transcripts_dir
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'clips_analysis.json'), 'w') as f:
        # We need to make sure uuid/datetime are serializable or just save the primitive fields
        safe_clips = []
        for c in clips:
            safe_c = c.copy()
            if 'id' in safe_c: safe_c['id'] = str(safe_c['id'])
            if 'source_chunk_ids' in safe_c: safe_c['source_chunk_ids'] = [str(x) for x in safe_c['source_chunk_ids']]
            if 'embedding' in safe_c: del safe_c['embedding']
            safe_clips.append(safe_c)
        json.dump(safe_clips, f, indent=2)
    log.debug("wrote clips_analysis.json dir=%s clips=%d", out_dir, len(safe_clips))


def _deduplicate_clips(clips: list[dict], overlap_threshold_seconds: float) -> list[dict]:
    """
    Remove clips whose start_time is within overlap_threshold of another clip.
    Keeps the one with the longer duration (more context).
    """
    if not clips: return []

    clips = sorted(clips, key=lambda c: to_sec(str(c["start_time"])))
    kept = []

    for clip in clips:
        if not kept:
            kept.append(clip)
            continue
        last = kept[-1]
        
        last_start = to_sec(str(last["start_time"]))
        curr_start = to_sec(str(clip["start_time"]))
        
        if curr_start - last_start < overlap_threshold_seconds:
            # Duplicate — keep the longer one
            if clip.get("duration_seconds", 0) > last.get("duration_seconds", 0):
                log.debug("dedup replace start=%.1f dur %.0f->%.0f", curr_start,
                          last.get("duration_seconds", 0), clip.get("duration_seconds", 0))
                kept[-1] = clip
            else:
                log.debug("dedup drop start=%.1f dur=%.0f", curr_start, clip.get("duration_seconds", 0))
        else:
            kept.append(clip)

    return kept