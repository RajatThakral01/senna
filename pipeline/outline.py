"""
pipeline/outline.py

Whole-video understanding, pass 1 of contextual discovery.

Builds a structured outline from the transcript:
  - topic sections with timestamp references + source sentence IDs
  - questions and answers
  - stories and their setup / development / payoff
  - speaker turns when diarization is available
  - a combined video-level outline for long videos

Summaries are navigation aids only — the original transcript text remains
authoritative. Candidates (pass 2, pipeline/discovery.py) must cite source
sentence IDs; timestamps are always derived from word data, never the LLM.

Graceful degradation: without an LLM key (or on API failure) a deterministic
outline is built from chunk boundaries and clearly marked method=deterministic.
"""
import json
import time

from logger import get_logger

log = get_logger("pipeline.outline")


def _window_sentences(sentences, window_size=40, overlap=4):
    """Split sentence list into overlapping windows of sentence IDs."""
    windows = []
    i = 0
    n = len(sentences)
    if n == 0:
        return windows
    step = max(1, window_size - overlap)
    while i < n:
        chunk = sentences[i:i + window_size]
        windows.append(chunk)
        if i + window_size >= n:
            break
        i += step
    return windows


def _speaker_for_sentence(s, diarization):
    if not diarization:
        return None
    mid = (s["start_time"] + s["end_time"]) / 2
    for seg in diarization:
        if seg.get("start", 0) <= mid <= seg.get("end", 0):
            return seg.get("speaker")
    return None


def _llm_section_outline(window, window_idx, cfg):
    """One LLM call: outline a window of sentences. Returns section dict or None."""
    try:
        from config import LLM_API_KEY, LLM_API_URL, LLM_MODEL
    except Exception:
        return None
    from pipeline.llm_client import pool_usable as _pool_usable
    if not _pool_usable():
        return None
    lines = []
    for s in window:
        lines.append(f'[{s["id"]}] ({s["start_time"]:.1f}s) {s["text"]}')
    prompt = (
        "You outline a video transcript window for short-form clip discovery.\n"
        "Sentences are listed as [ID] (start-seconds) text. Times are exact.\n"
        f"{chr(10).join(lines)}\n\n"
        "Return ONLY JSON with this shape:\n"
        '{"sections": [{"title": "...", "start_id": <int>, "end_id": <int>, '
        '"summary": "...", "topics": ["..."], '
        '"qa": [{"question": "...", "q_ids": [<int>], "answer": "...", "a_ids": [<int>]}], '
        '"stories": [{"arc": "...", "setup_ids": [<int>], "development_ids": [<int>], '
        '"payoff_ids": [<int>]}], '
        '"key_ids": [<int>]}]}\n'
        "Rules: use ONLY IDs listed above. Sections must be contiguous and cover "
        "the window in order. Keep summaries short."
    )
    try:
        import re
        from pipeline.llm_client import post_chat, SLOT_FOR_STAGE
        r = post_chat({"model": LLM_MODEL, "temperature": 0.2,
                       "max_tokens": 2000,
                       "messages": [{"role": "user", "content": prompt}]},
                      timeout=120, slot=SLOT_FOR_STAGE["outline"],
                      purpose="outline")
        raw = r.json()["choices"][0]["message"]["content"].strip()
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            return None
        data = json.loads(m.group(0))
        valid = {s["id"] for s in window}

        def _clean_ids(ids):
            return [i for i in (ids or []) if isinstance(i, int) and i in valid]

        sections = []
        for sec in data.get("sections", []) or []:
            sids = _clean_ids(sec.get("start_id") is not None and
                              list(range(sec.get("start_id"), (sec.get("end_id") or sec.get("start_id")) + 1)) or [])
            if not sids:
                sids = [window[0]["id"], window[-1]["id"]]
            by_id = {s["id"]: s for s in window}
            lo, hi = min(sids), max(sids)
            sections.append({
                "title": str(sec.get("title", ""))[:160],
                "start_id": lo, "end_id": hi,
                "start_time": by_id.get(lo, window[0])["start_time"],
                "end_time": by_id.get(hi, window[-1])["end_time"],
                "summary": str(sec.get("summary", ""))[:600],
                "topics": [str(t)[:80] for t in (sec.get("topics", []) or [])][:8],
                "qa": sec.get("qa", []) or [],
                "stories": sec.get("stories", []) or [],
                "key_ids": _clean_ids(sec.get("key_ids", []))[:12],
            })
        return {"sections": sections}
    except Exception:
        log.exception("outline section LLM failed (window %d)", window_idx)
        return None


def _combine_outlines(section_outlines, cfg):
    """Second LLM pass: combine section outlines into a video-level outline."""
    try:
        from config import LLM_API_KEY, LLM_API_URL, LLM_MODEL
    except Exception:
        return None
    from pipeline.llm_client import pool_usable as _pool_usable
    if not _pool_usable() or not section_outlines:
        return None
    lines = []
    for i, s in enumerate(section_outlines):
        lines.append(f'- S{i}: "{s.get("title", "")}" '
                     f'[{s.get("start_time", 0):.0f}-{s.get("end_time", 0):.0f}s] '
                     f'{s.get("summary", "")[:200]}')
    prompt = (
        "Combine these video section outlines into one video-level outline.\n"
        f"{chr(10).join(lines)}\n\n"
        'Return ONLY JSON: {"title": "...", "arc": "...", '
        '"main_topics": ["..."], "section_order": [0, 1, ...]} '
        "section_order lists section indexes in narrative order."
    )
    try:
        import re
        from pipeline.llm_client import post_chat, SLOT_FOR_STAGE
        r = post_chat({"model": LLM_MODEL, "temperature": 0.2,
                       "max_tokens": 800,
                       "messages": [{"role": "user", "content": prompt}]},
                      timeout=120, slot=SLOT_FOR_STAGE["outline"],
                      purpose="outline-combine")
        raw = r.json()["choices"][0]["message"]["content"].strip()
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        return json.loads(m.group(0)) if m else None
    except Exception:
        log.exception("outline combine LLM failed")
        return None


def build_outline(words, sentences, chunks=None, diarization=None, cfg=None):
    """Build the structured outline. Returns outline dict (method marked)."""
    cfg = cfg or {}
    o_cfg = cfg.get("outline", {}) if isinstance(cfg, dict) else {}
    t0 = time.monotonic()
    window_size = int(o_cfg.get("window_sentences", 40) or 40)
    max_windows = int(o_cfg.get("max_windows", 12) or 12)
    if not sentences:
        return {"method": "empty", "sections": [], "video": {},
                "sentence_count": 0}
    windows = _window_sentences(sentences, window_size)
    if len(windows) > max_windows:
        # even coverage: widen windows instead of dropping content
        window_size = max(window_size, -(-len(sentences) // max_windows) + 4)
        windows = _window_sentences(sentences, window_size)[:max_windows]
    log.info("outline start sentences=%d windows=%d", len(sentences), len(windows))

    sections, llm_ok = [], 0
    by_id = {s["id"]: s for s in sentences}
    for wi, window in enumerate(windows):
        res = _llm_section_outline(window, wi, cfg)
        if res and res.get("sections"):
            llm_ok += 1
            for sec in res["sections"]:
                sec["speaker_turns"] = _speaker_turns_for_range(
                    sec["start_id"], sec["end_id"], by_id, diarization)
                sections.append(sec)
        else:
            # deterministic fallback for this window
            sections.append({
                "title": _fallback_title(window),
                "start_id": window[0]["id"], "end_id": window[-1]["id"],
                "start_time": window[0]["start_time"],
                "end_time": window[-1]["end_time"],
                "summary": "", "topics": [], "qa": [], "stories": [],
                "key_ids": [s["id"] for s in window[:6]],
                "speaker_turns": _speaker_turns_for_range(
                    window[0]["id"], window[-1]["id"], by_id, diarization),
                "fallback": True,
            })
        # polite delay to respect free-tier rate limits
        time.sleep(2)
    video = _combine_outlines([s for s in sections if not s.get("fallback")], cfg) or {}
    method = "llm" if llm_ok == len(windows) else (
        "partial" if llm_ok else "deterministic")
    outline = {"method": method, "sections": sections, "video": video,
               "sentence_count": len(sentences),
               "elapsed": round(time.monotonic() - t0, 1)}
    log.info("outline done method=%s sections=%d elapsed=%.1fs",
             method, len(sections), time.monotonic() - t0)
    return outline


def _fallback_title(window):
    text = " ".join(s["text"] for s in window[:3])
    words = text.split()
    return " ".join(words[:8]) or f"{window[0]['start_time']:.0f}s"


def _speaker_turns_for_range(lo, hi, by_id, diarization):
    if not diarization:
        return []
    turns, last = [], None
    for sid in range(lo, hi + 1):
        s = by_id.get(sid)
        if not s:
            continue
        spk = _speaker_for_sentence(s, diarization)
        if spk != last:
            turns.append({"speaker": spk, "from_id": sid,
                          "from_time": s["start_time"]})
            last = spk
    return turns


def persist_outline(video_id, outline):
    """Write outline to DB (outlines table) + transcripts/outline.json."""
    from db.repositories import outline_repo
    outline_repo.clear_outlines(video_id)
    for i, sec in enumerate(outline.get("sections", [])):
        outline_repo.insert_outline(
            video_id, "section", i,
            start_time=sec.get("start_time"), end_time=sec.get("end_time"),
            title=sec.get("title"), content=sec)
    if outline.get("video"):
        outline_repo.insert_outline(video_id, "video", 0, title="video",
                                    content=outline["video"])
    try:
        import os
        os.makedirs("transcripts", exist_ok=True)
        with open("transcripts/outline.json", "w", encoding="utf-8") as f:
            json.dump(outline, f, indent=2, ensure_ascii=False)
    except Exception:
        log.exception("outline.json write failed")
    log.info("outline persisted video=%.8s sections=%d",
             video_id, len(outline.get("sections", [])))
    return outline
