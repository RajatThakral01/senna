"""pipeline/boundaries.py

Boundary refinement & completeness validation for viral clip candidates.

Problem this solves: the analyzer LLM returns HH:MM:SS timestamps that can
land mid-sentence, so rendered clips end before the speaker finishes their
thought.

Pipeline position: runs AFTER analyze + similarity (candidate selection) and
BEFORE render. It never treats similarity matches elsewhere in the video as
proof of direct continuation — only *adjacent* transcript content may extend
a boundary.

Strategy (deterministic first, LLM as validator):
  1. Load flat word list from transcripts/transcript.json
     (segments[].words[] or top-level word_segments).
  2. Group words into sentence/utterance units with stable integer IDs.
  3. Deterministically snap start -> sentence beginning, end -> sentence end.
  4. Ask the existing LLM (Groq OpenAI-compatible) to VALIDATE completeness
     and SELECT start/end sentence IDs from supplied context (~20s each side).
     Exact timestamps are derived from the transcript, never from the LLM.
  5. If no complete ending exists within limits, fall back to an earlier
     complete ending or reject the candidate with a recorded reason.
  6. Never truncate a sentence solely to meet a target duration.

Config (config.yaml [refine], all overridable without code edits):
  enabled, context_seconds (20), max_extension_seconds (15),
  min_duration (20), max_duration (90),
  start_padding (0.15), end_padding (0.3), llm_validation (true)

DB propagation: refined ranges are written back via clip_repo.update_refinement
(new columns, graceful fallback to start/end update when the migration has
not been applied). Render, subtitles and report all read the refined ranges.
"""
import json
import os
import re
import time

from logger import get_logger

log = get_logger("pipeline.boundaries")

SENTENCE_END_RE = re.compile(r"[.!?…]+$")
FILLER_ONLY_RE = re.compile(r"^(so|and|well|you know|um|uh|like)[,.\s]*$", re.IGNORECASE)


def load_words(transcript_path=None):
    """Return flat [{word, start, end}] list. Handles missing data gracefully.

    transcript_path defaults to the active video workspace's transcript.
    """
    if transcript_path is None:
        from pipeline.workspace import current
        transcript_path = current().transcript
    with open(transcript_path, encoding="utf-8") as f:
        raw = json.load(f)
    words = []
    if isinstance(raw, dict) and "segments" in raw:
        for seg in raw.get("segments", []):
            for w in seg.get("words", []) or []:
                if "start" not in w or "end" not in w:
                    continue
                try:
                    words.append({
                        "word": str(w.get("word", "")).strip(),
                        "start": float(w["start"]),
                        "end": float(w["end"]),
                    })
                except (TypeError, ValueError):
                    continue
    elif isinstance(raw, list):
        for w in raw:
            if "start" not in w or "end" not in w:
                continue
            try:
                words.append({"word": str(w.get("word", "")) .strip(),
                              "start": float(w["start"]), "end": float(w["end"])})
            except (TypeError, ValueError):
                continue
    elif isinstance(raw, dict) and "word_segments" in raw:
        for w in raw.get("word_segments", []) or []:
            if "start" not in w or "end" not in w:
                continue
            try:
                words.append({"word": str(w.get("word", "")) .strip(),
                              "start": float(w["start"]), "end": float(w["end"])})
            except (TypeError, ValueError):
                continue
    words = [w for w in words if w["word"]]
    words.sort(key=lambda w: (w["start"], w["end"]))
    log.debug("boundaries loaded words=%d path=%s", len(words), transcript_path)
    return words


def build_sentences(words, pause_threshold=1.0):
    """Group words into sentence/utterance units with stable IDs.

    A unit ends at [.?!…] punctuation. Long speech pauses (>pause_threshold)
    without punctuation still end a unit but it is flagged
    ends_with_punctuation=False / complete=False (punctuation alone is
    insufficient for completeness — the LLM validates the thought).
    """
    sentences = []
    cur = []
    for i, w in enumerate(words):
        cur.append(i)
        is_last = (i == len(words) - 1)
        punct_end = bool(SENTENCE_END_RE.search(w["word"].rstrip()))
        pause_end = False
        if not is_last:
            try:
                gap = float(words[i + 1]["start"]) - float(w["end"])
            except (KeyError, TypeError, ValueError):
                gap = 0.0
            pause_end = gap >= pause_threshold and len(cur) >= 3
        if punct_end or pause_end or is_last:
            idxs = list(cur)
            s_words = [words[k] for k in idxs]
            text = " ".join(x["word"] for x in s_words)
            sentences.append({
                "id": len(sentences),
                "start_word": idxs[0],
                "end_word": idxs[-1],
                "start_time": float(s_words[0]["start"]),
                "end_time": float(s_words[-1]["end"]),
                "text": text,
                "ends_with_punctuation": punct_end,
                # heuristic only; LLM makes the final completeness call
                "heuristic_complete": punct_end and not FILLER_ONLY_RE.match(text.strip()),
            })
            cur = []
    return sentences


def _sentence_at_time(sentences, t):
    for s in sentences:
        if s["start_time"] <= t <= s["end_time"]:
            return s
    # between sentences: nearest by start
    past = [s for s in sentences if s["end_time"] <= t]
    if past:
        return past[-1]
    return sentences[0] if sentences else None


def _clamp_padding(t, pad_before, pad_after, words, video_duration=None):
    """Apply padding bounded by video and neighbouring speech.

    Start never moves into the previous word; end never moves into the next
    word. Video bounds always win.
    """
    return t  # padding applied by caller with neighbour info; kept for API compat


def snap_range(start, end, words, sentences, cfg):
    """Deterministic snap: start -> sentence beginning, end -> sentence end.

    Returns (new_start, new_end, info_dict).
    """
    pad_b = float(cfg.get("start_padding", 0.15))
    pad_a = float(cfg.get("end_padding", 0.3))
    if not words or not sentences:
        return start, end, {"method": "passthrough_no_transcript"}

    s_sent = _sentence_at_time(sentences, start)
    e_sent = _sentence_at_time(sentences, end)
    if s_sent is None or e_sent is None:
        return start, end, {"method": "passthrough_no_sentence"}

    ns = s_sent["start_time"] - pad_b
    ne = e_sent["end_time"] + pad_a

    # Bound by neighbouring speech: do not cut into adjacent words.
    if s_sent["start_word"] > 0:
        prev_end = words[s_sent["start_word"] - 1]["end"]
        ns = max(ns, prev_end + 0.01)
    ns = max(0.0, ns)
    if e_sent["end_word"] + 1 < len(words):
        next_start = words[e_sent["end_word"] + 1]["start"]
        ne = min(ne, next_start - 0.01)
    ne = max(ne, ns + 0.5)
    info = {"method": "deterministic_snap",
            "start_sentence_id": s_sent["id"], "end_sentence_id": e_sent["id"],
            "start_punct": s_sent["ends_with_punctuation"],
            "end_punct": e_sent["ends_with_punctuation"]}
    return ns, ne, info


def _boundary_llm(prompt, cfg, purpose="boundaries"):
    """One boundary LLM call -> parsed verdict dict, or None on any failure."""
    try:
        from config import LLM_MODEL
    except Exception:
        return None
    from pipeline.llm_client import pool_usable as _pool_usable
    if not _pool_usable():
        log.debug("boundaries LLM skipped (no key)")
        return None
    try:
        from pipeline.llm_client import post_chat, SLOT_FOR_STAGE
        r = post_chat({"model": LLM_MODEL, "temperature": 0.1,
                       "max_tokens": 500,
                       "messages": [{"role": "user", "content": prompt}]},
                      timeout=60, slot=SLOT_FOR_STAGE["boundaries"],
                      purpose=purpose, json_mode=True)
        raw = r.json()["choices"][0]["message"]["content"].strip()
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            return None
        data = json.loads(m.group(0))
        # selection_complete = does [start_id..end_id] finish the thought.
        # Older single-field answers ("complete") are read the same way.
        sel = data.get("selection_complete", data.get("complete", True))
        return {"start_id": int(data["start_id"]), "end_id": int(data["end_id"]),
                "complete": bool(sel),
                "candidate_complete": bool(data.get("candidate_complete", sel)),
                "missing": str(data.get("missing", "") or "")[:200],
                "reason": str(data.get("reason", ""))[:300]}
    except Exception:
        log.exception("%s LLM call failed, deterministic fallback", purpose)
        return None


def _sentence_lines(sentences):
    out = []
    for s in sentences:
        flag = "complete?" if not s["ends_with_punctuation"] else "punct"
        out.append(f'[{s["id"]}] ({s["start_time"]:.1f}-{s["end_time"]:.1f}s, {flag}) {s["text"]}')
    return "\n".join(out)


_VERDICT_SCHEMA = ('Respond with ONLY JSON: {"start_id": <int>, "end_id": <int>, '
                   '"candidate_complete": <true|false>, "selection_complete": <true|false>, '
                   '"missing": "<what is still missing if selection_complete is false>", '
                   '"reason": "<short reason>"}')


def _llm_validate(clip, sentences_in_window, cfg):
    """Ask the LLM for start/end sentence IDs + completeness of THAT selection.

    Returns {start_id, end_id, complete, candidate_complete, missing, reason}
    or None on any failure (missing key, timeout, bad JSON) — caller falls
    back to deterministic. Timestamps are NEVER taken from the LLM, only IDs.

    The verdict used for decisions is `complete` = whether the SELECTED range
    finishes the thought. The old prompt only asked about the original
    candidate, so "incomplete" was ambiguous: the code then walked back from
    an ending the model had already fixed, or kept an unfinished one.
    """
    if not cfg.get("llm_validation", True):
        return None
    prompt = (
        "You are a video editor fixing the boundaries of a short clip.\n"
        f'Candidate clip: {clip.get("start_time")}s -> {clip.get("end_time")}s. '
        f'Hook: {clip.get("hook", "")}\n'
        "Sentences with [ID] (times are exact, from word timestamps):\n"
        f"{_sentence_lines(sentences_in_window)}\n\n"
        "1. Does the candidate as given complete its main thought, explanation or punchline? "
        "Punctuation alone is not enough — a complete sentence can still leave the thought "
        'unfinished (e.g. a "because..." setup, an unanswered question, a missing payoff).\n'
        "2. Select the best start sentence ID (a natural hook beginning, not filler) and the end "
        "sentence ID AFTER which the payoff has landed, using ONLY IDs above. Extend past the "
        "candidate end when the payoff comes later.\n"
        "3. selection_complete: does YOUR selected start..end range finish the thought?\n"
        + _VERDICT_SCHEMA
    )
    return _boundary_llm(prompt, cfg)


def _llm_extend(clip, start_sent, end_sent, forward, missing, cfg):
    """Second pass: find where the payoff lands in the sentences AFTER end_sent.

    forward: sentences from start_sent onward (wider than the first window).
    Returns the same verdict shape, or None.
    """
    prompt = (
        "You are a video editor. A short clip ends before its payoff.\n"
        f'Hook: {clip.get("hook", "")}\n'
        f'It currently runs from sentence [{start_sent["id"]}] to [{end_sent["id"]}], '
        f'and this is still missing: {missing or "the payoff / conclusion"}.\n'
        "Sentences from the clip start onward, with [ID]:\n"
        f"{_sentence_lines(forward)}\n\n"
        f'Keep start_id = {start_sent["id"]}. Choose the end_id AFTER which the payoff, answer '
        "or punchline has landed, as early as possible, using ONLY IDs above. If the thought "
        "never resolves within these sentences, set selection_complete to false.\n"
        + _VERDICT_SCHEMA
    )
    return _boundary_llm(prompt, cfg, purpose="boundaries-extend")


def refine_clip(clip, words, sentences, video_duration=None, cfg=None):
    """Refine one clip candidate. Returns (refined_clip_dict, action).

    action is one of: refined | rejected | passthrough.
    refined_clip_dict carries refined_start_time/refined_end_time plus
    refine_status/refine_reason/source_ranges for DB + report.
    """
    cfg = cfg or {}
    ctx = float(cfg.get("context_seconds", 20))
    max_ext = float(cfg.get("max_extension_seconds", 15))
    min_d = float(cfg.get("min_duration", 20))
    max_d = float(cfg.get("max_duration", 90))
    pad_b = float(cfg.get("start_padding", 0.15))
    pad_a = float(cfg.get("end_padding", 0.3))

    try:
        orig_start = float(clip["start_time"])
        orig_end = float(clip["end_time"])
    except (KeyError, TypeError, ValueError):
        return clip, "passthrough"

    if video_duration:
        orig_end = min(orig_end, float(video_duration))
    if orig_end <= orig_start:
        r = dict(clip); r.update({"refine_status": "rejected",
                                  "refine_reason": "invalid_time_range",
                                  "refined_start_time": orig_start,
                                  "refined_end_time": orig_end})
        return r, "rejected"

    # Window of sentences around the candidate for LLM context.
    window = [s for s in sentences
              if s["end_time"] >= orig_start - ctx and s["start_time"] <= orig_end + ctx]
    if not window:
        # No transcript coverage: keep as-is, mark passthrough.
        r = dict(clip)
        r.update({"refine_status": "passthrough_no_transcript",
                  "refine_reason": "no transcript coverage in window",
                  "refined_start_time": orig_start, "refined_end_time": orig_end,
                  "source_ranges": [[orig_start, orig_end]]})
        return r, "passthrough"

    det_start, det_end, det_info = snap_range(orig_start, orig_end, words, sentences, cfg)

    payoff_ext = float(cfg.get("max_payoff_extension_seconds", 30))
    verdict = _llm_validate({"start_time": orig_start, "end_time": orig_end,
                             "hook": clip.get("hook", "")}, window, cfg)
    valid_ids = {s["id"] for s in window}
    llm_checked_end = False
    if verdict and verdict["start_id"] in valid_ids and verdict["end_id"] in valid_ids:
        by_id = {s["id"]: s for s in sentences}
        s_sel, e_sel = by_id[verdict["start_id"]], by_id[verdict["end_id"]]
        if e_sel["id"] < s_sel["id"]:
            s_sel, e_sel = e_sel, s_sel
        method = "llm_validated"
        reason = verdict.get("reason", "")
        if verdict["complete"]:
            llm_checked_end = True
        else:
            # The selected range still ends before the payoff: look further
            # ahead (first window only reaches context_seconds past the end).
            forward = [s for s in sentences if s["id"] >= s_sel["id"]
                       and s["start_time"] <= orig_end + payoff_ext]
            ext = _llm_extend(clip, s_sel, e_sel, forward, verdict.get("missing", ""), cfg)
            fwd_ids = {s["id"] for s in forward}
            if (ext and ext["complete"] and ext["end_id"] in fwd_ids
                    and ext["end_id"] >= e_sel["id"]):
                e_sel = by_id[ext["end_id"]]
                method = "llm_extended_to_payoff"
                reason = f'{reason} | extended to [{e_sel["id"]}]: {ext.get("reason", "")}'
                llm_checked_end = True
            elif cfg.get("reject_incomplete", True):
                r = dict(clip)
                r.update({"refine_status": "rejected",
                          "refine_reason": ("thought unfinished within "
                                            f"{payoff_ext:.0f}s: {reason}")[:500],
                          "refined_start_time": orig_start, "refined_end_time": orig_end,
                          "source_ranges": [[orig_start, orig_end]]})
                return r, "rejected"
            else:
                method = "llm_incomplete_kept"
        llm_start = s_sel["start_time"] - pad_b
        llm_end = e_sel["end_time"] + pad_a
        # Bound by neighbouring speech + video.
        if s_sel["start_word"] > 0:
            llm_start = max(llm_start, words[s_sel["start_word"] - 1]["end"] + 0.01)
        llm_start = max(0.0, llm_start)
        if e_sel["end_word"] + 1 < len(words):
            llm_end = min(llm_end, words[e_sel["end_word"] + 1]["start"] - 0.01)
        if video_duration:
            llm_end = min(llm_end, float(video_duration))
            llm_start = max(0.0, min(llm_start, float(video_duration) - 0.5))
        new_start, new_end = llm_start, max(llm_end, llm_start + 0.5)
        info = {"method": method, "llm_reason": reason,
                "start_sentence_id": s_sel["id"], "end_sentence_id": e_sel["id"],
                "llm_complete": llm_checked_end}
    else:
        new_start, new_end, info = det_start, det_end, dict(det_info)
        info["llm_reason"] = "llm_unavailable_or_invalid; deterministic snap used"

    # An ending the LLM confirmed lands the payoff may extend up to
    # max_payoff_extension_seconds; unverified endings keep the tighter cap.
    if llm_checked_end:
        max_ext = max(max_ext, payoff_ext)

    # Enforce extension + duration limits WITHOUT truncating a sentence.
    if new_end - orig_end > max_ext:
        # cap extension by pulling the end back to a sentence end within limit
        cap = orig_end + max_ext
        cands = [s for s in window if s["end_time"] + pad_a <= cap and s["ends_with_punctuation"]]
        if cands:
            e = cands[-1]
            new_end = e["end_time"] + pad_a
            info["method"] += "+capped_to_complete_ending"
        else:
            # keep deterministic end if already within cap else reject-able
            if det_end <= cap:
                new_end = det_end
            else:
                r = dict(clip)
                r.update({"refine_status": "rejected",
                          "refine_reason": "no complete ending within max_extension",
                          "refined_start_time": orig_start, "refined_end_time": orig_end,
                          "source_ranges": [[orig_start, orig_end]]})
                return r, "rejected"
    dur = new_end - new_start
    if dur > max_d:
        # shorten from the START to a later sentence beginning (never cut the end)
        cands = [s for s in window if s["start_time"] >= new_end - max_d
                 and s["start_time"] >= new_start]
        if cands:
            new_start = max(new_start, cands[0]["start_time"] - pad_b)
            info["method"] += "+start_trimmed_for_max_duration"
        else:
            r = dict(clip)
            r.update({"refine_status": "rejected",
                      "refine_reason": f"cannot fit max_duration {max_d}s without cutting a sentence",
                      "refined_start_time": new_start, "refined_end_time": new_end,
                      "source_ranges": [[new_start, new_end]]})
            return r, "rejected"
    if dur < 1.0:
        r = dict(clip)
        r.update({"refine_status": "rejected", "refine_reason": "refined duration < 1s",
                  "refined_start_time": new_start, "refined_end_time": new_end,
                  "source_ranges": [[new_start, new_end]]})
        return r, "rejected"
    # min_duration is advisory (analyzer already filters); do not pad into
    # unrelated speech just to reach it.
    _ = min_d

    r = dict(clip)
    r.update({"refined_start_time": round(new_start, 3),
              "refined_end_time": round(new_end, 3),
              "refined_duration": round(new_end - new_start, 3),
              "refine_status": "refined" if info.get("method") != "passthrough_no_transcript" else "passthrough",
              "refine_reason": f'{info.get("method")}: {info.get("llm_reason", "")}'[:500],
              "refine_start_sentence": info.get("start_sentence_id"),
              "refine_end_sentence": info.get("end_sentence_id"),
              "source_ranges": [[round(new_start, 3), round(new_end, 3)]]})
    return r, "refined"


def _merged_total(ranges):
    """Total seconds covered by ranges after merging overlaps/touching ones."""
    total, cur = 0.0, None
    for a, b in sorted(ranges):
        if cur is None or a > cur[1] + 0.05:
            if cur is not None:
                total += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    return total + ((cur[1] - cur[0]) if cur else 0.0)


def refine_all_clips(clips, transcript_path=None,
                     video_duration=None, cfg=None):
    """Refine every clip; stitched continuations get their own refined ranges.

    Each clip may carry related_segments (confirmed). The primary range and
    each continuation range are refined independently against ADJACENT
    transcript only; the combined source_ranges + output timeline mapping
    ({source_start, source_end, output_start, output_end} per segment) is
    stored on the clip for the cutter, subtitle generator and report.
    Rejected clips are kept in the list with refine_status=rejected so the
    caller can skip rendering them with a recorded reason.
    """
    cfg = cfg or {}
    t0 = time.monotonic()
    words = load_words(transcript_path)
    sentences = build_sentences(words)
    log.info("refine start clips=%d words=%d sentences=%d", len(clips), len(words), len(sentences))
    out = []
    try:
        from pipeline.fusion import limit_stitch_ranges as _lim
        _has_limits = True
    except Exception:
        _has_limits = False
    for clip in clips:
        # primary range refinement
        refined, action = refine_clip(clip, words, sentences, video_duration, cfg)
        if action == "rejected":
            out.append(refined)
            log.info("clip %s rejected: %s", clip.get("clip_number"), refined.get("refine_reason"))
            continue
        # validate non-contiguous joins before spending refinement on them
        segs = clip.get("related_segments", []) or []
        if _has_limits:
            try:
                kept, dropped = _lim(
                    (float(clip.get("start_time", 0)),
                     float(clip.get("end_time", 0))),
                    segs, cfg)
                for _seg, _why in dropped:
                    log.info("clip %s continuation %.1f-%.1f dropped: %s",
                             clip.get("clip_number"),
                             float(_seg.get("start_time", 0)),
                             float(_seg.get("end_time", 0)), _why)
                segs = kept
            except Exception:
                log.debug("stitch limits skipped", exc_info=True)
        ranges = [tuple(refined["source_ranges"][0])]
        # the whole stitched clip must respect the total cap; refinement can
        # extend each part (payoff extension), so re-check after every part
        max_total = float(cfg.get("max_total_seconds")
                          or (cfg.get("similarity") or {}).get("max_total_seconds")
                          or cfg.get("max_duration", 90))
        # refine each confirmed continuation independently
        for seg in segs:
            try:
                s0, s1 = float(seg["start_time"]), float(seg["end_time"])
            except (KeyError, TypeError, ValueError):
                continue
            pseudo = {"start_time": s0, "end_time": s1, "hook": ""}
            rr, act = refine_clip(pseudo, words, sentences, video_duration, cfg)
            if act == "rejected":
                log.info("clip %s continuation %.1f-%.1f dropped: %s",
                         clip.get("clip_number"), s0, s1, rr.get("refine_reason"))
                continue
            cand = (rr["refined_start_time"], rr["refined_end_time"])
            if _merged_total(ranges + [cand]) > max_total + 0.5:
                log.info("clip %s continuation %.1f-%.1f dropped: total would exceed %.0fs",
                         clip.get("clip_number"), cand[0], cand[1], max_total)
                continue
            ranges.append(cand)
        # merge overlaps, sort
        ranges = sorted(ranges)
        merged = []
        for a, b in ranges:
            if not merged or a > merged[-1][1] + 0.05:
                merged.append([a, b])
            else:
                merged[-1][1] = max(merged[-1][1], b)
        # output timeline mapping (gap removal on concat)
        timeline, out_t = [], 0.0
        for a, b in merged:
            d = b - a
            timeline.append({"source_start": round(a, 3), "source_end": round(b, 3),
                             "output_start": round(out_t, 3),
                             "output_end": round(out_t + d, 3)})
            out_t += d
        refined["source_ranges"] = [[round(a, 3), round(b, 3)] for a, b in merged]
        refined["timeline"] = timeline
        refined["output_duration"] = round(out_t, 3)
        refined["refined_start_time"] = merged[0][0]
        refined["refined_end_time"] = merged[-1][1]
        refined["refined_duration"] = round(out_t, 3)
        out.append(refined)
    log.info("refine done clips=%d elapsed=%.1fs", len(out), time.monotonic() - t0)
    return out


def persist_refinements(video_id, refined_clips):
    """Write refined boundaries back to the clips table (best-effort).

    Uses the new columns when the v4 migration has been applied; otherwise
    falls back to updating start/end/duration so render+report stay correct.
    Never raises — logs and returns counts.
    """
    ok, failed = 0, 0
    try:
        from db.repositories import clip_repo
        from db.connection import get_conn, release_conn
    except Exception:
        log.warning("refine persist skipped (no db layer)")
        return 0, len(refined_clips)
    for c in refined_clips:
        try:
            if hasattr(clip_repo, "update_refinement"):
                clip_repo.update_refinement(
                    c["id"],
                    float(c.get("refined_start_time", c["start_time"])),
                    float(c.get("refined_end_time", c["end_time"])),
                    c.get("refine_status", "refined"),
                    c.get("refine_reason", ""),
                    c.get("source_ranges"),
                    c.get("timeline"),
                )
            else:
                conn = get_conn()
                try:
                    with conn.cursor() as cur:
                        cur.execute("UPDATE clips SET start_time=%s, end_time=%s, "
                                    "duration_seconds=%s WHERE id=%s",
                                    (float(c.get("refined_start_time", c["start_time"])),
                                     float(c.get("refined_end_time", c["end_time"])),
                                     float(c.get("refined_duration",
                                                c.get("duration_seconds", 0))),
                                     c["id"]))
                        conn.commit()
                finally:
                    release_conn(conn)
            ok += 1
        except Exception:
            log.exception("refine persist failed clip=%s", c.get("clip_number"))
            failed += 1
    log.info("refine persist video=%.8s ok=%d failed=%d", str(video_id), ok, failed)
    return ok, failed
