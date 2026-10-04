"""
pipeline/chunker.py

Semantic + Silence-Gap + Sliding Window Chunker.

Input:
    transcript_path (str): Path to transcript.json produced by pipeline/transcriber.py.
                           The file must contain either:
                             - A list of word dicts at top level: [{word, start, end}, ...]
                             - A dict with a "word_segments" key: {"word_segments": [{word, start, end}, ...]}
    video_id (str):         UUID of the video row in the DB (from video_repo.insert_video).

Output:
    list[dict]: Chunk dicts written to the DB via chunk_repo.insert_chunk.
                Each dict contains:
                    id, chunk_index, start_time, end_time, text,
                    token_count, is_overlap_tail, silence_gap_before, words

Chunking Strategy (three layers applied in order):
    A. Silence-Gap Detection    — gaps >= silence_threshold_seconds are candidate cut points.
    B. Semantic Validation      — a candidate cut is confirmed only at a sentence boundary
                                  (word ending in . ? !). If no boundary found within 20 words,
                                  the gap itself is used as a fallback cut.
    C. Token Budget Safety Valve — chunks exceeding max_tokens_per_chunk are split at the
                                  nearest mid-point sentence boundary.
    D. Sliding Window Overlap   — the last overlap_seconds of each chunk's text is prepended
                                  to the next chunk so that viral moments near boundaries
                                  are never lost.
"""

import json
import time
from logger import get_logger, log_stage
from config import get_config
from db.repositories import chunk_repo

log = get_logger("pipeline.chunker")

SENTENCE_ENDINGS = {'.', '!', '?'}


def _approximate_token_count(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 characters."""
    return max(1, len(text) // 4)


def _is_sentence_end(word: str) -> bool:
    """True if the word ends with a sentence-ending punctuation mark."""
    stripped = word.strip()
    return len(stripped) > 0 and stripped[-1] in SENTENCE_ENDINGS


def _load_words(transcript_path: str) -> list:
    """
    Load the flat list of word dicts from transcript.json.

    Handles two formats:
      1. List at top level:          [{word, start, end}, ...]
      2. Dict with word_segments key: {"word_segments": [{word, start, end}, ...]}
         (this is the format produced by WhisperX / our transcriber.py)

    Filters out any word entries that are missing start or end timestamps,
    as those cannot be used for silence-gap detection.
    """
    with open(transcript_path, encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        words = raw
    elif isinstance(raw, dict) and "word_segments" in raw:
        words = raw["word_segments"]
    else:
        raise ValueError(
            f"Unrecognised transcript format in {transcript_path}. "
            "Expected a list of word dicts or a dict with 'word_segments' key."
        )

    # Filter out words missing timestamps (WhisperX occasionally omits them)
    valid = [w for w in words if "start" in w and "end" in w]
    skipped = len(words) - len(valid)
    if skipped:
        log.warning("skipped %d words with missing timestamps", skipped)

    return valid


def build_chunks(transcript_path: str, video_id: str) -> list:
    """
    Main entry point. Reads transcript.json, applies three-layer chunking,
    writes each chunk to the DB, and returns the list of chunk dicts.

    Args:
        transcript_path: Path to transcript.json.
        video_id:        UUID string of the video row in the DB.

    Returns:
        List of chunk dicts (with DB-assigned 'id' key populated).
    """
    cfg = get_config()["chunking"]
    silence_threshold = cfg["silence_threshold_seconds"]
    max_tokens        = cfg["max_tokens_per_chunk"]
    overlap_seconds   = cfg["overlap_seconds"]
    min_duration      = cfg["min_chunk_duration_seconds"]
    t0 = time.monotonic()
    log.info("chunk start video=%.8s path=%s silence_thr=%.2f max_tokens=%d overlap=%ds min_dur=%ds",
             video_id, transcript_path, silence_threshold, max_tokens, overlap_seconds, min_duration)

    words = _load_words(transcript_path)
    if not words:
        log.error("no usable words in transcript path=%s", transcript_path)
        raise ValueError(f"No usable words found in transcript at {transcript_path}")

    log.info("loaded %d words", len(words))

    # ── Step A + B: Silence-gap detection + Semantic validation ──────────────
    cut_indices = []  # word indices where a NEW chunk starts (exclusive upper bound of prev chunk)

    for i in range(1, len(words)):
        prev_word = words[i - 1]
        curr_word = words[i]
        gap = curr_word["start"] - prev_word["end"]

        if gap >= silence_threshold:
            if _is_sentence_end(prev_word["word"]):
                # Clean cut at sentence boundary — confirm immediately
                cut_indices.append(i)
            else:
                # Walk forward up to 20 words to find the next sentence boundary
                found = False
                for j in range(i, min(i + 20, len(words))):
                    if _is_sentence_end(words[j]["word"]):
                        cut_indices.append(j + 1)
                        found = True
                        break
                if not found:
                    # Safety fallback: cut at the silence gap itself
                    cut_indices.append(i)

    # ── Step C: Token Budget Safety Valve ────────────────────────────────────
    verified_cuts = sorted(set(cut_indices))
    final_cuts = []
    chunk_start_idx = 0

    for cut in verified_cuts:
        chunk_words = words[chunk_start_idx:cut]
        chunk_text = " ".join(w["word"] for w in chunk_words)
        if _approximate_token_count(chunk_text) > max_tokens:
            # Split this oversized chunk at the nearest mid-point sentence boundary
            mid = (chunk_start_idx + cut) // 2
            split_point = None
            for k in range(mid, cut):
                if _is_sentence_end(words[k]["word"]):
                    split_point = k + 1
                    break
            final_cuts.append(split_point if split_point else mid)
        final_cuts.append(cut)
        chunk_start_idx = cut

    final_cuts = sorted(set(final_cuts))

    # ── Build raw chunks from the confirmed cut points ────────────────────────
    boundaries = [0] + final_cuts + [len(words)]
    raw_chunks = []

    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx   = boundaries[i + 1]
        chunk_words = words[start_idx:end_idx]

        if not chunk_words:
            continue

        start_time = chunk_words[0]["start"]
        end_time   = chunk_words[-1]["end"]
        duration   = end_time - start_time

        if duration < min_duration:
            log.debug("skip short chunk %.1fs < %ss", duration, min_duration)
            continue

        text = " ".join(w["word"] for w in chunk_words)
        silence_gap = (
            chunk_words[0]["start"] - words[start_idx - 1]["end"]
            if start_idx > 0 else 0.0
        )

        raw_chunks.append({
            "chunk_index":        len(raw_chunks),
            "start_time":         start_time,
            "end_time":           end_time,
            "text":               text,
            "token_count":        _approximate_token_count(text),
            "is_overlap_tail":    False,
            "silence_gap_before": silence_gap,
            "words":              chunk_words,   # kept in memory only, not stored in DB
        })

    log.info("raw chunks=%d cuts=%d", len(raw_chunks), len(final_cuts))

    # ── Step D: Sliding Window Overlap ────────────────────────────────────────
    final_chunks = []

    for i, chunk in enumerate(raw_chunks):
        if i == 0:
            final_chunks.append(chunk)
            continue

        prev_chunk = raw_chunks[i - 1]
        overlap_cutoff = chunk["start_time"] - overlap_seconds

        # Collect words from the previous chunk that fall inside the overlap window
        overlap_words = [
            w for w in prev_chunk["words"]
            if w["end"] >= overlap_cutoff
        ]

        if overlap_words:
            overlap_text = " ".join(w["word"] for w in overlap_words)
            full_text    = overlap_text + " " + chunk["text"]
            new_start    = overlap_words[0]["start"]
        else:
            full_text = chunk["text"]
            new_start = chunk["start_time"]

        final_chunks.append({
            **chunk,
            "text":            full_text,
            "start_time":      new_start,
            "token_count":     _approximate_token_count(full_text),
            "is_overlap_tail": bool(overlap_words),
        })

    # ── Write chunks to DB ────────────────────────────────────────────────────
    db_chunks = []
    for i, chunk in enumerate(final_chunks):
        chunk["chunk_index"] = i   # re-index after overlap adjustments

        chunk_id = chunk_repo.insert_chunk(
            video_id=video_id,
            chunk_index=chunk["chunk_index"],
            start_time=chunk["start_time"],
            end_time=chunk["end_time"],
            text=chunk["text"],
            token_count=chunk["token_count"],
            is_overlap_tail=chunk["is_overlap_tail"],
            silence_gap_before=chunk["silence_gap_before"],
        )
        chunk["id"] = chunk_id
        db_chunks.append(chunk)
        log.debug("chunk saved idx=%d id=%.8s %.1fs->%.1fs tokens=%d overlap=%s",
                  chunk["chunk_index"], str(chunk_id), chunk["start_time"],
                  chunk["end_time"], chunk["token_count"], chunk["is_overlap_tail"])

    log.info("chunk done video=%.8s chunks=%d elapsed=%.1fs", video_id, len(db_chunks), time.monotonic() - t0)
    return db_chunks
