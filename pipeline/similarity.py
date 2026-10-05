"""
pipeline/similarity.py

For each discovered clip, search all other chunks via pgvector cosine similarity.
Related chunks above the threshold are stored in related_segments table.
Optionally, a second LLM call confirms whether the related chunk is a genuine
continuation before setting confirmed_by_llm = TRUE.
"""

import os
import json
import time
import requests
import re
from logger import get_logger
from db.repositories import chunk_repo, clip_repo
from db.connection import get_conn, release_conn

from config import LLM_API_KEY, LLM_API_URL, LLM_MODEL

log = get_logger("pipeline.similarity")


def find_related_segments(video_id: str, clips: list[dict],
                           top_k: int = 3, threshold: float = 0.50) -> list[dict]:
    """
    Main entry point.
    For each clip, run a pgvector similarity search against all chunks of the same video.
    Excludes the chunk the clip was already derived from.
    Writes results to related_segments table.
    Returns the clips list augmented with a 'related_segments' key.
    """
    t0 = time.monotonic()
    log.info("similarity start video=%.8s clips=%d top_k=%d threshold=%.2f", video_id, len(clips), top_k, threshold)
    for clip in clips:
        cn = clip.get("clip_number")
        if not clip.get("embedding"):
            log.warning("clip %s has no embedding — skipping", cn)
            continue

        log.debug("clip %s vector search top_k=%d thr=%.2f", cn, top_k, threshold)
        related = chunk_repo.find_similar_chunks(
            query_embedding=clip["embedding"],
            video_id=video_id,
            top_k=top_k,
            threshold=threshold,
        )
        log.debug("clip %s candidates=%d", cn, len(related))

        # Exclude the source chunk itself
        source_ids = set(str(s) for s in clip.get("source_chunk_ids", []))
        related = [r for r in related if str(r["id"]) not in source_ids]
        if len(related) != len(related):
            log.debug("clip %s filtered source chunks", cn)

        confirmed = []
        for rel_chunk in related:
            decision = _llm_confirm_continuation(clip, rel_chunk)
            # Tuning line: similarity score + LLM verdict per candidate
            log.info("clip %s candidate chunk_idx=%s sim=%.4f decision=%s range=%.1f-%.1f",
                     cn, rel_chunk.get("chunk_index"),
                     rel_chunk.get("similarity", 0), decision,
                     rel_chunk.get("start_time", 0), rel_chunk.get("end_time", 0))

            _write_related_segment(
                clip_id=clip["id"],
                related_chunk_id=rel_chunk["id"],
                similarity_score=rel_chunk["similarity"],
                confirmed_by_llm=(decision == "stitch"),
                decision=decision,
            )
            if decision == "stitch":
                confirmed.append(rel_chunk)

        clip["related_segments"] = confirmed
        if confirmed:
            log.info("clip %s has %d confirmed continuations", cn, len(confirmed))
        else:
            log.debug("clip %s no confirmed continuations", cn)

    log.info("similarity done video=%.8s elapsed=%.1fs", video_id, time.monotonic() - t0)
    return clips


def _llm_confirm_continuation(clip: dict, candidate_chunk: dict) -> str:
    """
    Ask the LLM to classify whether candidate_chunk is a continuation of clip.
    Returns one of three strings: 'stitch' | 'standalone' | 'noise'

    stitch     - direct narrative continuation, should be stitched into the clip
    standalone - same topic but different angle, works as its own clip
    noise      - coincidental overlap, not meaningfully related
    """
    prompt = f"""You are reviewing a short-form video clip and a candidate segment from later in the same video.

CLIP:
Title: {clip.get('suggested_title', '')}
Hook: {clip.get('hook', '')}
Transcript: {clip.get('text', '')[:300]}
Timestamps: {clip['start_time']:.1f}s \u2192 {clip['end_time']:.1f}s

CANDIDATE SEGMENT:
Transcript: {candidate_chunk['text'][:300]}
Timestamps: {candidate_chunk['start_time']:.1f}s \u2192 {candidate_chunk['end_time']:.1f}s

Classify the relationship between the candidate and the clip using exactly one of these three options:

"stitch"     \u2192 The candidate is a direct continuation of the clip's story, argument, or arc. A viewer watching the clip would feel this segment completes or resolves something the clip set up. Include it.

"standalone" \u2192 The candidate covers the same topic but from a different angle. It makes sense on its own but stitching it would feel abrupt or disconnected. Do not stitch.

"noise"      \u2192 Coincidental keyword overlap. Not meaningfully related. Reject.

Respond ONLY with a JSON object. No other text. No markdown fences.
{{"decision": "stitch"}} or {{"decision": "standalone"}} or {{"decision": "noise"}}"""

    max_retries = 4
    for attempt in range(max_retries):
        try:
            response = requests.post(
                LLM_API_URL,
                headers={"Authorization": f"Bearer {LLM_API_KEY}",
                         "Content-Type": "application/json"},
                json={
                    "model": LLM_MODEL,
                    "temperature": 0.1,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=30,
            )
            # 429 = Groq rate limit — back off and retry (free-tier bursts)
            if response.status_code == 429:
                wait = 10 * (attempt + 1)
                log.warning("LLM 429 rate-limited attempt %d/%d, waiting %ds",
                            attempt + 1, max_retries, wait)
                time.sleep(wait)
                continue
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"]

            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            if not raw:
                log.warning("LLM empty classification response, mapping to noise")
                return "noise"

            decision = json.loads(raw).get("decision", "noise")
            if decision not in ("stitch", "standalone", "noise"):
                log.warning("LLM invalid decision=%.20s, mapping to noise", decision)
                return "noise"
            return decision
        except requests.exceptions.RequestException as e:
            log.warning("LLM classification error attempt %d/%d: %s",
                        attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
    log.warning("LLM classification exhausted retries, mapping to noise")
    return "noise"


def _write_related_segment(clip_id: str, related_chunk_id: str,
                            similarity_score: float, confirmed_by_llm: bool,
                            decision: str):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO related_segments
                  (clip_id, related_chunk_id, similarity_score,
                   confirmed_by_llm, decision)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (clip_id, related_chunk_id, similarity_score,
                  confirmed_by_llm, decision))
            conn.commit()
            log.debug("related_segment saved decision=%s sim=%.3f confirmed=%s", decision, similarity_score, confirmed_by_llm)
    except Exception:
        log.exception("related_segment insert failed")
        raise
    finally:
        release_conn(conn)
