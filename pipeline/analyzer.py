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
            response = requests.post(
                LLM_API_URL,
                headers={
                    "Authorization": f"Bearer {LLM_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2000,
                    "temperature": 0.3
                },
                timeout=120
            )
            response.raise_for_status()
            break  # Success
        except requests.exceptions.RequestException as e:
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


def analyze(video_id: str, chunks: list[dict], config: dict) -> list[dict]:
    """
    Analyze each chunk independently for viral moments.
    Returns a merged, deduplicated list of clip dicts.
    """
    all_clips = []
    clip_number = 1
    t0 = time.monotonic()
    log.info("analyze start video=%.8s chunks=%d", video_id, len(chunks))

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
    
    # Save the deduplicated clips to clips_analysis.json for legacy compatibility
    import os
    os.makedirs('transcripts', exist_ok=True)
    with open('transcripts/clips_analysis.json', 'w') as f:
        # We need to make sure uuid/datetime are serializable or just save the primitive fields
        safe_clips = []
        for c in all_clips:
            safe_c = c.copy()
            if 'id' in safe_c: safe_c['id'] = str(safe_c['id'])
            if 'source_chunk_ids' in safe_c: safe_c['source_chunk_ids'] = [str(x) for x in safe_c['source_chunk_ids']]
            if 'embedding' in safe_c: del safe_c['embedding']
            safe_clips.append(safe_c)
        json.dump(safe_clips, f, indent=2)
    log.debug("wrote transcripts/clips_analysis.json clips=%d", len(safe_clips))

    return all_clips


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