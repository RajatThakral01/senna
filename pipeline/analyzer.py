"""
pipeline/analyzer.py

Analyzes transcript chunks to identify viral short-form video clips.
"""

import json
import requests

from config import LLM_API_KEY, LLM_MODEL, LLM_API_URL
from db.repositories import chunk_repo, clip_repo
from pipeline.embedder import embed_text

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
            print("  (JSON found inside think block)")
            return inside[start:end+1]

    print("RAW TEXT THAT FAILED JSON EXTRACTION:")
    print(raw_text)
    raise ValueError("JSON extraction failed")


def _extract_clips_from_chunk(chunk: dict, config: dict) -> list[dict]:
    """
    Send a single chunk's text to the LLM to extract 0-2 viral moments.
    """
    
    # Optional: build a timestamped text representation if word-level is available in chunk text,
    # but the chunk dictionary in DB just has "text", "start_time", "end_time".
    # For now we'll just give the LLM the text and overall timestamps, 
    # and ask it to estimate timestamps or we just use the chunk's timestamps as a rough boundary.
    # Actually, the user's prompt needs exact timestamps. Since the DB doesn't store word-level 
    # timestamps for the chunk, wait, how can the LLM give exact timestamps?
    # Let's provide the start and end time of the chunk so the LLM knows the bounds.
    # Wait, the old code built a timestamped transcript. 
    # We should probably pass the chunk text as is and ask for the hook.
    
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
            print(f"  [analyzer] API request failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return []

    raw = response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
    if not raw:
        return []
    
    try:
        raw = extract_json_from_response(raw)
        clips = json.loads(raw) if raw != '[]' else []
    except ValueError:
        return []

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
                print(f"  Skipped short clip (only {int(dur)}s)")
        except Exception as e:
            print(f"  Skipped malformed clip timestamp: {e}")

    return valid_clips


def analyze(video_id: str, chunks: list[dict], config: dict) -> list[dict]:
    """
    Analyze each chunk independently for viral moments.
    Returns a merged, deduplicated list of clip dicts.
    """
    all_clips = []
    clip_number = 1

    for chunk in chunks:
        print(f"[analyzer] Analyzing chunk {chunk['chunk_index']} ({chunk['start_time']:.1f}s → {chunk['end_time']:.1f}s)")
        chunk_clips = _extract_clips_from_chunk(chunk, config)

        for clip in chunk_clips:
            clip["clip_number"] = clip_number
            clip["source_chunk_ids"] = [chunk["id"]]

            # Embed the clip's hook immediately
            clip_text = f"{clip.get('hook', '')} {clip.get('reason', '')}"
            clip["embedding"] = embed_text(clip_text)

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
    all_clips = _deduplicate_clips(all_clips, overlap_threshold_seconds=5.0)

    print(f"[analyzer] Found {len(all_clips)} total clips across {len(chunks)} chunks")
    
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
                kept[-1] = clip
        else:
            kept.append(clip)

    return kept