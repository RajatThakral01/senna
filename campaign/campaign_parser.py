import os
import json
import time
import uuid
import requests
import sys
from logger import get_logger
log = get_logger("campaign")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLM_API_KEY, LLM_API_URL, LLM_MODEL

SYSTEM_PROMPT = """
You are a video editing campaign configuration assistant.
Given a campaign description, extract the editing requirements and return ONLY a valid JSON object.
No explanation, no markdown, no preamble. Just raw JSON.

**CRITICAL RULE**: If the user asks for a generic asset (e.g., "add a logo", "use some music"), you MUST use the first available filename from the `available_assets` list provided in the user prompt. DO NOT invent or hallucinate filenames.
**CRITICAL RULE 2**: Only include features the user EXPLICITLY asks for. If the user does not mention a feature (like logo, music, split_screen), its value in the JSON MUST be null. Do not add features proactively.

JSON schema:
{
  "campaign_id": "string (generate a short UUID)",
  "logo": "string (filename only if mentioned, else null)",
  "logo_position": "top-right | top-left | bottom-right | bottom-left | null",
  "music": "string (filename only if mentioned, else null)",
  "music_volume": "float between 0.1-0.5 (default 0.3 if music mentioned, else null)",
  "split_screen": "boolean (true only if explicitly mentioned)",
  "split_screen_source": "null always for now",
  "layout": "auto | speaker_crop | stacked_split | branded_fit (default auto; use stacked_split only if user asks for split-screen/stacked, speaker_crop for face-tracked crop, branded_fit for full-frame-on-brand-colour)",
  "subtitles": "boolean (default true unless user says no subtitles)",
  "subtitle_style": "karaoke | standard | null",
  "aspect_ratio": "9:16 | 16:9 | 1:1 (default 9:16)",
  "template": "podcast_clip | tiktok_reaction | motivational_reel | null",
  "fade": "boolean (default true)",
  "min_clip_duration": "integer seconds (default 45)"
}
"""

def default_campaign() -> dict:
    """Static safe default: subtitles on, 9:16 karaoke, fades on, no branding."""
    return validate_campaign_config({
        "campaign_id": "default",
        "logo": None,
        "logo_position": None,
        "music": None,
        "music_volume": None,
        "split_screen": False,
        "split_screen_source": None,
        "layout": "auto",
        "subtitles": True,
        "subtitle_style": "karaoke",
        "aspect_ratio": "9:16",
        "template": None,
        "fade": True,
        "min_clip_duration": 45,
    })


def parse_campaign(description: str, available_assets: dict) -> dict:
    """
    available_assets = {
        "logos": ["logo.png", "brand.png"],
        "music": ["track1.mp3", "upbeat.mp3"]
    }
    """
    log.info("parse_campaign desc=%.120s logos=%d music=%d",
             description, len(available_assets.get("logos", [])),
             len(available_assets.get("music", [])))
    user_prompt = f"""
Campaign description: {description}

Available assets in assets/ folder:
Logos: {available_assets.get('logos', [])}
Music: {available_assets.get('music', [])}

Match mentioned assets to available files. If a specific file is not mentioned
but a type is (e.g., "add logo"), use the first available one.
Return only the JSON config.
"""
    if not LLM_API_KEY or LLM_API_KEY in ("your_groq_api_key_here", "your_grok_api_key_here", "your_nvidia_api_key_here", "your_minimax_api_key_here"):
        log.warning("LLM key missing, returning mock campaign config")
        # Return a mock response for testing without a real API key
        mock_config = {
            "campaign_id": "mock_id",
            "logo": "assets/logos/logo.png" if "logo" in description else None,
            "logo_position": "top-right",
            "music": None,
            "music_volume": None,
            "split_screen": "split screen" in description,
            "split_screen_source": None,
            "subtitles": "no subtitles" not in description,
            "subtitle_style": "karaoke",
            "aspect_ratio": "9:16",
            "template": "podcast_clip" if "podcast" in description else None,
            "fade": True,
            "min_clip_duration": 45
        }
        # A simple way to add a template
        if "motivational" in description:
            mock_config["template"] = "motivational_reel"
        if "reaction" in description:
            mock_config["template"] = "tiktok_reaction"
        lowered = description.lower()
        if "stacked" in lowered or "split-screen" in lowered or "split screen" in lowered:
            mock_config["layout"] = "stacked_split"
        elif "speaker crop" in lowered or "face-track" in lowered or "face track" in lowered:
            mock_config["layout"] = "speaker_crop"
        elif "branded" in lowered or "full frame" in lowered:
            mock_config["layout"] = "branded_fit"
        else:
            mock_config["layout"] = "auto"
        log.info("parse_campaign mock done logo=%s subtitles=%s template=%s",
                 mock_config.get("logo"), mock_config.get("subtitles"), mock_config.get("template"))
        return validate_campaign_config(mock_config)


    t0 = time.monotonic()
    log.debug("parse_campaign LLM call model=%s", LLM_MODEL)

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 1000,
        "temperature": 0.1
    }
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(LLM_API_URL, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
    except Exception:
        log.exception("parse_campaign LLM request failed")
        if not (description or "").strip():
            # Empty description + API failure (e.g. 429): fall back to safe
            # defaults so render/resume paths stay functional.
            log.warning("empty campaign + API error -> default campaign config")
            return default_campaign()
        raise

    raw = response.json()["choices"][0]["message"]["content"].strip()

    import re
    # Remove <think>...</think> blocks
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    config = json.loads(raw)
    config["campaign_id"] = str(uuid.uuid4())[:8]
    config = validate_campaign_config(config)
    log.info("parse_campaign done id=%s logo=%s music=%s subtitles=%s elapsed=%.1fs",
             config.get("campaign_id"), config.get("logo"), config.get("music"),
             config.get("subtitles"), time.monotonic() - t0)
    return config


SUPPORTED_LAYOUTS = ("auto", "speaker_crop", "stacked_split", "branded_fit")


def validate_campaign_config(config: dict) -> dict:
    """Validate campaign config against supported layout values (in-place)."""
    layout = str(config.get("layout") or "auto").strip().lower()
    if layout not in SUPPORTED_LAYOUTS:
        log.warning("campaign layout '%s' unsupported, using 'auto'", config.get("layout"))
        layout = "auto"
    config["layout"] = layout
    # legacy split_screen flag maps to stacked_split unless layout set explicitly
    if config.get("split_screen") and not config.get("_layout_explicit"):
        if layout == "auto":
            config["layout"] = "stacked_split"
    return config
