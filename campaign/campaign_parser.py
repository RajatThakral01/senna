import os
import json
import uuid
import requests
import sys
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
  "subtitles": "boolean (default true unless user says no subtitles)",
  "subtitle_style": "karaoke | standard | null",
  "aspect_ratio": "9:16 | 16:9 | 1:1 (default 9:16)",
  "template": "podcast_clip | tiktok_reaction | motivational_reel | null",
  "fade": "boolean (default true)",
  "min_clip_duration": "integer seconds (default 45)"
}
"""

def parse_campaign(description: str, available_assets: dict) -> dict:
    """
    available_assets = {
        "logos": ["logo.png", "brand.png"],
        "music": ["track1.mp3", "upbeat.mp3"]
    }
    """
    user_prompt = f"""
Campaign description: {description}

Available assets in assets/ folder:
Logos: {available_assets.get('logos', [])}
Music: {available_assets.get('music', [])}

Match mentioned assets to available files. If a specific file is not mentioned
but a type is (e.g., "add logo"), use the first available one.
Return only the JSON config.
"""
    if not LLM_API_KEY or LLM_API_KEY == "your_minimax_api_key_here":
        print("⚠️ LLM_API_KEY not set. Returning a mock response.")
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
        return mock_config


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
    response = requests.post(LLM_API_URL, json=payload, headers=headers)
    response.raise_for_status()

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
    return config
