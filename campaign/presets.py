"""campaign/presets.py — export presets per platform.

Resolution comes from the deliverable's aspect ratio (ASPECT_SIZES); a preset
adds encoding settings and conservative limits used by the QA checks (C6).
Limits are deliberately conservative defaults, NOT official platform specs —
platforms change them; adjust here when a campaign states its own limits.
"""

ASPECT_SIZES = {
    "9:16": (1080, 1920),
    "4:5": (1080, 1350),
    "1:1": (1080, 1080),
    "16:9": (1920, 1080),
}

PLATFORMS = {
    "generic": {"label": "Universal vertical", "aspect": "9:16", "max_duration": 180,
                "max_size_mb": 250},
    "tiktok": {"label": "TikTok", "aspect": "9:16", "max_duration": 180, "max_size_mb": 250},
    "reels": {"label": "Instagram Reels", "aspect": "9:16", "max_duration": 90,
              "max_size_mb": 250},
    "shorts": {"label": "YouTube Shorts", "aspect": "9:16", "max_duration": 180,
               "max_size_mb": 250},
    "youtube": {"label": "YouTube (landscape)", "aspect": "16:9", "max_duration": 3600,
                "max_size_mb": 2000},
    "x": {"label": "X / Twitter", "aspect": "9:16", "max_duration": 140, "max_size_mb": 500},
    "facebook": {"label": "Facebook Reels", "aspect": "9:16", "max_duration": 90,
                 "max_size_mb": 250},
}

# encoding shared by all presets (H.264 + AAC in MP4 plays everywhere)
ENCODING = {"video_codec": "libx264", "preset": "fast", "crf": 20, "pix_fmt": "yuv420p",
            "audio_codec": "aac", "audio_bitrate": "192k", "max_fps": 60,
            "loudness_lufs": -14.0}

ALIASES = {"instagram": "reels", "ig": "reels", "insta": "reels", "yt": "shorts",
           "youtube shorts": "shorts", "twitter": "x", "fb": "facebook", "tik tok": "tiktok"}


def normalize_platform(name: str) -> str | None:
    n = str(name or "").strip().lower()
    n = ALIASES.get(n, n)
    return n if n in PLATFORMS else None


def output_size(aspect: str) -> tuple:
    return ASPECT_SIZES.get(aspect, ASPECT_SIZES["9:16"])


def preset(platform: str) -> dict:
    p = PLATFORMS.get(normalize_platform(platform) or "generic")
    return {**ENCODING, **p, "platform": normalize_platform(platform) or "generic"}
