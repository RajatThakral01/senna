# config.py
"""
Configuration module for Viral Clips Automator.

Exposes both:
  - Legacy flat constants (AI_API_KEY, AI_MODEL, etc.) for backward compatibility.
    These are aliased so existing modules (analyzer.py, campaign_parser.py, subtitler.py)
    continue to work without changes to their import lines.
  - get_config() — reads config.yaml and merges .env overrides, used by all new modules
    (db/connection.py, pipeline/chunker.py, pipeline/embedder.py, pipeline/similarity.py).

Provider: NVIDIA NIM (OpenAI-compatible)
  Base URL:   https://integrate.api.nvidia.com/v1
  LLM model:  moonshotai/kimi-k2.6
  Embed model: nvidia/llama-nemotron-embed-1b-v2
"""
import os
import yaml
from dotenv import load_dotenv

load_dotenv()

# ── NVIDIA NIM credentials ─────────────────────────────────────────────────────
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")

NVIDIA_BASE_URL  = "https://integrate.api.nvidia.com/v1"
NVIDIA_LLM_MODEL = "moonshotai/kimi-k2.6"
NVIDIA_EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"  # 1024-dim, fits pgvector HNSW index

# ── Provider-agnostic LLM aliases (used by all pipeline modules) ─────────────
# analyzer.py, campaign_parser.py, subtitler.py, similarity.py all do:
#   from config import LLM_API_KEY, LLM_MODEL, LLM_API_URL
LLM_API_KEY = NVIDIA_API_KEY
LLM_MODEL   = NVIDIA_LLM_MODEL
LLM_API_URL = f"{NVIDIA_BASE_URL}/chat/completions"

# WhisperX settings
WHISPER_MODEL    = "base"
WHISPER_LANGUAGE = "en"
WHISPER_DEVICE   = "cpu"

# Clip settings
MIN_CLIP_DURATION = 30
MAX_CLIP_DURATION = 90
NUM_CLIPS         = 3

# Subtitle style (for FFmpeg)
SUBTITLE_FONT          = "Arial"
SUBTITLE_FONTSIZE      = 18
SUBTITLE_COLOR         = "white"
SUBTITLE_OUTLINE_COLOR = "black"
SUBTITLE_OUTLINE_WIDTH = 2


# ── New: get_config() reads config.yaml + env overrides ──────────────────────

_CONFIG_PATH   = os.path.join(os.path.dirname(__file__), "config.yaml")
_cached_config = None


def get_config() -> dict:
    """
    Load and return the full configuration dictionary from config.yaml,
    with environment variable overrides applied for sensitive values.

    Returns a dict with keys: ffmpeg, paths, ai, defaults, database,
    chunking, embeddings, similarity.
    """
    global _cached_config
    if _cached_config is not None:
        return _cached_config

    with open(_CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    _cached_config = {
        "ffmpeg":   cfg["ffmpeg"],
        "paths":    cfg["paths"],
        "ai":       cfg["ai"],
        "defaults": cfg["defaults"],
        "database": {
            "host":     os.getenv("DB_HOST", cfg["database"]["host"]),
            "port":     int(os.getenv("DB_PORT", cfg["database"].get("port", 5432))),
            "name":     os.getenv("DB_NAME", cfg["database"]["name"]),
            "user":     os.getenv("DB_USER", cfg["database"]["user"]),
            "password": os.getenv("DB_PASSWORD", cfg["database"].get("password", "")),
        },
        "chunking":   cfg["chunking"],
        "embeddings": cfg["embeddings"],
        "similarity": cfg["similarity"],
    }
    return _cached_config