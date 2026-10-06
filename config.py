# config.py
"""
Configuration module for Viral Clips Automator.

Exposes both:
  - Provider-agnostic LLM aliases (LLM_API_KEY, LLM_MODEL, LLM_API_URL) plus
    WhisperX settings, used by pipeline modules.
  - get_config() — reads config.yaml and merges .env overrides, used by all new modules
    (db/connection.py, pipeline/chunker.py, pipeline/embedder.py, pipeline/similarity.py).

Provider: Groq (OpenAI-compatible) for LLM + local SentenceTransformer for embeddings.
  LLM Base URL: https://api.groq.com/openai/v1
  LLM model:    configurable via GROQ_MODEL env var (default: openai/gpt-oss-120b)
  Embed model:  local path via EMBED_MODEL env var (default: models/Qwen3-Embedding-0.6B)
"""
import os
import shutil
import yaml
from dotenv import load_dotenv

load_dotenv()

# ── Groq LLM ───────────────────────────────────────────────────────────────────
# Paste your key into .env as GROQ_API_KEY. Model name is also set via .env
# so you can swap models without touching code.
# (GROK_API_KEY / XAI_API_KEY accepted as legacy aliases from the xAI chapter.)
GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY", os.getenv("GROK_API_KEY", os.getenv("XAI_API_KEY", ""))
)

GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_LLM_MODEL = os.getenv(
    "GROQ_MODEL", os.getenv("GROK_MODEL", os.getenv("GROK_LLM_MODEL", "openai/gpt-oss-120b"))
)

# ── Local embeddings (SentenceTransformer, no API key needed) ─────────────────
# Directory or HF id of the embedding model. EMBED_MODEL_PATH is accepted as a
# legacy alias. Device: "auto" (CUDA if available, else CPU), or "cuda"/"cpu".
# EMBED_DIM truncates vectors (Matryoshka-style slice, must match DB vector(N)).
EMBED_MODEL = os.getenv("EMBED_MODEL", os.getenv("EMBED_MODEL_PATH", "models/Qwen3-Embedding-0.6B"))
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "auto")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "8"))
EMBED_DTYPE = os.getenv("EMBED_DTYPE", "auto")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
EMBED_MAX_SEQ = int(os.getenv("EMBED_MAX_SEQ", "1024"))

# ── Provider-agnostic LLM aliases (used by all pipeline modules) ─────────────
# analyzer.py, campaign_parser.py, similarity.py all do:
#   from config import LLM_API_KEY, LLM_MODEL, LLM_API_URL
LLM_API_KEY = GROQ_API_KEY
LLM_MODEL = GROQ_LLM_MODEL
LLM_API_URL = f"{GROQ_BASE_URL.rstrip('/')}/chat/completions"

# WhisperX settings (defaults; config.yaml [transcription] + WHISPER_* env win)
WHISPER_MODEL    = "base"
WHISPER_LANGUAGE = "en"
WHISPER_DEVICE   = "cpu"
WHISPER_COMPUTE  = "float32"


# ── New: get_config() reads config.yaml + env overrides ──────────────────────

_CONFIG_PATH   = os.path.join(os.path.dirname(__file__), "config.yaml")
_cached_config = None


def get_config() -> dict:
    """
    Load and return the full configuration dictionary from config.yaml,
    with environment variable overrides applied for sensitive values.

    Returns a dict with keys: ffmpeg, paths, ai, defaults, database,
    chunking, embeddings, similarity, logging.
    """
    global _cached_config
    if _cached_config is not None:
        return _cached_config

    with open(_CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    # Groq LLM overrides: env vars win over config.yaml so models can be swapped
    # from .env without touching code.
    ai_cfg = dict(cfg.get("ai", {}))
    ai_cfg["provider"] = "groq"
    ai_cfg["llm_model"] = GROQ_LLM_MODEL
    ai_cfg["llm_endpoint"] = LLM_API_URL

    # Local embeddings: model path / device / batch size / truncation from .env.
    yaml_emb = cfg.get("embeddings", {}) or {}
    emb_cfg = dict(yaml_emb)
    emb_cfg["provider"] = "local"
    emb_cfg["model"] = os.getenv("EMBED_MODEL", os.getenv("EMBED_MODEL_PATH", yaml_emb.get("model", EMBED_MODEL)))
    emb_cfg["device"] = os.getenv("EMBED_DEVICE", yaml_emb.get("device", EMBED_DEVICE))
    emb_cfg["batch_size"] = int(os.getenv("EMBED_BATCH_SIZE", yaml_emb.get("batch_size", EMBED_BATCH_SIZE)))
    emb_cfg["dtype"] = os.getenv("EMBED_DTYPE", yaml_emb.get("dtype", EMBED_DTYPE))
    emb_cfg["truncate_dim"] = int(os.getenv("EMBED_DIM", yaml_emb.get("truncate_dim", EMBED_DIM)))
    emb_cfg["max_seq_length"] = int(os.getenv("EMBED_MAX_SEQ", yaml_emb.get("max_seq_length", EMBED_MAX_SEQ)))
    emb_cfg["normalize"] = bool(yaml_emb.get("normalize", True))
    # Drop provider-specific extras from previous providers (e.g. NVIDIA/Grok
    # endpoint/input_type) — local embeddings need none of them.
    emb_cfg.pop("endpoint", None)
    emb_cfg.pop("input_type_storage", None)
    emb_cfg.pop("input_type_query", None)

    _cached_config = {
        "ffmpeg":   cfg["ffmpeg"],
        "paths":    cfg["paths"],
        "ai":       ai_cfg,
        "defaults": cfg["defaults"],
        "database": {
            "host":     os.getenv("DB_HOST", cfg["database"]["host"]),
            "port":     int(os.getenv("DB_PORT", cfg["database"].get("port", 5432))),
            "name":     os.getenv("DB_NAME", cfg["database"]["name"]),
            "user":     os.getenv("DB_USER", cfg["database"]["user"]),
            "password": os.getenv("DB_PASSWORD", cfg["database"].get("password", "")),
        },
        "chunking":   cfg["chunking"],
        "embeddings": emb_cfg,
        "outline":    cfg.get("outline", {}) or {},
        "discovery":  cfg.get("discovery", {}) or {},
        "audio_events": cfg.get("audio_events", {}) or {},
        "fusion":     cfg.get("fusion", {}) or {},
        "captions":   cfg.get("captions", {}) or {},
        "audio":      cfg.get("audio", {}) or {},
        "export":     cfg.get("export", {}) or {},
        "speakers":   cfg.get("speakers", {}) or {},
        "visual":     cfg.get("visual", {}) or {},
        "transcription": {
            "model": os.getenv("WHISPER_MODEL",
                               cfg.get("transcription", {}).get("model", WHISPER_MODEL)),
            "device": os.getenv("WHISPER_DEVICE",
                                cfg.get("transcription", {}).get("device", WHISPER_DEVICE)),
            "compute_type": os.getenv(
                "WHISPER_COMPUTE",
                cfg.get("transcription", {}).get("compute_type", WHISPER_COMPUTE)),
            "language": os.getenv("WHISPER_LANGUAGE",
                                  cfg.get("transcription", {}).get("language", WHISPER_LANGUAGE)),
            "glossary": cfg.get("transcription", {}).get("glossary", []) or [],
        },
        "similarity": cfg["similarity"],
        "refine":     cfg.get("refine", {"enabled": True, "context_seconds": 20,
                                         "max_extension_seconds": 15,
                                         "min_duration": 20, "max_duration": 90,
                                         "start_padding": 0.15, "end_padding": 0.3,
                                         "llm_validation": True}),
        "framing":    cfg.get("framing", {"default_layout": "auto"}),
        "fades":      cfg.get("fades", {"video": True, "audio": False,
                                        "duration": 0.5}),
        "logging":    cfg.get("logging", {"level": "INFO",
                                          "file": "logs/viral-clips.log",
                                          "console": True}),
    }
    return _cached_config


def _resolve_bin(name: str, configured: str | None = None) -> str:
    """Resolve an executable path: FFMPEG_PATH-style env > config value > PATH.

    Returns a usable path, falling back to the bare name (PATH lookup at
    exec time) when nothing on disk matches.
    """
    candidates = []
    for raw in (os.getenv("FFMPEG_PATH"), configured):
        if raw and raw != name:
            candidates.append(os.path.expanduser(raw))
            # allow a directory (yt-dlp --ffmpeg-location style)
            if os.path.isdir(os.path.expanduser(raw)):
                candidates.append(os.path.join(os.path.expanduser(raw), name))
    which = shutil.which(name)
    if which:
        candidates.append(which)
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return name


def ffmpeg_path() -> str:
    """Path to the ffmpeg binary (FFMPEG_PATH env > config.yaml > PATH)."""
    try:
        configured = get_config().get("ffmpeg", {}).get("path")
    except Exception:
        configured = None
    return _resolve_bin("ffmpeg", configured)


def ffprobe_path() -> str:
    """Path to the ffprobe binary (same dir as ffmpeg when possible)."""
    ff = ffmpeg_path()
    sibling = os.path.join(os.path.dirname(ff), "ffprobe") if os.path.dirname(ff) else None
    if sibling and os.path.exists(sibling):
        return sibling
    if sibling and os.path.dirname(ff):
        win_sibling = sibling + ".exe"
        if os.path.exists(win_sibling):
            return win_sibling
    return _resolve_bin("ffprobe")