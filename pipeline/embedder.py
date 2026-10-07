"""
pipeline/embedder.py

Generates and stores text embeddings for all transcript chunks of a video,
using a LOCAL SentenceTransformer model (no API calls, no rate limits).

Input:
    video_id (str):     UUID of the video row in the DB.
    chunks (list[dict]): Chunk dicts as returned by pipeline/chunker.build_chunks().
                         Each dict must contain 'id' (DB UUID) and 'text'.

Output:
    The same chunks list with 'embedding' (list[float]) populated on each dict.
    Each embedding vector is also persisted to the chunks table via chunk_repo.update_embedding().

Model:
    Default: models/Qwen3-Embedding-0.6B (native 1024-dim, cosine similarity).
    Switch models without code changes via the EMBED_MODEL env var
    (local dir or HuggingFace id). The model is loaded lazily once per
    process and reused for all chunks/clips.

Prompts:
    Stored vectors (chunks, clip hook+reason) use NO prefix (prompt_name=None).
    Search queries use prompt_name="query" (see analyzer.py clip embeddings).

Vector dimensions:
    Native 1024-dim, optionally truncated via EMBED_DIM (default 1024 =
    no-op slice). The DB schema must match — see db/schema.sql vector(1024).
"""

import gc
import time
from logger import get_logger

from config import get_config
from db.repositories import chunk_repo

log = get_logger("pipeline.embedder")

_MODEL = None
_MODEL_KEY = None


def _resolve_device(configured: str) -> str:
    configured = (configured or "auto").lower()
    if configured in ("cuda", "cpu"):
        return configured
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _resolve_dtype(configured: str | None, device: str):
    """Map EMBED_DTYPE to a torch dtype. auto → bfloat16 on CUDA, float32 on CPU."""
    import torch
    name = (configured or "auto").lower()
    if name == "auto":
        return torch.bfloat16 if device == "cuda" else torch.float32
    mapping = {
        "float32": torch.float32, "fp32": torch.float32,
        "float16": torch.float16, "fp16": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    }
    if name not in mapping:
        log.warning("unknown EMBED_DTYPE '%s', falling back to auto", configured)
        return torch.bfloat16 if device == "cuda" else torch.float32
    return mapping[name]


def free_gpu_memory() -> None:
    """Release cached GPU memory (e.g. WhisperX leftovers) before embedding."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            log.debug("freed GPU cache")
    except Exception:
        pass


def _get_model():
    """Load (once) and return the SentenceTransformer model."""
    global _MODEL, _MODEL_KEY
    cfg = get_config()["embeddings"]
    model_path = cfg.get("model", "models/Qwen3-Embedding-0.6B")
    device = _resolve_device(cfg.get("device", "auto"))
    dtype = _resolve_dtype(cfg.get("dtype", "auto"), device)
    max_seq = int(cfg.get("max_seq_length", 1024) or 1024)
    key = f"{model_path}|{device}|{dtype}|{max_seq}"
    if _MODEL is not None and _MODEL_KEY == key:
        return _MODEL
    from sentence_transformers import SentenceTransformer
    log.info("loading local embedding model path=%s device=%s dtype=%s max_seq=%d",
             model_path, device, dtype, max_seq)
    t0 = time.monotonic()
    # trust_remote_code needed for Qwen3 custom pooling/model code.
    # (transformers>=4.40 uses `dtype`; older versions use `torch_dtype`.)
    load_kwargs = dict(device=device, trust_remote_code=True)
    for dtype_key in ("dtype", "torch_dtype"):
        try:
            _MODEL = SentenceTransformer(
                model_path, model_kwargs={dtype_key: dtype}, **load_kwargs
            )
            break
        except TypeError:
            continue
    else:
        # very old sentence-transformers without model_kwargs support
        _MODEL = SentenceTransformer(model_path, **load_kwargs)
    _MODEL.max_seq_length = max_seq
    _MODEL_KEY = key
    if hasattr(_MODEL, "get_embedding_dimension"):
        dim = _MODEL.get_embedding_dimension()
    else:
        dim = _MODEL.get_sentence_embedding_dimension()
    log.info("embedding model ready native_dim=%s elapsed=%.1fs", dim, time.monotonic() - t0)
    return _MODEL


def get_embedding_dim() -> int | None:
    """Return the EFFECTIVE output dimension after EMBED_DIM truncation."""
    try:
        cfg = get_config()["embeddings"]
        native = (
            _get_model().get_embedding_dimension()
            if hasattr(_get_model(), "get_embedding_dimension")
            else _get_model().get_sentence_embedding_dimension()
        )
        return min(int(cfg.get("truncate_dim", native) or native), native)
    except Exception:
        log.exception("get_embedding_dim failed")
        return None


def _encode(texts: list[str], prompt_name: str | None) -> list[list[float]]:
    cfg = get_config()["embeddings"]
    model = _get_model()
    batch_size = int(cfg.get("batch_size", 8) or 8)
    normalize = bool(cfg.get("normalize", True))
    truncate_dim = int(cfg.get("truncate_dim", 0) or 0)
    kwargs = dict(
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    # Only pass prompt_name if the model actually defines it
    try:
        prompts = getattr(model, "prompts", None) or {}
        if prompt_name and prompt_name in prompts:
            kwargs["prompt_name"] = prompt_name
        elif prompt_name:
            log.debug("prompt '%s' not in model, encoding without prefix", prompt_name)
    except Exception:
        pass
    # Sort by length so batches pad evenly, then restore original order
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    sorted_texts = [texts[i] for i in order]
    sorted_vecs = model.encode(sorted_texts, **kwargs)
    if truncate_dim > 0:
        sorted_vecs = sorted_vecs[:, :truncate_dim]
    vecs = [None] * len(texts)
    for rank, idx in enumerate(order):
        vecs[idx] = sorted_vecs[rank].tolist()
    return vecs


def embed_text(text: str, prompt_name: str | None = None) -> list:
    """
    Embed a single text with the local model and return a float vector.

    Args:
        text:        The text string to embed.
        prompt_name: None for stored chunks/clips (no prefix),
                     "query" for search queries (clip hook+reason).

    Returns:
        list[float] — the embedding vector (1024-dim by default).
    """
    log.debug("embed_text chars=%d prompt=%s", len(text), prompt_name)
    vec = _encode([text], prompt_name=prompt_name)[0]
    log.debug("embed_text done dim=%d", len(vec))
    return vec


def embed_texts(texts: list[str], prompt_name: str | None = None) -> list[list[float]]:
    """Embed a batch of texts. Returns one vector per input, in order."""
    if not texts:
        return []
    log.debug("embed_texts count=%d prompt=%s", len(texts), prompt_name)
    return _encode(texts, prompt_name=prompt_name)


def embed_all_chunks(video_id: str, chunks: list) -> list:
    """
    Embed every chunk for a video and persist the vectors to the DB.

    Skips chunks that already have an embedding (idempotent — safe for re-runs).
    Encodes pending chunks in batches with the local model.

    Args:
        video_id: UUID string of the video (used only for logging).
        chunks:   List of chunk dicts (must have 'id' and 'text' keys).

    Returns:
        The same chunks list with 'embedding' key populated on each dict.
    """
    free_gpu_memory()
    cfg = get_config()["embeddings"]
    total = len(chunks)
    pending = [c for c in chunks if not c.get("embedding")]
    skipped = total - len(pending)
    t0 = time.monotonic()
    log.info("embed_all start video=%.8s total=%d pending=%d model=%s",
             video_id, total, len(pending), cfg.get("model"))

    if not pending:
        log.info("embed_all done video=%.8s nothing pending", video_id)
        return chunks

    # Warm up model once so load time is logged separately from encoding
    dim = get_embedding_dim()
    texts = [c["text"] for c in pending]
    log.info("encoding %d chunks dim=%s", len(texts), dim)
    vectors = _encode(texts, prompt_name=None)

    for chunk, vector in zip(pending, vectors):
        chunk_repo.update_embedding(chunk["id"], vector)
        chunk["embedding"] = vector
        log.debug("embedded chunk_idx=%s dim=%d", chunk.get("chunk_index"), len(vector))

    log.info("embed_all done video=%.8s embedded=%d skipped=%d elapsed=%.1fs",
             video_id, len(pending), skipped, time.monotonic() - t0)
    return chunks
