"""
pipeline/embedder.py

Generates and stores text embeddings for all transcript chunks of a video.

Input:
    video_id (str):     UUID of the video row in the DB.
    chunks (list[dict]): Chunk dicts as returned by pipeline/chunker.build_chunks().
                         Each dict must contain 'id' (DB UUID) and 'text'.

Output:
    The same chunks list with 'embedding' (list[float]) populated on each dict.
    Each embedding vector is also persisted to the chunks table via chunk_repo.update_embedding().

API:
    Uses the NVIDIA NIM embeddings endpoint (OpenAI-compatible).
    Same NVIDIA API key used for all LLM calls throughout the project.
    Endpoint and model are read from config.yaml so they can be changed without touching code.

Rate limiting:
    NVIDIA NIM free tier allows ~40 RPM. This module adds a configurable delay between
    requests (default 1.5s) and retries up to 3 times on 429 / rate-limit responses.

Vector dimensions:
    nvidia/llama-nemotron-embed-1b-v2 produces 4096-dimensional vectors.
    The DB schema must have vector(4096) — see db/schema.sql.
"""

import time
import requests

from config import NVIDIA_API_KEY, get_config
from db.repositories import chunk_repo


def embed_text(text: str, endpoint: str = None, model: str = None) -> list:
    """
    Call the NVIDIA NIM embeddings API for a single text and return a float vector.

    Uses OpenAI-compatible format:
        POST /v1/embeddings
        body: {"model": ..., "input": ..., "encoding_format": "float"}
        response: {"data": [{"embedding": [...]}]}

    Args:
        text:     The text string to embed.
        endpoint: Override the endpoint URL (defaults to config.yaml value).
        model:    Override the model name (defaults to config.yaml value).

    Returns:
        list[float] — the embedding vector.

    Raises:
        RuntimeError if all retries are exhausted.
        requests.HTTPError on non-rate-limit HTTP errors.
    """
    cfg      = get_config()["embeddings"]
    endpoint = endpoint or cfg["endpoint"]
    model    = model    or cfg["model"]
    input_type = cfg.get("input_type_storage", "passage")

    max_retries = 3
    for attempt in range(max_retries):
        response = requests.post(
            endpoint,
            headers={
                "Authorization":  f"Bearer {NVIDIA_API_KEY}",
                "Content-Type":   "application/json",
            },
            json={
                "model":            model,
                "input":            [text],        # list required by NVIDIA NIM asymmetric models
                "input_type":       input_type,    # 'passage' for storage, 'query' for search
                "encoding_format":  "float",
                "truncate":         "END",
            },
            timeout=60,
        )

        # 429 = rate limited
        if response.status_code == 429:
            wait_secs = 20 * (attempt + 1)
            print(f"[embedder] Rate limited (attempt {attempt + 1}/{max_retries}) — waiting {wait_secs}s...")
            time.sleep(wait_secs)
            continue

        response.raise_for_status()
        data = response.json()

        embeddings = data.get("data", [])
        if not embeddings:
            raise RuntimeError(f"Empty embeddings response: {data}")

        return embeddings[0]["embedding"]

    raise RuntimeError(f"Embeddings API rate limit exceeded after {max_retries} retries for text: {text[:60]}...")


def embed_all_chunks(video_id: str, chunks: list, delay_seconds: float = 1.5) -> list:
    """
    Embed every chunk for a video and persist the vectors to the DB.

    Skips chunks that already have an embedding (idempotent — safe for re-runs).
    Adds a small delay between requests to avoid hitting the RPM rate limit.

    Args:
        video_id:      UUID string of the video (used only for logging).
        chunks:        List of chunk dicts (must have 'id' and 'text' keys).
        delay_seconds: Pause between API calls (default 1.5s).

    Returns:
        The same chunks list with 'embedding' key populated on each dict.
    """
    total    = len(chunks)
    embedded = 0
    skipped  = 0

    for chunk in chunks:
        if chunk.get("embedding"):
            skipped += 1
            continue   # already embedded — skip (idempotent re-run safety)

        print(f"[embedder] Embedding chunk {chunk['chunk_index'] + 1}/{total} "
              f"({chunk['start_time']:.1f}s → {chunk['end_time']:.1f}s) ...")

        vector = embed_text(chunk["text"])
        chunk_repo.update_embedding(chunk["id"], vector)
        chunk["embedding"] = vector
        embedded += 1

        if embedded < (total - skipped):
            time.sleep(delay_seconds)   # rate-limit guard between requests

    print(f"[embedder] ✅ Done — embedded {embedded} chunks, skipped {skipped} (already done).")
    return chunks
