"""
pipeline/llm_client.py

Central Groq chat-completions caller with usage-based key slots + fallback.

Workload split (measured on a 58-min video, ~170K LLM tokens/run;
gpt-oss-120b free tier = 200K TPD / 8K TPM per key):
  slot 1 (heavy,  ~100K): outline + discovery + audio-event enrich
  slot 2 (medium, ~35K):  similarity classification
  slot 3 (light,  ~35K):  campaign parsing + analyzer + boundary validation
  slot 4 (fallback):      tried whenever the slot key fails (429/4xx/5xx/
                          network). Remaining pool keys are tried after it.

Key order per call: [slot, fallback, rest-of-pool...], empties skipped,
duplicates collapsed. A 429 moves to the next key immediately (different
quota bucket — no sleep); callers keep their own backoff/retry loops, which
now trigger only when EVERY key is exhausted. 4xx (dead key/model) also
rotates instead of pointlessly retrying the same key.

Key labels (key1..key4) appear in logs; key VALUES never do.

No keys configured -> raises requests ConnectionError, which every caller
already handles via its offline/mock fallback path.
"""

from logger import get_logger

import requests

log = get_logger("pipeline.llm_client")

SLOT_HEAVY = 1    # outline + discovery + audio enrich
SLOT_MATCH = 2    # similarity
SLOT_LIGHT = 3    # campaign + analyzer + boundaries
SLOT_FALLBACK = 4

SLOT_FOR_STAGE = {
    "outline": SLOT_HEAVY,
    "discovery": SLOT_HEAVY,
    "enrich": SLOT_HEAVY,
    "similarity": SLOT_MATCH,
    "campaign": SLOT_LIGHT,
    "analyzer": SLOT_LIGHT,
    "boundaries": SLOT_LIGHT,
}

_DEAD_STATUSES = (400, 401, 403, 404)

_PLACEHOLDERS = {
    "", "your_groq_api_key_here", "your_grok_api_key_here",
    "your_nvidia_api_key_here", "your_minimax_api_key_here",
    "your_groq_api_key_1_here", "your_groq_api_key_2_here",
    "your_groq_api_key_3_here", "your_groq_api_key_4_here",
    "your_groq_api_key_fallback_here",
}


def has_keys():
    """True when at least one key slot is non-empty."""
    return bool(key_pool())


def pool_usable():
    """True when at least one slot holds a non-placeholder key."""
    return any(k not in _PLACEHOLDERS for _, k in key_pool())


def key_pool():
    """Ordered [(label, key)] for slots 1..4, empties dropped, dupes collapsed."""
    import os
    from config import (GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3,
                        GROQ_API_KEY_FALLBACK)
    # os.getenv re-read keeps module-attr monkeypatching working in tests
    raw = [
        ("key1", os.getenv("GROQ_API_KEY", GROQ_API_KEY)),
        ("key2", os.getenv("GROQ_API_KEY_2", GROQ_API_KEY_2)),
        ("key3", os.getenv("GROQ_API_KEY_3", GROQ_API_KEY_3)),
        ("key4", os.getenv("GROQ_API_KEY_FALLBACK",
                           os.getenv("GROQ_API_KEY_4", GROQ_API_KEY_FALLBACK))),
    ]
    pool, seen = [], set()
    for label, key in raw:
        if key and key not in _PLACEHOLDERS and key not in seen:
            seen.add(key)
            pool.append((label, key))
    return pool


def _ordered_keys(slot):
    pool = dict(key_pool())
    labels = [f"key{slot}", "key4", "key1", "key2", "key3"]
    ordered = []
    for label in labels:
        if label in pool and label not in [lb for lb, _ in ordered]:
            ordered.append((label, pool[label]))
    return ordered


def post_chat(payload, timeout, slot, purpose="llm"):
    """POST one chat-completions call, rotating keys on failure.

    Returns the requests.Response on the first 2xx. Raises the last
    HTTPError when every key fails (429 everywhere included), or
    requests ConnectionError when no keys are configured.
    """
    from config import LLM_API_URL
    keys = _ordered_keys(slot)
    if not keys:
        raise requests.exceptions.ConnectionError(
            "no GROQ_API_KEYs configured (key1..key4 all empty)")
    last = None
    for label, key in keys:
        try:
            r = requests.post(
                LLM_API_URL,
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"},
                json=payload, timeout=timeout)
        except requests.exceptions.RequestException as e:
            log.warning("%s network error on %s, trying next key: %s",
                        purpose, label, e)
            last = e
            continue
        if r.status_code == 429:
            log.warning("%s 429 rate-limited on %s, trying next key",
                        purpose, label)
            last = requests.exceptions.HTTPError(
                f"429 rate-limited on {label}", response=r)
            continue
        if r.status_code in _DEAD_STATUSES:
            log.error("%s HTTP %s on %s (dead key/model), trying next key",
                      purpose, r.status_code, label)
            last = requests.exceptions.HTTPError(
                f"{r.status_code} on {label}", response=r)
            continue
        try:
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            log.warning("%s HTTP error on %s, trying next key: %s",
                        purpose, label, e)
            last = e
            continue
        if label != f"key{slot}":
            log.info("%s succeeded on fallback %s", purpose, label)
        else:
            log.debug("%s succeeded on %s", purpose, label)
        return r
    raise last
