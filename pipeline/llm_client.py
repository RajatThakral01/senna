"""
pipeline/llm_client.py

Central LLM caller: Claude (Anthropic) primary, Groq fallback.

Provider order per call (post_chat):
  1. Claude, when ANTHROPIC_API_KEY is set and LLM_PROVIDER != "groq".
     Purposes in CLAUDE_SONNET_STAGES (default discovery, analyzer) go to
     Sonnet 5.5; every other call goes to Haiku 5.5. The OpenAI-style payload
     is translated to the Messages API and the reply is wrapped back into the
     chat-completions shape, so callers keep reading
     r.json()["choices"][0]["message"]["content"] unchanged.
  2. Groq (below), when Claude fails: network error, 5xx/overloaded, 429
     after the SDK's own retries, auth error, refusal, or empty reply.
     A connection/5xx/overload failure also opens a short circuit-breaker
     (CLAUDE_COOLDOWN_SECONDS, default 60) so a Claude outage doesn't add
     retry latency to every call; an auth failure disables Claude for the
     rest of the process.

Groq chat-completions caller with usage-based key slots + fallback.

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

import os
import re
import time

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


_CLAUDE_PLACEHOLDERS = {"", "your_anthropic_api_key_here"}

# Claude circuit-breaker: skip Claude until this time.time() after an outage.
_claude_down_until = 0.0
_claude_clients = {}


def has_keys():
    """True when at least one key slot is non-empty."""
    return bool(key_pool())


def claude_configured():
    """True when Claude should be tried first (key set, provider not forced to groq)."""
    from config import ANTHROPIC_API_KEY, LLM_PROVIDER
    if os.getenv("LLM_PROVIDER", LLM_PROVIDER).strip().lower() == "groq":
        return False
    return os.getenv("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY) not in _CLAUDE_PLACEHOLDERS


def pool_usable():
    """True when any LLM is callable: Claude key, or a non-placeholder Groq key."""
    return claude_configured() or any(k not in _PLACEHOLDERS for _, k in key_pool())


def claude_model_for(purpose):
    """Sonnet for the configured high-value purposes, Haiku for the rest."""
    from config import (CLAUDE_SONNET_MODEL, CLAUDE_HAIKU_MODEL,
                        CLAUDE_SONNET_STAGES)
    stages = {x.strip() for x in os.getenv(
        "CLAUDE_SONNET_STAGES", CLAUDE_SONNET_STAGES).split(",") if x.strip()}
    if purpose in stages:
        return os.getenv("CLAUDE_SONNET_MODEL", CLAUDE_SONNET_MODEL)
    return os.getenv("CLAUDE_HAIKU_MODEL", CLAUDE_HAIKU_MODEL)


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


def _parse_duration(text):
    """Seconds from Groq duration strings: "55.5s", "1m2.5s", "450ms"."""
    total, found = 0.0, False
    for num, unit in re.findall(r"([0-9]*\.?[0-9]+)\s*(ms|h|m|s)", str(text or "")):
        found = True
        total += float(num) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
    return total if found else None


def _rate_wait(r):
    """(seconds_to_wait, is_daily_limit) for a 429 response."""
    msg = ""
    try:
        msg = str(((r.json() or {}).get("error") or {}).get("message", ""))
    except Exception:
        pass
    daily = "per day" in msg.lower() or "(tpd)" in msg.lower() or "(rpd)" in msg.lower()
    hdr = getattr(r, "headers", {}) or {}
    cands = []
    m = re.search(r"try again in ([0-9hms.]+)", msg)
    if m:
        cands.append(_parse_duration(m.group(1)))
    for h in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        v = hdr.get(h)
        if v:
            d = _parse_duration(v) if not str(v).replace(".", "", 1).isdigit() else float(v)
            cands.append(d)
    cands = [c for c in cands if c is not None and c >= 0]
    wait = min(cands) if cands else 20.0
    return min(max(wait + 0.5, 2.0), 65.0), daily


def _is_reasoning_model(model):
    m = str(model or "").lower()
    return any(k in m for k in ("gpt-oss", "qwen3", "deepseek-r1", "reasoning"))


def _prepare_payload(payload, json_mode):
    """Copy of payload with reasoning headroom and optional JSON mode.

    Reasoning models (gpt-oss) spend hidden reasoning tokens out of the same
    max_tokens budget as the answer; call sites size max_tokens for the
    ANSWER, so a long reasoning pass truncated the JSON mid-object (outline
    windows failing with "Expecting ',' delimiter"). LLM_REASONING_HEADROOM
    (default 4000) is added on top for those models.
    """
    p = dict(payload)
    if p.get("max_tokens") and _is_reasoning_model(p.get("model")):
        p["max_tokens"] = int(p["max_tokens"]) + int(
            os.getenv("LLM_REASONING_HEADROOM", "4000"))
    if json_mode:
        p["response_format"] = {"type": "json_object"}
    return p


def _is_json_mode_rejection(r):
    try:
        err = (r.json() or {}).get("error") or {}
    except Exception:
        return False
    return r.status_code == 400 and (
        err.get("code") == "json_validate_failed"
        or "json" in str(err.get("message", "")).lower())


def _post_groq(payload, timeout, slot, purpose="llm", json_mode=False):
    """POST one Groq chat-completions call, rotating keys on failure.

    Returns the requests.Response on the first 2xx. Raises the last
    HTTPError when every key fails (429 everywhere included), or
    requests ConnectionError when no keys are configured.

    json_mode=True asks the API for a strict JSON object
    (response_format json_object). If the API rejects the model's output as
    invalid JSON (HTTP 400 json_validate_failed) the same key is retried
    once without JSON mode instead of being treated as a dead key.
    A response cut off at max_tokens (finish_reason "length") is logged.
    """
    from config import LLM_API_URL
    payload = _prepare_payload(payload, json_mode)
    keys = _ordered_keys(slot)
    if not keys:
        raise requests.exceptions.ConnectionError(
            "no GROQ_API_KEYs configured (key1..key4 all empty)")
    max_wait = float(os.getenv("LLM_RATE_LIMIT_MAX_WAIT", "300"))
    waited = 0.0
    while True:
        last, waits = None, []
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
                waits.append(_rate_wait(r))
                log.warning("%s 429 rate-limited on %s, trying next key",
                            purpose, label)
                last = requests.exceptions.HTTPError(
                    f"429 rate-limited on {label}", response=r)
                continue
            if "response_format" in payload and _is_json_mode_rejection(r):
                log.warning("%s JSON mode rejected the output on %s; retrying "
                            "without JSON mode", purpose, label)
                payload = {k: v for k, v in payload.items() if k != "response_format"}
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
                    waits.append(_rate_wait(r))
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
            try:
                fin = (r.json().get("choices") or [{}])[0].get("finish_reason")
            except Exception:
                fin = None
            if fin == "length":
                log.warning("%s response truncated at max_tokens=%s (finish_reason=length)",
                            purpose, payload.get("max_tokens"))
            if label != f"key{slot}":
                log.info("%s succeeded on fallback %s", purpose, label)
            else:
                log.debug("%s succeeded on %s", purpose, label)
            return r
        # every key rate-limited on a per-minute budget: wait for the
        # reset Groq reports, then go round the keys again (free-tier keys
        # allow ~1 large call per key per minute). Daily quotas don't reset
        # in time, so those fail straight away.
        if waits and len(waits) == len(keys) and not any(d for _, d in waits):
            w = min(t for t, _ in waits)
            if waited + w <= max_wait:
                log.warning("%s: all %d keys rate-limited; waiting %.1fs for the "
                            "per-minute budget to reset", purpose, len(keys), w)
                time.sleep(w)
                waited += w
                continue
        raise last



# ── Claude (primary) ─────────────────────────────────────────────────────────

class ClaudeChatResponse:
    """Minimal requests.Response stand-in carrying a chat-completions body.

    Callers only use .json()["choices"][0]["message"]["content"]; status_code,
    text and raise_for_status() are provided for code that inspects them.
    """

    status_code = 200

    def __init__(self, content, finish_reason, model, usage):
        self._body = {
            "model": model,
            "choices": [{"index": 0, "finish_reason": finish_reason,
                         "message": {"role": "assistant", "content": content}}],
            "usage": usage,
        }
        self.headers = {}

    def json(self):
        return self._body

    @property
    def text(self):
        import json as _json
        return _json.dumps(self._body)

    def raise_for_status(self):
        return None


_JSON_INSTRUCTION = ("Respond with a single valid JSON object only: no prose, "
                     "no markdown, no code fences.")


def _claude_client(timeout):
    import anthropic
    from config import ANTHROPIC_API_KEY
    key = os.getenv("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY)
    retries = int(os.getenv("CLAUDE_MAX_RETRIES", "2"))
    ck = (key, retries)
    if ck not in _claude_clients:
        _claude_clients[ck] = anthropic.Anthropic(api_key=key, max_retries=retries)
    return _claude_clients[ck].with_options(timeout=float(timeout))


def _to_claude_request(payload, json_mode, model):
    """OpenAI chat payload -> Messages API kwargs.

    temperature is dropped: Sonnet 5.5 / Haiku 5.5 reject non-default
    sampling values. Adaptive thinking spends from max_tokens, and call
    sites size max_tokens for the ANSWER, so CLAUDE_THINKING_HEADROOM
    (default 4000) is added on top.
    """
    system_parts, messages = [], []
    for m in payload.get("messages") or []:
        if m.get("role") == "system":
            system_parts.append(str(m.get("content") or ""))
        else:
            messages.append({"role": m.get("role", "user"),
                             "content": m.get("content") or ""})
    if json_mode:
        system_parts.append(_JSON_INSTRUCTION)
    answer_tokens = int(payload.get("max_tokens") or 1024)
    req = {
        "model": model,
        "max_tokens": answer_tokens + int(os.getenv("CLAUDE_THINKING_HEADROOM", "4000")),
        "messages": messages,
        "thinking": {"type": "adaptive"},
    }
    if system_parts:
        req["system"] = "\n\n".join(system_parts)
    return req


def _claude_effort(model):
    from config import CLAUDE_SONNET_MODEL
    if model == os.getenv("CLAUDE_SONNET_MODEL", CLAUDE_SONNET_MODEL):
        return os.getenv("CLAUDE_SONNET_EFFORT", "medium")
    return os.getenv("CLAUDE_HAIKU_EFFORT", "medium")


def _post_claude(payload, timeout, purpose, json_mode):
    """One Claude call. Returns ClaudeChatResponse; raises on any failure."""
    from config import CLAUDE_SONNET_MODEL
    model = claude_model_for(purpose)
    req = _to_claude_request(payload, json_mode, model)
    req["output_config"] = {"effort": _claude_effort(model)}
    client = _claude_client(timeout)
    if model == os.getenv("CLAUDE_SONNET_MODEL", CLAUDE_SONNET_MODEL):
        # Server-side refusal fallback: a safety-classifier decline is
        # re-run on a fallback model inside the same call.
        msg = client.beta.messages.create(
            betas=["server-side-fallback-2026-07-01"], fallbacks="default", **req)
    else:
        msg = client.messages.create(**req)

    if msg.stop_reason == "refusal":
        cat = getattr(getattr(msg, "stop_details", None), "category", None)
        raise RuntimeError(f"Claude refusal (category={cat})")
    text = "".join(b.text for b in msg.content if b.type == "text").strip()
    if not text:
        raise RuntimeError(f"Claude returned no text (stop_reason={msg.stop_reason})")
    finish = "length" if msg.stop_reason == "max_tokens" else "stop"
    if finish == "length":
        log.warning("%s Claude response truncated at max_tokens=%s",
                    purpose, req["max_tokens"])
    u = msg.usage
    usage = {"prompt_tokens": u.input_tokens, "completion_tokens": u.output_tokens,
             "total_tokens": u.input_tokens + u.output_tokens}
    log.debug("%s served by %s (in=%d out=%d)", purpose, msg.model,
              u.input_tokens, u.output_tokens)
    return ClaudeChatResponse(text, finish, msg.model, usage)


def _note_claude_failure(e, purpose):
    """Log a Claude failure and open the circuit-breaker when Claude looks down."""
    global _claude_down_until
    import anthropic
    if isinstance(e, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        log.error("%s Claude auth failed (%s); using Groq for the rest of this run",
                  purpose, type(e).__name__)
        _claude_down_until = float("inf")
    elif isinstance(e, (anthropic.APIConnectionError, anthropic.InternalServerError,
                        anthropic.RateLimitError)) or (
            isinstance(e, anthropic.APIStatusError) and e.status_code >= 500):
        cool = float(os.getenv("CLAUDE_COOLDOWN_SECONDS", "60"))
        log.warning("%s Claude unavailable (%s); falling back to Groq for %.0fs",
                    purpose, type(e).__name__, cool)
        _claude_down_until = time.time() + cool
    else:
        log.warning("%s Claude call failed (%s: %s); falling back to Groq",
                    purpose, type(e).__name__, e)


def post_chat(payload, timeout, slot, purpose="llm", json_mode=False):
    """One chat call: Claude first (if configured), Groq as fallback.

    Returns an object exposing .json() in chat-completions shape (a
    ClaudeChatResponse or a Groq requests.Response). When Claude fails and no
    Groq keys exist, raises requests ConnectionError so existing callers'
    offline/mock paths still apply.
    """
    if claude_configured() and time.time() >= _claude_down_until:
        try:
            return _post_claude(payload, timeout, purpose, json_mode)
        except Exception as e:
            _note_claude_failure(e, purpose)
            if not key_pool():
                raise requests.exceptions.ConnectionError(
                    f"Claude failed ({type(e).__name__}) and no GROQ_API_KEYs "
                    "configured for fallback") from e
    return _post_groq(payload, timeout, slot, purpose, json_mode)
