"""campaign/brief_parser.py — campaign brief (free text) → validated edit recipe.

parse_brief(brief, catalog) returns
    {"recipe", "summary", "questions", "errors", "warnings", "method"}
where `method` is "llm" or "keyword". The LLM (Groq via llm_client, JSON mode,
temperature 0) writes a raw recipe from the brief and the campaign's asset
catalog; `campaign.recipe.validate_recipe` then normalises it — assets can
only come from the catalog, missing information becomes a question. Without a
usable LLM (or if it fails) a keyword parser covers the common requests so the
tool still works offline.
"""
import json
import re

from logger import get_logger
from campaign.recipe import OPS, POSITIONS, validate_recipe, describe_recipe
from campaign.presets import PLATFORMS

log = get_logger("campaign.brief")


def _schema_text() -> str:
    lines = []
    for name, spec in OPS.items():
        params = []
        for k, (typ, default, rule) in spec.items():
            if typ == "enum":
                t = "|".join(rule)
            elif typ == "asset":
                t = f"<{'/'.join(rule)} asset name>"
            else:
                t = typ
            params.append(f"{k}: {t}")
        lines.append(f'  {{"op": "{name}"' + (", " + ", ".join(params) if params else "") + "}")
    return "\n".join(lines)


SYSTEM_PROMPT = """You turn a video clipping-campaign brief into an edit recipe (JSON only).

The tool either CUTS clips from a long video ("clip" mode) or MODIFIES videos that are
supplied as they are ("edit" mode: replace audio, add a logo, text, captions, reframe...).

Return ONLY a JSON object:
{
 "mode": "auto|clip|edit",
 "clips": {"count": int|null, "min_duration": seconds|null, "max_duration": seconds|null, "focus": "topics to prioritise"|null},
 "ops": [ ...operations... ],
 "export": ["generic|tiktok|reels|shorts|youtube|x|facebook", ...],
 "output": {"filename": "{campaign}_{input}_{clip}_{platform}"},
 "variants": [{"name": "short-name", "ops": [ops that replace the base op of the same kind],
               "remove": ["op names to drop"]}],
 "questions": [{"id": "short-id", "question": "...", "default": "..."|null, "options": [..]|null}],
 "summary": ["one short line per requirement, in the brief's own terms"]
}

Operations (all settings optional unless the brief states them):
%s

Rules:
1. Include ONLY what the brief explicitly asks for. No extra ops, no invented settings.
2. Asset settings (track, asset, font) must be a NAME from the asset list below. If the brief
   needs a file that is not in the list, leave it null and add a question asking to upload it.
3. mode: "clip" if the brief talks about cutting/clipping highlights from a long video; "edit" if
   it asks to modify given videos without cutting; otherwise "auto".
4. Durations in seconds. "under 60 seconds" -> trim.max_duration 60 (edit) or clips.max_duration 60 (clip).
5. Timed overlays: "last 3 seconds" -> from_end 3; "first 5 seconds" -> start 0, end 5.
6. Text overlays: role hook (attention line at the start), cta ("link in bio", "follow"),
   handle ("@brand"), disclaimer (legal / #ad); copy the exact text from the brief.
7. Ambiguity or conflict -> a question with a sensible default, never a silent guess. Example:
   "replace the audio" without saying whether the speaker's voice stays -> keep_original 0 and
   ask "Remove the speaker's voice completely?" with default "yes".
8. Platforms: only those named; none named -> ["generic"].
9. Leave every setting the brief does not state as null so the defaults apply (caption style,
   positions, sizes, volumes, fonts...). Do not pick a style because it "sounds right".
10. Never ask for the source video(s) — they are attached to each job separately — and never
    ask about file names or the campaign name; those are automatic.
11. Voice-over / narration on top of the original sound -> "audio.replace" with the voice-over
    file and keep_original 0.25 (ask if the brief doesn't say how loud the original stays).
    Background music under the speech -> "audio.add_music".
12. Variants only when the brief asks for several versions ("2 versions with different hooks",
    "one with music and one without"): the base ops are version 1; each variant lists only what
    differs. Otherwise "variants": [].
13. Intro / outro / end card need a video or image asset (or end-card text); speed 0.5–2.0;
    color presets vivid|warm|cool|bw; watermark = small translucent text. An end card with
    only text is fine — don't ask for an image unless the brief mentions one.
"""


def _llm_parse(brief, catalog):
    from config import LLM_MODEL
    from pipeline.llm_client import post_chat, pool_usable, SLOT_FOR_STAGE
    if not pool_usable():
        return None
    assets = [{"name": a["name"], "kind": a["kind"], **({"duration": a["duration"]}
               if a.get("duration") else {})} for a in catalog or []]
    user = (f"Campaign brief:\n\"\"\"\n{brief.strip()}\n\"\"\"\n\n"
            f"Assets available (use these names only): {json.dumps(assets)}\n"
            f"Return the JSON recipe.")
    try:
        r = post_chat({"model": LLM_MODEL, "temperature": 0, "max_tokens": 1800,
                       "messages": [{"role": "system", "content": SYSTEM_PROMPT % _schema_text()},
                                    {"role": "user", "content": user}]},
                      timeout=90, slot=SLOT_FOR_STAGE["campaign"], purpose="brief",
                      json_mode=True)
        raw = r.json()["choices"][0]["message"]["content"]
        raw = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL)
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        return json.loads(m.group(0)) if m else None
    except Exception:
        log.exception("brief LLM parse failed; keyword fallback")
        return None


# ── keyword fallback ─────────────────────────────────────────────────────────

_POS_WORDS = [(p, re.compile(r"\b" + p.replace("-", r"[\s-]") + r"\b")) for p in
              sorted(POSITIONS, key=len, reverse=True)]


def _position_near(text, word):
    """Position phrase within ~40 chars after/before a keyword."""
    for m in re.finditer(word, text):
        window = text[max(0, m.start() - 40): m.end() + 40]
        for p, rx in _POS_WORDS:
            if rx.search(window):
                return p
    return None


def _secs(rx, text):
    m = re.search(rx, text)
    if not m:
        return None
    n = float(m.group(1))
    unit = (m.group(2) or "s").lower()
    return n * 60 if unit.startswith("m") else n


def _pick(catalog, kind, hint_text):
    pool = [a for a in catalog or [] if a["kind"] in kind]
    for a in pool:
        stem = a["name"].rsplit(".", 1)[0].lower()
        if stem and stem in hint_text:
            return a["name"]
    return pool[0]["name"] if len(pool) == 1 else None


CTA_WORDS = (r"\b(cta|call to action|link|follow|subscribe|shop|buy|order|download|sign up|"
             r"visit|swipe|tap|click|join|get yours|use code|install|book now)\b")


def keyword_parse(brief, catalog=None):
    """Deterministic best-effort parse of common requests (no LLM)."""
    t = " " + str(brief or "").lower() + " "
    rec = {"mode": "auto", "clips": {}, "ops": [], "export": [], "questions": []}
    if re.search(r"\b(clip|clips|highlights?|cut (?:it|the video)? ?(?:into|down)|shorts? from)\b", t) \
            and re.search(r"\b(long|podcast|stream|episode|full video|interview)\b", t):
        rec["mode"] = "clip"
    elif re.search(r"\b(replace|swap|change) (?:the )?(?:audio|sound|music)\b|\bdon'?t cut\b|"
                   r"\bno cutting\b|\bas (?:they|it) (?:are|is)\b", t):
        rec["mode"] = "edit"
    n = re.search(r"\b(\d{1,2})\s+(?:short\s+)?(?:clips|shorts|videos)\b", t)
    if n and rec["mode"] != "edit":
        rec["clips"]["count"] = int(n.group(1))
    mx = _secs(r"(?:under|max(?:imum)?|at most|no longer than|up to|<)\s*(\d+(?:\.\d+)?)\s*(s|sec|secs|seconds|m|min|mins|minutes)\b", t)
    mn = _secs(r"(?:at least|min(?:imum)?|longer than|>)\s*(\d+(?:\.\d+)?)\s*(s|sec|secs|seconds|m|min|mins|minutes)\b", t)
    if rec["mode"] == "clip":
        if mx:
            rec["clips"]["max_duration"] = mx
        if mn:
            rec["clips"]["min_duration"] = mn
    elif mx or mn:
        rec["ops"].append({"op": "trim", "max_duration": mx, "min_duration": mn})

    aspect = None
    if re.search(r"\b(9[:x/]16|vertical|portrait)\b", t):
        aspect = "9:16"
    elif re.search(r"\b(1[:x/]1|square)\b", t):
        aspect = "1:1"
    elif re.search(r"\b4[:x/]5\b", t):
        aspect = "4:5"
    elif re.search(r"\b(16[:x/]9|landscape|horizontal)\b", t):
        aspect = "16:9"
    if aspect:
        rec["ops"].append({"op": "reframe", "aspect": aspect})

    if re.search(r"\b(replace|swap|change) (?:the )?(?:audio|sound|music|track)\b|\bnew (?:audio|track|sound)\b", t):
        keep = 0.15 if re.search(r"\bkeep (?:the )?(?:voice|speaker|original)\b", t) else 0.0
        rec["ops"].append({"op": "audio.replace", "track": _pick(catalog, ("audio",), t),
                           "keep_original": keep})
        if not keep:
            rec["questions"].append({"id": "audio.keep_voice", "question":
                                     "Remove the speaker's original voice completely?",
                                     "default": "yes", "options": ["yes", "keep it quietly under the new track"]})
    elif re.search(r"\b(add|use|put|background) (?:some |a |the )?(?:background )?(?:music|song|track)\b", t):
        rec["ops"].append({"op": "audio.add_music", "track": _pick(catalog, ("audio",), t)})
    if re.search(r"\b(mute|remove|no) (?:the )?(?:original )?(?:audio|sound)\b", t) \
            and not any(o["op"] == "audio.replace" for o in rec["ops"]):
        rec["ops"].append({"op": "audio.mute_original"})

    if re.search(r"\blogo\b|\bwatermark\b", t) and not re.search(r"\bno logo\b", t):
        size = re.search(r"logo[^.]{0,40}?(\d{1,2})\s*%", t)
        rec["ops"].append({"op": "logo", "asset": _pick(catalog, ("logo", "image"), t),
                           "position": _position_near(t, r"\blogo\b") or "top-right",
                           **({"size_pct": float(size.group(1))} if size else {})})

    if re.search(r"\bno (?:captions|subtitles)\b|\bwithout (?:captions|subtitles)\b", t):
        rec["ops"].append({"op": "captions", "enabled": False})
    elif re.search(r"\b(captions?|subtitles?|subs)\b", t):
        style = "karaoke" if re.search(r"karaoke|word.by.word|highlight", t) else \
                "boxed" if re.search(r"\bbox", t) else "karaoke"
        rec["ops"].append({"op": "captions", "style": style,
                           "uppercase": bool(re.search(r"\b(uppercase|all caps|caps)\b", t))})

    for m in re.finditer(r"[\"“']([^\"”']{2,120})[\"”']", str(brief or "")):
        text = m.group(1).strip()
        ctx = t[max(0, m.start() - 60): m.start() + 1]
        role = ("cta" if re.search(CTA_WORDS, text.lower() + " " + ctx) else
                "handle" if text.startswith("@") else
                "hook" if re.search(r"\bhook\b", ctx) else
                "disclaimer" if re.search(r"#ad|disclaimer|sponsored|paid partnership", text.lower() + ctx) else
                "custom")
        op = {"op": "text_overlay", "text": text, "role": role}
        after = t[m.end() + 1: m.end() + 60]
        last = re.search(r"(?:last|final)\s+(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds)\b", after)
        first = re.search(r"(?:first)\s+(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds)\b", after)
        if last:
            op["from_end"] = float(last.group(1))
        elif first:
            op["start"], op["end"] = 0.0, float(first.group(1))
        rec["ops"].append(op)

    from campaign.presets import normalize_platform
    for name in list(PLATFORMS) + ["instagram", "youtube shorts", "twitter", "tik tok"]:
        if name == "generic":
            continue
        if re.search(r"\b" + re.escape(name) + r"\b", t):
            n = normalize_platform(name)
            if n and n not in rec["export"]:
                rec["export"].append(n)
    if "shorts" in rec["export"] and "youtube shorts" in t:
        rec["export"] = [p for p in rec["export"] if p != "youtube"]
    return rec


def _fill_gaps(raw, kw) -> list:
    """Hard numbers the LLM dropped but the keyword parser found (duration limits,
    named platforms) are added back — never guessed, only copied from the brief."""
    notes = []
    if not isinstance(raw, dict):
        return notes
    ops = raw.setdefault("ops", [])
    clips = raw.setdefault("clips", {}) or {}
    raw["clips"] = clips
    kw_trim = next((o for o in kw.get("ops") or [] if o.get("op") == "trim"), {})
    kw_clips = kw.get("clips") or {}
    for key in ("max_duration", "min_duration"):
        val = kw_trim.get(key) or kw_clips.get(key)
        if not val:
            continue
        have_trim = any(o.get("op") == "trim" and o.get(key) for o in ops if isinstance(o, dict))
        if have_trim or clips.get(key):
            continue
        if (raw.get("mode") or kw.get("mode")) == "clip":
            clips[key] = val
        else:
            trim = next((o for o in ops if isinstance(o, dict) and o.get("op") == "trim"), None)
            if trim is None:
                trim = {"op": "trim"}
                ops.insert(0, trim)
            trim[key] = val
        notes.append(f"{key.replace('_', ' ')} {val:g}s taken from the brief (the LLM missed it)")
    # file naming is automatic unless the brief itself asks for a naming scheme
    if raw.get("output") and not re.search(r"file ?names?|named|naming",
                                           str(kw.get("_brief") or ""), re.I):
        raw.pop("output", None)

    # on-screen text must be copied from the brief, never made up (e.g. "HOOK_TEXT")
    def norm(x):
        return re.sub(r"[^a-z0-9@#]+", " ", str(x or "").lower()).strip()
    brief_n = norm(kw.get("_brief"))
    kw_roles = {norm(o.get("text")): o.get("role") for o in kw.get("ops") or []
                if o.get("op") == "text_overlay"}

    def clean(op_list, where):
        kept = []
        for o in op_list:
            if not isinstance(o, dict) or o.get("op") not in ("text_overlay", "watermark",
                                                              "end_card"):
                kept.append(o)
                continue
            t = norm(o.get("text"))
            if t and brief_n and t not in brief_n:
                notes.append(f"{where}{o['op']} text {o.get('text')!r} dropped — it is not in "
                             "the brief")
                if o["op"] == "end_card":
                    o = dict(o, text=None)
                else:
                    continue
            if o.get("op") == "text_overlay" and o.get("role") in (None, "custom") \
                    and kw_roles.get(t) not in (None, "custom"):
                o = dict(o, role=kw_roles[t])
            kept.append(o)
        return kept
    raw["ops"] = clean(ops, "")
    for v in raw.get("variants") or []:
        if isinstance(v, dict):
            v["ops"] = clean(v.get("ops") or [], f"variant {v.get('name')}: ")

    kw_plats = [p for p in kw.get("export") or [] if p != "generic"]
    if kw_plats and not [p for p in raw.get("export") or [] if p not in (None, "generic")]:
        raw["export"] = kw_plats
        notes.append(f"platforms {', '.join(kw_plats)} taken from the brief (the LLM missed them)")
    return notes


def parse_brief(brief: str, catalog: list = None, use_llm: bool = True) -> dict:
    brief = str(brief or "").strip()
    if not brief:
        res = validate_recipe({}, catalog)
        return {**res, "summary": describe_recipe(res["recipe"]), "method": "empty"}
    raw, method = None, "keyword"
    if use_llm:
        raw = _llm_parse(brief, catalog)
        if raw is not None:
            method = "llm"
    if raw is None:
        raw = keyword_parse(brief, catalog)
    llm_summary = raw.pop("summary", None) if isinstance(raw, dict) else None
    notes = (_fill_gaps(raw, dict(keyword_parse(brief, catalog), _brief=brief))
             if method == "llm" else [])
    res = validate_recipe(raw, catalog, autopick=(method == "keyword"))
    res["warnings"] = notes + res["warnings"]
    summary = describe_recipe(res["recipe"])
    log.info("brief parsed method=%s ops=%s questions=%d errors=%d", method,
             [o["op"] for o in res["recipe"]["ops"]], len(res["questions"]), len(res["errors"]))
    return {**res, "summary": summary, "brief_summary": llm_summary or [], "method": method}
