"""campaign/recipe.py — the edit recipe (schema v1) and its validator.

A recipe is the machine-readable form of a campaign's requirements:

    {"version": 1,
     "mode": "auto" | "clip" | "edit",       # clip = cut a long video, edit = modify given videos
     "clips": {"count", "min_duration", "max_duration", "focus"},   # clip mode only
     "ops": [{"op": "logo", "asset": "brand.png", "position": "top-right"}, ...],
     "export": ["tiktok", "reels"],          # platform presets (campaign/presets.py)
     "output": {"filename": "{campaign}_{input}_{clip}_{platform}"},
     "questions": [{"id", "question", "default"?, "options"?}]}

`validate_recipe()` normalises a raw recipe (from the LLM, the keyword
fallback or the UI): unknown ops/params dropped with a warning, numbers
clamped to their ranges, defaults filled, enums checked, asset references
resolved against the campaign's asset catalog (never invented), single-use
ops de-duplicated, conflicts reported. Missing information becomes a
`question` for the user instead of a guess.

Ops marked phase "C8" are accepted and stored, but not rendered yet.
"""
import copy
import re

SCHEMA_VERSION = 1
REQUIRED = object()

ASPECTS = ("9:16", "4:5", "1:1", "16:9")
POSITIONS = ("top-left", "top-right", "bottom-left", "bottom-right", "top", "bottom",
             "center", "top-center", "bottom-center")
TEXT_ROLES = ("hook", "cta", "handle", "disclaimer", "custom")
ROLE_POSITION = {"hook": "top", "cta": "bottom", "handle": "bottom-right",
                 "disclaimer": "bottom", "custom": "center"}
MODES = ("auto", "clip", "edit")

_T = {"start": ("float", 0.0, (0.0, None)), "end": ("float", None, (0.0, None)),
      "from_end": ("float", None, (0.3, 600.0))}

# param spec: name -> (type, default, rule)
#   float/int: rule = (min, max) — values clamped;  enum: rule = allowed values
#   asset: rule = allowed asset kinds;  font: font asset name or a system font name
#   text: rule = max length;  color: "#RRGGBB";  bool
OPS = {
    "trim": {"start": _T["start"], "end": _T["end"],
             "max_duration": ("float", None, (1.0, 3600.0)),
             "min_duration": ("float", None, (0.0, 3600.0))},
    "reframe": {"aspect": ("enum", "9:16", ASPECTS),
                "method": ("enum", "auto", ("auto", "face", "center", "fit_blur"))},
    "audio.replace": {"track": ("asset", REQUIRED, ("audio",)),
                      "start": ("float", 0.0, (0.0, None)),
                      "loop": ("bool", True, None),
                      "volume": ("float", 1.0, (0.0, 2.0)),
                      "keep_original": ("float", 0.0, (0.0, 1.0)),
                      "fade_in": ("float", 0.0, (0.0, 10.0)),
                      "fade_out": ("float", 1.0, (0.0, 10.0))},
    "audio.add_music": {"track": ("asset", REQUIRED, ("audio",)),
                        "volume": ("float", 0.15, (0.0, 1.0)),
                        "duck": ("bool", True, None),
                        "start": ("float", 0.0, (0.0, None)),
                        "fade_out": ("float", 1.0, (0.0, 10.0))},
    "audio.mute_original": {},
    "audio.volume": {"level": ("float", 1.0, (0.0, 3.0))},
    "logo": {"asset": ("asset", REQUIRED, ("logo", "image")),
             "position": ("enum", "top-right", POSITIONS),
             "size_pct": ("float", 12.0, (3.0, 60.0)),
             "opacity": ("float", 1.0, (0.05, 1.0)),
             "margin_pct": ("float", 3.0, (0.0, 20.0)),
             "start": _T["start"], "end": _T["end"], "from_end": _T["from_end"]},
    "captions": {"enabled": ("bool", True, None),
                 "style": ("enum", "karaoke", ("karaoke", "plain", "boxed")),
                 "font": ("font", None, None),
                 "size": ("int", None, (20, 160)),
                 "color": ("color", None, None),
                 "highlight_color": ("color", None, None),
                 "position": ("enum", "auto", ("auto", "top", "center", "bottom")),
                 "uppercase": ("bool", False, None)},
    "text_overlay": {"text": ("text", REQUIRED, 200),
                     "role": ("enum", "custom", TEXT_ROLES),
                     "position": ("enum", None, POSITIONS),
                     "start": _T["start"], "end": _T["end"], "from_end": _T["from_end"],
                     "font": ("font", None, None),
                     "size": ("int", None, (16, 200)),
                     "color": ("color", "#FFFFFF", None),
                     "box": ("bool", True, None),
                     "animation": ("enum", "fade", ("none", "fade", "pop"))},
    # ── phase C8 ──
    "watermark": {"text": ("text", REQUIRED, 80), "opacity": ("float", 0.35, (0.05, 1.0)),
                  "position": ("enum", "bottom-right", POSITIONS)},
    "intro": {"asset": ("asset", REQUIRED, ("video", "image")),
              "duration": ("float", 2.0, (0.5, 30.0)),
              "transition": ("enum", "cut", ("cut", "fade"))},
    "outro": {"asset": ("asset", REQUIRED, ("video", "image")),
              "duration": ("float", 2.0, (0.5, 30.0)),
              "transition": ("enum", "cut", ("cut", "fade"))},
    "end_card": {"asset": ("asset", None, ("image", "logo")), "text": ("text", None, 120),
                 "duration": ("float", 2.0, (0.5, 15.0))},
    "speed": {"factor": ("float", REQUIRED, (0.5, 2.0))},
    "color": {"preset": ("enum", "none", ("none", "vivid", "warm", "cool", "bw")),
              "brightness": ("float", 0.0, (-0.5, 0.5)),
              "contrast": ("float", 1.0, (0.5, 2.0)),
              "saturation": ("float", 1.0, (0.0, 3.0))},
}
LATER_OPS = set()          # ops accepted but not rendered yet (none since C8)
MAX_VARIANTS = 5
SINGLE_OPS = {"trim", "reframe", "audio.replace", "audio.add_music", "audio.mute_original",
              "audio.volume", "captions", "speed", "color", "intro", "outro", "end_card",
              "watermark"}
FILENAME_TOKENS = {"campaign", "input", "clip", "platform", "variant", "n"}
DEFAULT_FILENAME = "{campaign}_{input}_{clip}_{platform}"


def empty_recipe() -> dict:
    return {"version": SCHEMA_VERSION, "mode": "auto",
            "clips": {"count": None, "min_duration": None, "max_duration": None, "focus": None},
            "ops": [], "export": ["generic"], "output": {"filename": DEFAULT_FILENAME},
            "variants": [], "questions": []}


# ── helpers ──────────────────────────────────────────────────────────────────

def _num(v, kind):
    if isinstance(v, bool):
        raise ValueError("not a number")
    if isinstance(v, str):
        m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*(s|sec|secs|seconds|%)?\s*$", v.lower())
        if not m:
            raise ValueError("not a number")
        v = m.group(1)
    return int(round(float(v))) if kind == "int" else float(v)


def _bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("true", "yes", "on", "1", "y"):
        return True
    if s in ("false", "no", "off", "0", "n", "none", "null"):
        return False
    raise ValueError("not a boolean")


_NAMED_COLORS = {"white": "#FFFFFF", "black": "#000000", "yellow": "#FFE000", "red": "#FF3B30",
                 "green": "#34C759", "blue": "#0A84FF", "orange": "#FF9500", "pink": "#FF2D92",
                 "purple": "#AF52DE", "gray": "#8E8E93", "grey": "#8E8E93"}


def _color(v):
    s = str(v).strip().lower()
    if s in _NAMED_COLORS:
        return _NAMED_COLORS[s]
    m = re.match(r"^#?([0-9a-f]{6})$", s)
    if not m:
        raise ValueError("colour must be #RRGGBB or a basic colour name")
    return "#" + m.group(1).upper()


def find_asset(value, catalog, kinds):
    """Match an asset reference (name, path or bare stem) to the catalog."""
    if value in (None, ""):
        return None
    v = str(value).strip().replace("\\", "/")
    base = v.rsplit("/", 1)[-1].lower()
    stem = base.rsplit(".", 1)[0]
    cands = [a for a in catalog or [] if a.get("kind") in kinds]
    for a in cands:
        if a["path"].replace("\\", "/") == v or a["name"].lower() == base:
            return a
    for a in cands:
        if a["name"].lower().rsplit(".", 1)[0] == stem:
            return a
    return None


class _Ctx:
    def __init__(self, catalog, autopick=False):
        self.catalog = catalog or []
        self.autopick = autopick
        self.errors, self.warnings, self.questions = [], [], []

    def ask(self, qid, question, default=None, options=None):
        if any(q["id"] == qid for q in self.questions):
            return
        q = {"id": qid, "question": question}
        if default is not None:
            q["default"] = default
        if options:
            q["options"] = list(options)
        self.questions.append(q)


def _norm_op(raw, idx, ctx):
    name = str(raw.get("op", "")).strip().lower().replace("-", "_")
    name = {"music": "audio.add_music", "add_music": "audio.add_music",
            "replace_audio": "audio.replace", "mute": "audio.mute_original",
            "subtitles": "captions", "text": "text_overlay", "overlay": "text_overlay",
            "volume": "audio.volume", "crop": "reframe", "aspect": "reframe"}.get(name, name)
    if name not in OPS:
        ctx.warnings.append(f"op {idx}: unknown operation {raw.get('op')!r} ignored")
        return None
    spec, out = OPS[name], {"op": name}
    for key in raw:
        # <asset key>_path is the validator's own resolved path (recomputed below)
        if key != "op" and key not in spec and not (key.endswith("_path") and key[:-5] in spec):
            ctx.warnings.append(f"{name}: unknown setting {key!r} ignored")
    for key, (typ, default, rule) in spec.items():
        v = raw.get(key, None)
        if v in (None, "") and default is REQUIRED:
            if typ == "asset":
                pool = [a for a in ctx.catalog if a["kind"] in rule]
                if len(pool) == 1 and ctx.autopick:
                    out[key] = pool[0]["name"]
                    out[f"{key}_path"] = pool[0]["path"]
                    ctx.warnings.append(f"{name}: no {key} named, using the only "
                                        f"{'/'.join(rule)} asset {pool[0]['name']!r}")
                    continue
                if pool:
                    ctx.ask(f"{name}.{key}", f"Which file should '{name}' use?",
                            default=pool[0]["name"] if len(pool) == 1 else None,
                            options=[a["name"] for a in pool])
                else:
                    ctx.ask(f"{name}.{key}", f"'{name}' needs a {'/'.join(rule)} file — "
                            f"upload one to the campaign's assets.")
                ctx.errors.append(f"{name}: missing {key}")
            elif typ == "text":
                ctx.ask(f"{name}.{key}", f"What text should '{name}' show?")
                ctx.errors.append(f"{name}: missing {key}")
            else:
                ctx.errors.append(f"{name}: missing {key}")
            out[key] = None
            continue
        if v in (None, ""):
            out[key] = default
            continue
        try:
            if typ in ("float", "int"):
                n = _num(v, typ)
                lo, hi = rule
                c = n if lo is None or n >= lo else lo
                c = c if hi is None or c <= hi else hi
                if c != n:
                    ctx.warnings.append(f"{name}.{key}: {n} clamped to {c}")
                out[key] = c
            elif typ == "bool":
                out[key] = _bool(v)
            elif typ == "enum":
                s = str(v).strip().lower()
                if key == "position":
                    s = re.sub(r"[\s_]+", "-", s)
                    s = {"middle": "center", "centre": "center", "top-middle": "top-center",
                         "bottom-middle": "bottom-center"}.get(s, s)
                if key == "aspect":
                    s = {"vertical": "9:16", "portrait": "9:16", "square": "1:1",
                         "landscape": "16:9", "horizontal": "16:9"}.get(s, s)
                if s not in rule:
                    raise ValueError(f"must be one of {', '.join(rule)}")
                out[key] = s
            elif typ == "text":
                s = str(v).strip()
                if len(s) > rule:
                    ctx.warnings.append(f"{name}.{key}: text cut to {rule} characters")
                    s = s[:rule]
                out[key] = s
            elif typ == "color":
                out[key] = _color(v)
            elif typ == "asset":
                a = find_asset(v, ctx.catalog, rule)
                if a is None:
                    names = [x["name"] for x in ctx.catalog if x["kind"] in rule]
                    ctx.errors.append(f"{name}.{key}: {v!r} is not in the campaign's assets")
                    ctx.ask(f"{name}.{key}", f"'{v}' was not found in the campaign's assets — "
                            f"upload it or pick one.", options=names or None)
                    out[key] = None
                else:
                    out[key] = a["name"]
                    out[f"{key}_path"] = a["path"]
            elif typ == "font":
                a = find_asset(v, ctx.catalog, ("font",))
                out[key] = a["name"] if a else str(v).strip()
                if a:
                    out[f"{key}_path"] = a["path"]
        except ValueError as e:
            ctx.warnings.append(f"{name}.{key}: {v!r} {e}; using default")
            out[key] = None if default is REQUIRED else default
    if name == "text_overlay" and not out.get("position"):
        out["position"] = ROLE_POSITION.get(out.get("role") or "custom", "center")
    if out.get("end") is not None and out.get("start") is not None and out["end"] <= out["start"]:
        ctx.errors.append(f"{name}: end ({out['end']}s) must be after start ({out['start']}s)")
    if name in LATER_OPS:
        ctx.warnings.append(f"{name}: accepted, but not rendered yet")
    return out


def validate_recipe(raw: dict, catalog: list = None, autopick: bool = False) -> dict:
    """Normalise a raw recipe. Returns {recipe, errors, warnings, questions, ok}.

    autopick: an op that needs a file but names none uses the campaign's only
    file of that kind. Off by default — an LLM leaves the asset empty exactly
    when the brief asks for a file that isn't uploaded (e.g. "the voiceover"),
    and silently substituting another file would be wrong. The keyword
    fallback turns it on ("add the logo" with one logo uploaded).
    """
    ctx = _Ctx(catalog, autopick)
    raw = copy.deepcopy(raw or {})
    r = empty_recipe()
    if raw.get("version") not in (None, SCHEMA_VERSION):
        ctx.warnings.append(f"recipe version {raw.get('version')} read as v{SCHEMA_VERSION}")

    mode = str(raw.get("mode") or "auto").strip().lower()
    if mode not in MODES:
        ctx.warnings.append(f"mode {mode!r} unknown, using auto")
        mode = "auto"
    r["mode"] = mode

    clips = raw.get("clips") or {}
    for key, kind, lo, hi in (("count", "int", 1, 30), ("min_duration", "float", 5.0, 600.0),
                              ("max_duration", "float", 5.0, 600.0)):
        v = clips.get(key)
        if v in (None, ""):
            continue
        try:
            n = _num(v, kind)
            r["clips"][key] = min(max(n, lo), hi)
        except ValueError:
            ctx.warnings.append(f"clips.{key}: {v!r} is not a number")
    if clips.get("focus"):
        r["clips"]["focus"] = str(clips["focus"]).strip()[:300]
    c = r["clips"]
    if c["min_duration"] and c["max_duration"] and c["min_duration"] > c["max_duration"]:
        ctx.errors.append(f"clips: min_duration {c['min_duration']}s > max_duration {c['max_duration']}s")

    ops = []
    for i, op in enumerate(raw.get("ops") or []):
        if not isinstance(op, dict):
            ctx.warnings.append(f"op {i}: not an object, ignored")
            continue
        n = _norm_op(op, i, ctx)
        if n is None:
            continue
        if n["op"] in SINGLE_OPS:
            prev = [k for k, x in enumerate(ops) if x["op"] == n["op"]]
            if prev:
                ctx.warnings.append(f"{n['op']} given more than once; the last one is used")
                ops = [x for k, x in enumerate(ops) if k not in prev]
        ops.append(n)
    names = {o["op"] for o in ops}
    if "audio.replace" in names and "audio.mute_original" in names:
        ops = [o for o in ops if o["op"] != "audio.mute_original"]
        ctx.warnings.append("audio.mute_original dropped: audio.replace already removes the original")
    if "audio.replace" in names and "audio.add_music" in names:
        ctx.ask("audio.conflict", "The brief asks to replace the audio AND add music — keep "
                "both (music under the new track) or only the replacement?",
                default="only the replacement", options=["only the replacement", "both"])
    if mode == "clip":
        t = next((o for o in ops if o["op"] == "trim"), None)
        if t and (t.get("start") or t.get("end") is not None):
            ctx.warnings.append("trim start/end ignored in clip mode (clips are cut by the "
                                "clip finder); max_duration still applies per clip")
    r["ops"] = ops

    # variants: alternate versions of every deliverable (e.g. another hook or track).
    # Each lists ops that replace the base op of the same kind (text by role) and/or
    # op names to remove.
    seen = {"main"}
    for j, v in enumerate(raw.get("variants") or []):
        if j >= MAX_VARIANTS:
            ctx.warnings.append(f"variants: only the first {MAX_VARIANTS} are used")
            break
        if not isinstance(v, dict):
            ctx.warnings.append(f"variant {j}: not an object, ignored")
            continue
        name = re.sub(r"[^a-z0-9_-]+", "-", str(v.get("name") or f"v{j + 2}").lower()).strip("-")
        if not name or name in seen:
            name = f"v{j + 2}"
        seen.add(name)
        vops = [n for n in (_norm_op(op, f"variant {name} op {k}", ctx)
                            for k, op in enumerate(v.get("ops") or []) if isinstance(op, dict))
                if n is not None]
        remove = [x for x in v.get("remove") or [] if x in OPS]
        if not vops and not remove:
            ctx.warnings.append(f"variant {name}: changes nothing, ignored")
            continue
        r["variants"].append({"name": name, "ops": vops, "remove": remove})
    # "2 versions: hook A / hook B" is often written as two variants that both only ADD
    # an op the base lacks — the first one is then the main version, not an extra file
    def _adds_only(v):
        def matches(vo):
            return any(o["op"] == vo["op"] and (vo["op"] != "text_overlay"
                                                or o.get("role") == vo.get("role"))
                       for o in r["ops"])
        return not v["remove"] and not any(matches(vo) for vo in v["ops"])
    if len(r["variants"]) >= 2 and all(_adds_only(v) for v in r["variants"]):
        first = r["variants"].pop(0)
        r["ops"] += first["ops"]
        ctx.warnings.append(f"variant {first['name']} used as the main version "
                            f"(every variant only added ops)")

    from campaign.presets import normalize_platform, PLATFORMS
    plats = []
    for p in raw.get("export") or []:
        p = p.get("platform") if isinstance(p, dict) else p
        n = normalize_platform(p)
        if n is None:
            ctx.warnings.append(f"export: unknown platform {p!r} ignored")
        elif n not in plats:
            plats.append(n)
    r["export"] = plats or ["generic"]
    aspect = next((o["aspect"] for o in ops if o["op"] == "reframe"), None)
    for p in r["export"]:
        if aspect and PLATFORMS[p]["aspect"] != aspect:
            ctx.warnings.append(f"export {p}: platform default is {PLATFORMS[p]['aspect']} "
                                f"but the recipe reframes to {aspect}")
        cap = next((o.get("max_duration") for o in ops if o["op"] == "trim"), None) \
            or r["clips"]["max_duration"]
        if cap and cap > PLATFORMS[p]["max_duration"]:
            ctx.warnings.append(f"export {p}: max duration {cap:.0f}s is above the "
                                f"{PLATFORMS[p]['max_duration']}s preset limit")

    fn = str((raw.get("output") or {}).get("filename") or DEFAULT_FILENAME).strip()
    bad = set(re.findall(r"{(\w+)}", fn)) - FILENAME_TOKENS
    if bad or re.search(r"[/\\:]", fn):
        ctx.warnings.append(f"output.filename {fn!r} invalid; using the default")
        fn = DEFAULT_FILENAME
    r["output"]["filename"] = fn

    # parser questions first-class; a validator "upload a file for X" question is
    # dropped when the parser already asked about X more specifically
    raw_qs = [q for q in raw.get("questions") or []
              if (isinstance(q, dict) and q.get("question")) or (isinstance(q, str) and q.strip())]
    raw_text = " ".join((q["question"] if isinstance(q, dict) else q).lower() for q in raw_qs)
    ctx.questions = [q for q in ctx.questions
                     if not (q["question"].endswith("upload one to the campaign's assets.")
                             and q["id"].split(".")[0].replace("audio.", "").replace("_", " ")
                             in raw_text)]
    for q in raw_qs:
        if isinstance(q, dict) and q.get("question"):
            ctx.ask(str(q.get("id") or f"q{len(ctx.questions)}"), str(q["question"]),
                    q.get("default"), q.get("options"))
        elif isinstance(q, str) and q.strip():
            ctx.ask(f"q{len(ctx.questions)}", q.strip())
    r["questions"] = ctx.questions
    return {"recipe": r, "errors": ctx.errors, "warnings": ctx.warnings,
            "questions": ctx.questions, "ok": not ctx.errors}


# ── human-readable summary ───────────────────────────────────────────────────

def _when(o):
    if o.get("from_end"):
        return f" for the last {o['from_end']:g}s"
    if o.get("start") or o.get("end") is not None:
        end = f"{o['end']:g}s" if o.get("end") is not None else "the end"
        return f" from {o.get('start') or 0:g}s to {end}"
    return ""


def describe_op(o: dict) -> str:
    """One plain-English line for an op."""
    op = o["op"]
    if op == "trim":
        bits = []
        if o.get("start") or o.get("end") is not None:
            bits.append(f"keep {o.get('start') or 0:g}s–" +
                        (f"{o['end']:g}s" if o.get("end") is not None else "end"))
        if o.get("max_duration"):
            bits.append(f"max {o['max_duration']:g}s")
        if o.get("min_duration"):
            bits.append(f"min {o['min_duration']:g}s")
        return "Trim: " + (", ".join(bits) or "no change")
    if op == "reframe":
        return f"Reframe to {o['aspect']} ({o['method']})"
    if op == "audio.replace":
        keep = (f", original kept at {o['keep_original']:.0%}" if o.get("keep_original")
                else ", original removed")
        return f"Replace audio with {o.get('track') or '?'}{keep}"
    if op == "audio.add_music":
        return (f"Add music {o.get('track') or '?'} at {o['volume']:.0%}"
                + (" (ducked under speech)" if o.get("duck") else ""))
    if op == "audio.mute_original":
        return "Mute the original audio"
    if op == "audio.volume":
        return f"Original audio volume {o['level']:.0%}"
    if op == "logo":
        return (f"Logo {o.get('asset') or '?'} {o['position']}, {o['size_pct']:g}% wide"
                + (f", {o['opacity']:.0%} opacity" if o.get("opacity", 1) < 1 else "")
                + _when(o))
    if op == "captions":
        return ("Captions off" if not o.get("enabled") else
                f"Captions: {o['style']}, position {o['position']}"
                + (", UPPERCASE" if o.get("uppercase") else ""))
    if op == "text_overlay":
        return f"Text ({o['role']}) \"{o.get('text') or '?'}\" at {o['position']}{_when(o)}"
    if op == "watermark":
        return f"Watermark \"{o.get('text')}\" {o['position']} at {o['opacity']:.0%}"
    if op in ("intro", "outro"):
        return (f"{op.capitalize()} {o.get('asset') or '?'}"
                + (f" ({o['duration']:g}s if an image)" if o.get("duration") else "")
                + (", fade" if o.get("transition") == "fade" else ""))
    if op == "end_card":
        return (f"End card {o['duration']:g}s" + (f" with {o['asset']}" if o.get("asset") else "")
                + (f" \"{o['text']}\"" if o.get("text") else ""))
    if op == "speed":
        return f"Speed x{o['factor']:g}"
    if op == "color":
        bits = [o["preset"]] if o.get("preset") not in (None, "none") else []
        for k, d in (("brightness", 0.0), ("contrast", 1.0), ("saturation", 1.0)):
            if o.get(k) is not None and abs(o[k] - d) > 1e-6:
                bits.append(f"{k} {o[k]:g}")
        return "Colour: " + (", ".join(bits) or "no change")
    return op


def describe_recipe(recipe: dict) -> list:
    """Plain-English lines for the UI and the campaign's requirements summary."""
    lines = []
    mode = recipe.get("mode", "auto")
    lines.append({"auto": "Mode: decided per input (long videos are clipped, short ones edited)",
                  "clip": "Mode: cut clips from long videos",
                  "edit": "Mode: edit the supplied videos (no cutting)"}[mode])
    c = recipe.get("clips") or {}
    if mode != "edit" and any(c.get(k) for k in ("count", "min_duration", "max_duration", "focus")):
        parts = []
        if c.get("count"):
            parts.append(f"{c['count']} clips")
        if c.get("min_duration") or c.get("max_duration"):
            parts.append(f"{c.get('min_duration') or '?'}–{c.get('max_duration') or '?'}s each")
        if c.get("focus"):
            parts.append(f"focus: {c['focus']}")
        lines.append("Clips: " + ", ".join(parts))
    lines += [describe_op(o) for o in recipe.get("ops") or []]
    for v in recipe.get("variants") or []:
        bits = [describe_op(o) for o in v.get("ops") or []]
        bits += [f"without {x}" for x in v.get("remove") or []]
        lines.append(f"Variant {v['name']}: " + "; ".join(bits))
    from campaign.presets import PLATFORMS
    lines.append("Export: " + ", ".join(PLATFORMS[p]["label"] for p in recipe.get("export") or ["generic"]))
    return lines


# ── variants ─────────────────────────────────────────────────────────────────

def apply_variant(recipe: dict, variant: dict) -> dict:
    """The recipe of one variant: base ops minus `remove`, with each variant op
    replacing the base op of the same kind (text overlays by role, logos all)."""
    base = copy.deepcopy(recipe)
    base.pop("variants", None)
    ops = [o for o in base.get("ops") or [] if o["op"] not in set(variant.get("remove") or [])]
    for vo in variant.get("ops") or []:
        if vo["op"] == "text_overlay":
            same = [k for k, o in enumerate(ops) if o["op"] == "text_overlay"
                    and o.get("role") == vo.get("role")]
        elif vo["op"] == "logo" or vo["op"] in SINGLE_OPS:
            same = [k for k, o in enumerate(ops) if o["op"] == vo["op"]]
        else:
            same = []
        if same:
            ops[same[0]] = copy.deepcopy(vo)
            ops = [o for k, o in enumerate(ops) if k not in same[1:]]
        else:
            ops.append(copy.deepcopy(vo))
    base["ops"] = ops
    return base


def variants_of(recipe: dict) -> list:
    """[(name, recipe)] — "main" first, then each declared variant."""
    main = copy.deepcopy(recipe or {})
    main.pop("variants", None)
    return [("main", main)] + [(v["name"], apply_variant(recipe, v))
                               for v in (recipe or {}).get("variants") or []]
