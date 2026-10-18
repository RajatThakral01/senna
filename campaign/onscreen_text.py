"""campaign/onscreen_text.py — AI-written on-screen text, one line per clip.

A recipe text overlay whose text is the token "{auto}" is written per clip by
the LLM, following the campaign brief's own rules (tone, example lines to adapt,
required names, no negativity, nothing invented). The same call writes the POST
CAPTION with every @tag / #hashtag / song name the brief requires. Clip mode
resolves it per clip from the clip's transcript; edit mode per input.

resolve_auto_text() returns a copy of the recipe with every "{auto}" replaced,
the texts it used (stored on the deliverable so a re-render reuses them instead
of writing new ones) and warnings. If the LLM is unavailable the overlay is
dropped with a warning: a generic filler line would break most briefs.
"""
import copy
import json
import re

from logger import get_logger

log = get_logger("campaign.onscreen_text")

AUTO = "{auto}"
MAX_CHARS = 80          # hard cap on screen (3 lines at hook size)
TARGET_CHARS = 60       # what the model is asked for — the briefs' examples run 25–60

PROMPT = """You are the editor of a short-form clipping page. For ONE vertical clip
(TikTok / Instagram Reels / YouTube Shorts) write the ON-SCREEN TEXT and the POST CAPTION.

Campaign brief — its rules are binding:
<brief>
{brief}
</brief>

{focus}What is said / sung in this clip (transcript):
<transcript>
{transcript}
</transcript>
{context}{avoid}
ON-SCREEN TEXT (shown on the video):
- one line, at most {max_chars} characters, English, plain text, no hashtags, no emojis
  unless the brief's examples use them
- if the brief gives example lines, pick the example group that fits THIS clip (e.g.
  discovery vs. talent vs. the song) and write in that spirit — rephrasing an example is
  fine, but vary wording across clips so the page looks organic
- specific to this moment; make people stop, react, comment; don't just narrate
- wrap the 1–2 most important words (usually a name, song or album) in *asterisks* — they
  are highlighted on screen
- if the brief requires a name/title in the text OR caption, put it in at least one of them
- positive; never put words in a real person's mouth or claim anything not in the clip

POST CAPTION (pasted when posting):
- 1–2 short sentences in English that fit the clip, then the hashtags
- MUST include every @tag, #hashtag, name or song title the brief tells posters to include
  in captions (e.g. "tag @x", "caption must include #y", "song to tag")
- otherwise add 2–4 relevant hashtags; no invented facts

Return JSON only: {{"text": "<on-screen line>", "caption": "<post caption>"}}"""


def has_auto(recipe) -> bool:
    for o in (recipe or {}).get("ops") or []:
        if o.get("op") == "text_overlay" and str(o.get("text") or "").strip() == AUTO:
            return True
    for v in (recipe or {}).get("variants") or []:
        if has_auto({"ops": v.get("ops")}):
            return True
    return False


def _norm(text):
    """Collapse whitespace; drop quotes only when they wrap the WHOLE line (a line
    like '"This Man" isn't even out yet' keeps its quotes)."""
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    while len(t) > 1 and t[0] in '"“\'' and t[-1] in '"”\'' and \
            t[1:-1].count('"') + t[1:-1].count('“') + t[1:-1].count('”') == 0:
        t = t[1:-1].strip()
    return t


def _clean(text):
    """Normalised line within MAX_CHARS. Over the cap: keep whole sentences when one
    fits, otherwise cut at a word and add an ellipsis (never a dangling half-phrase)."""
    t = _norm(text)
    if len(t) <= MAX_CHARS:
        return t
    head = t[:MAX_CHARS]
    ends = [m.end() for m in re.finditer(r"[.!?…](?=\s|$)", head)]
    if ends and ends[-1] >= 20:
        return head[:ends[-1]].strip()
    cut = head[:MAX_CHARS - 1].rsplit(" ", 1)[0].rstrip(",;:-—")
    return (cut or head[:MAX_CHARS - 1]) + "…"


def write_text(brief, transcript, context="", avoid=(), focus=None):
    """One on-screen line from the LLM, or None when it can't be had."""
    post = write_post(brief, transcript, context, avoid, focus)
    return post["text"] if post else None


def write_post(brief, transcript, context="", avoid=(), focus=None):
    """{"text": on-screen line, "caption": post caption} from ONE LLM call, or None."""
    from pipeline.llm_client import post_chat, pool_usable
    if not pool_usable():
        return None
    prompt = PROMPT.format(
        brief=(brief or "(no brief)")[:6000],
        focus=(f"Campaign focus — the people/subject of THIS campaign (if the brief names "
               f"someone else, it is a reused template: follow its rules, but write about "
               f"this subject):\n{focus[:800]}\n\n") if focus else "",
        transcript=(transcript or "(no speech)")[:4000],
        context=f"\nAbout this clip: {context[:600]}\n" if context else "",
        avoid=("\nAlready used for other clips (write something different):\n"
               + "\n".join(f"- {a}" for a in avoid) + "\n") if avoid else "",
        max_chars=TARGET_CHARS)
    messages = [{"role": "user", "content": prompt}]
    try:
        text, caption = "", ""
        for attempt in range(2):
            r = post_chat({"temperature": 0.8, "max_tokens": 500, "messages": messages},
                          timeout=60, slot=3, purpose="onscreen_text", json_mode=True)
            raw = r.json()["choices"][0]["message"]["content"] or ""
            m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            data = json.loads(m.group(0)) if m else {"text": raw}
            text = _norm(data.get("text"))
            # highlight markers are for the video only, never for the posted caption
            caption = re.sub(r"[ \t]+", " ", str(data.get("caption") or "")).replace("*", "").strip()
            if len(text.replace("*", "")) <= MAX_CHARS:
                break
            # too long for the screen: ask once for a shorter line instead of chopping it
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"The on-screen text is {len(text)} characters. "
                                            f"Rewrite it in at most {TARGET_CHARS}. JSON only."}]
        text = _clean(text)
        return {"text": text, "caption": caption} if text else None
    except Exception:
        log.exception("on-screen text generation failed")
        return None


def segments_text(segments) -> str:
    return " ".join(str(s.get("text") or "").strip() for s in segments or []).strip()


def resolve_auto_text(recipe, brief, transcript, context="", avoid=(), stored=None,
                      generate=True, focus=None, captions=None):
    """→ (recipe copy with "{auto}" filled, texts used, warnings).

    stored: texts from an earlier render of the same file, reused in order so a
    re-render keeps the approved wording. generate=False never calls the LLM:
    overlays without a stored text keep "{auto}" (display only). captions: a list the
    AI-written post captions are appended to (one per written line).
    """
    rec = copy.deepcopy(recipe or {})
    used, warnings = [], []
    stored = list(stored or [])
    keep = []
    for o in rec.get("ops") or []:
        if o.get("op") == "text_overlay" and str(o.get("text") or "").strip() == AUTO:
            if not stored and not generate:
                keep.append(o)
                continue
            if stored:
                text = stored.pop(0)
            else:
                post = write_post(brief, transcript, context, list(avoid) + used, focus=focus)
                text = post["text"] if post else None
                if post and post.get("caption") and captions is not None:
                    captions.append(post["caption"])
            if not text:
                warnings.append("on-screen text: could not be written (LLM unavailable); "
                                "overlay left out")
                continue
            o["text"] = text
            used.append(text)
        keep.append(o)
    rec["ops"] = keep
    if captions is not None and generate and not used and not has_auto(recipe) and brief:
        # no AI text on screen, but the brief may still require tags / hashtags / a song
        # name in every caption (e.g. "tag @daniaustin @jordanjramirez on every post")
        post = write_post(brief, transcript, context, list(avoid), focus=focus)
        if post and post.get("caption"):
            captions.append(post["caption"])
    return rec, used, warnings
