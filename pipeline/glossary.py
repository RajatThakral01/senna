"""
pipeline/glossary.py

Display-text corrections traceable to source words.

A glossary entry {match, replacement} rewrites the DISPLAY text of matching
words (case-insensitive whole-word/phrase match over the source "word"
field) while timings and the source text stay untouched. Every change is
recorded in corrections[] and persisted to transcripts/glossary.json, so
display text never invents speech silently.

Consumers (SRT export, ASS captions) prefer word["display"] when present.
"""
from logger import get_logger

log = get_logger("pipeline.glossary")


def normalize_glossary(entries):
    """Clean [{match, replacement}] entries; drop empties."""
    out = []
    for e in entries or []:
        if not isinstance(e, dict):
            continue
        match = str(e.get("match", "")).strip()
        repl = str(e.get("replacement", "")).strip()
        if match and repl and match != repl:
            out.append({"match": match, "replacement": repl})
    return out


def apply_glossary(words, glossary):
    """Apply display corrections. Returns (words, corrections).

    words: list of {word, start, end, ...} (mutated in place with "display").
    Multi-word matches span consecutive words; the replacement goes on the
    first word's display and later words get display="" (hidden downstream).
    """
    glossary = normalize_glossary(glossary)
    corrections = []
    if not glossary or not words:
        return words, corrections
    lowered = [str(w.get("word", "")) for w in words]
    for g in glossary:
        parts = g["match"].split()
        m, n = len(parts), len(lowered)
        i = 0
        while i <= n - m:
            if [x.lower() for x in lowered[i:i + m]] == [p.lower() for p in parts]:
                words[i]["display"] = g["replacement"]
                for j in range(i + 1, i + m):
                    words[j]["display"] = ""
                corrections.append({
                    "source": " ".join(lowered[i:i + m]),
                    "display": g["replacement"],
                    "start": words[i].get("start"),
                    "end": words[i + m - 1].get("end"),
                })
                i += m
            else:
                i += 1
    if corrections:
        log.info("glossary applied corrections=%d", len(corrections))
    return words, corrections


def display_text(word):
    """Display text for a word dict (display fallback to source)."""
    d = word.get("display", word.get("word", ""))
    return d if d is not None else word.get("word", "")
