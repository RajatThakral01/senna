"""
pipeline/fusion.py

Candidate fusion + global ranking (Phase 3).

Fuses transcript-outline and audio-event candidates into one pool, then:
  1. merges substantially overlapping proposals (temporal IoU),
  2. scores every candidate on editorial components (heuristics — NOT
     calibrated predictions of views/virality),
  3. selects a diverse final set (temporal + content diversity; same-topic
     but distinct moments are preserved),
  4. enforces a quality floor: fewer clips when quality is insufficient,
     with recorded reasons,
  5. after boundary refinement, deduplicates again (boundaries can expand).

Editorial quality dominates: audio-spike evidence is a small bonus and can
never rescue an incoherent passage.
"""
import math
import time

from logger import get_logger

log = get_logger("pipeline.fusion")

FILLER_OPENERS = ("so", "and", "well", "you know", "um", "uh", "like",
                  "okay", "right", "now")
PRONOUN_OPENERS = ("it", "this", "that", "these", "those", "they", "them",
                   "he", "she", "his", "her", "its", "there")


def _ranges(c):
    return [(float(a), float(b)) for a, b in (c.get("source_ranges") or [])]


def _span(ranges):
    if not ranges:
        return 0.0
    return sum(b - a for a, b in ranges)


def temporal_iou(c1, c2):
    """IoU over source ranges (union timeline)."""
    r1, r2 = _ranges(c1), _ranges(c2)
    if not r1 or not r2:
        return 0.0
    inter = 0.0
    for a1, b1 in r1:
        for a2, b2 in r2:
            lo, hi = max(a1, a2), min(b1, b2)
            if hi > lo:
                inter += hi - lo
    union = _span(r1) + _span(r2) - inter
    return inter / union if union > 0 else 0.0


def merge_overlaps(candidates, iou_threshold=0.7):
    """Merge proposals with IoU >= threshold. Returns (kept, n_merged).

    Winner = lower uncertainty; provenance merged (sources + ids recorded).
    Losers are returned separately so callers can mark them rejected/merged.
    """
    remaining = sorted(candidates,
                       key=lambda c: float(c.get("uncertainty", 0.5)))
    kept, dropped = [], []
    for cand in remaining:
        placed = False
        for k in kept:
            if temporal_iou(cand, k) >= iou_threshold:
                prov = k.setdefault("provenance", {})
                prov.setdefault("merged_from", []).append({
                    "hook": (cand.get("hook") or "")[:120],
                    "source": cand.get("provenance", {}).get("source")
                    if isinstance(cand.get("provenance"), dict) else None,
                    "origin": cand.get("source", "?")})
                sids = sorted(set(k.get("sentence_ids", [])) |
                              set(cand.get("sentence_ids", [])))
                k["sentence_ids"] = sids
                cand["fusion_status"] = "rejected"
                cand["fusion_reason"] = "merged into overlapping proposal"
                dropped.append((cand, k))
                placed = True
                break
        if not placed:
            kept.append(cand)
    return kept, dropped


def _opening_strength(c):
    hook = (c.get("hook") or "").strip()
    if not hook:
        return 0.3
    first = hook.split()[0].strip(" ,.!?\"'").lower() if hook.split() else ""
    score = 0.6
    if first in FILLER_OPENERS:
        score -= 0.35
    if hook.rstrip().endswith("?") or hook.rstrip().endswith("!"):
        score += 0.15
    n = len(hook.split())
    if 3 <= n <= 20:
        score += 0.1
    return round(min(1.0, max(0.0, score)), 3)


def _standalone(c):
    if (c.get("required_context") or "").strip():
        base = 0.3
    else:
        base = 0.8
    hook = (c.get("hook") or "").strip()
    if hook:
        first = hook.split()[0].strip(" ,.!?\"'").lower()
        if first in PRONOUN_OPENERS:
            base -= 0.3
    return round(min(1.0, max(0.0, base)), 3)


def _payoff(c):
    return 1.0 if (c.get("payoff") or "").strip() else 0.4


def _completeness(c):
    # pre-refine heuristic: low uncertainty + payoff present reads complete
    u = float(c.get("uncertainty", 0.5))
    s = 1.0 - 0.6 * min(1.0, max(0.0, u))
    if not (c.get("payoff") or "").strip():
        s -= 0.2
    return round(min(1.0, max(0.0, s)), 3)


def _relevance(c, campaign_text=""):
    ct = (campaign_text or "").lower()
    if not ct:
        return 0.5  # neutral when no campaign guidance
    import re
    words = set(re.findall(r"[a-z]{3,}", ct))
    if not words:
        return 0.5
    text = f"{c.get('hook', '')} {c.get('main_idea', '')}".lower()
    hit = sum(1 for w in words if w in text)
    return round(min(1.0, hit / max(1, min(len(words), 6))), 3)


def _editability(c):
    span = _span(_ranges(c))
    if 20 <= span <= 90:
        base = 1.0
    elif span < 20:
        base = max(0.2, span / 20)
    else:
        base = max(0.2, 1.0 - (span - 90) / 120)
    prov = c.get("provenance") or {}
    if isinstance(prov, dict) and prov.get("energy_increase"):
        base = min(1.0, base + 0.1)
    return round(base, 3)


def _audio_support(c):
    prov = c.get("provenance") or {}
    if isinstance(prov, dict) and prov.get("energy_increase"):
        return 0.8
    if (c.get("source") or "") == "audio_event":
        return 0.7
    return 0.3


DEFAULT_WEIGHTS = {
    "opening": 0.22, "standalone": 0.24, "payoff": 0.16,
    "completeness": 0.16, "relevance": 0.08, "editability": 0.09,
    "audio_support": 0.05,
}


def score_candidate(c, campaign_text="", weights=None):
    """Editorial component scores + weighted total. Total in scores['total']."""
    w = dict(DEFAULT_WEIGHTS)
    w.update(weights or {})
    parts = {
        "opening": _opening_strength(c),
        "standalone": _standalone(c),
        "payoff": _payoff(c),
        "completeness": _completeness(c),
        "relevance": _relevance(c, campaign_text),
        "editability": _editability(c),
        "audio_support": _audio_support(c),
    }
    total = round(sum(parts[k] * w.get(k, 0) for k in parts), 4)
    scores = dict(parts)
    scores["total"] = total
    scores["weights"] = {k: w.get(k, 0) for k in parts}
    scores["note"] = ("editorial heuristic, not a virality prediction; "
                      "audio evidence is a small bonus only")
    return scores


def _content_similarity(texts):
    """Cosine similarity matrix via local embeddings. Falls back to token IoU."""
    try:
        from pipeline.embedder import embed_texts
        vecs = embed_texts(texts)
        import math as _m
        sims = []
        for i in range(len(vecs)):
            row = []
            for j in range(len(vecs)):
                a, b = vecs[i], vecs[j]
                dot = sum(x * y for x, y in zip(a, b))
                na = _m.sqrt(sum(x * x for x in a)) or 1.0
                nb = _m.sqrt(sum(x * x for x in b)) or 1.0
                row.append(dot / (na * nb))
            sims.append(row)
        return sims
    except Exception:
        log.debug("embedding similarity unavailable, token-IoU fallback", exc_info=True)
        toks = [set(t.lower().split()) for t in texts]
        sims = []
        for a in toks:
            row = []
            for b in toks:
                u = a | b
                row.append(len(a & b) / len(u) if u else 0.0)
            sims.append(row)
        return sims


def rank_and_select(candidates, campaign_text="", cfg=None):
    """Full fusion: merge -> score -> diverse select with quality floor.

    Returns (selected, report) where report records merged/dropped/floor
    decisions with reasons. target_clips is a target/maximum: fewer clips
    are returned when quality is insufficient.
    """
    cfg = cfg or {}
    f_cfg = cfg.get("fusion", {}) if isinstance(cfg, dict) else {}
    d_cfg = cfg.get("discovery", {}) if isinstance(cfg, dict) else {}
    t0 = time.monotonic()
    target = int(d_cfg.get("target_clips", 8) or 8)
    merge_iou = float(f_cfg.get("merge_iou", 0.7))
    div_iou = float(f_cfg.get("diversity_iou", 0.5))
    div_sim = float(f_cfg.get("diversity_sim", 0.92))
    min_score = float(f_cfg.get("min_score", 0.45))
    weights = f_cfg.get("weights", {})

    kept, dropped = merge_overlaps(candidates, merge_iou)
    for cand in kept:
        cand["scores"] = score_candidate(cand, campaign_text, weights)
    ranked = sorted(kept, key=lambda c: c["scores"]["total"], reverse=True)

    texts = [f"{c.get('hook', '')} {c.get('main_idea', '')}" for c in ranked]
    sims = _content_similarity(texts) if len(ranked) > 1 else [[1.0]]

    selected, skipped = [], []
    for i, cand in enumerate(ranked):
        total = cand["scores"]["total"]
        if total < min_score:
            cand["fusion_status"] = "rejected"
            cand["fusion_reason"] = (f"below quality floor {min_score} "
                                     f"(total={total})")
            skipped.append((cand, cand["fusion_reason"]))
            continue
        clash = None
        for j, sel in enumerate(selected):
            si = next(k for k, c in enumerate(ranked) if c is sel)
            iou = temporal_iou(cand, sel)
            sim = sims[i][si]
            # near-duplicate only: high overlap, or moderate overlap + near-identical text
            if iou >= div_iou or (iou >= 0.3 and sim >= div_sim):
                clash = (sel, f"near-duplicate of selected (iou={iou:.2f}, sim={sim:.2f})")
                break
        if clash:
            cand["fusion_status"] = "rejected"
            cand["fusion_reason"] = clash[1]
            skipped.append((cand, clash[1]))
            continue
        cand["fusion_status"] = "shortlisted"
        cand["fusion_reason"] = f"ranked #{len(selected) + 1} (total={total})"
        selected.append(cand)
        if len(selected) >= target:
            # remaining ranked candidates are surplus, not failures
            for rest in ranked[i + 1:]:
                rest["fusion_status"] = "proposed"
                rest["fusion_reason"] = f"surplus beyond target={target}"
                skipped.append((rest, rest["fusion_reason"]))
            break
    report = {
        "merged": [(d[0].get("hook", "")[:80], d[1].get("hook", "")[:80]) for d in dropped],
        "skipped": [(c.get("hook", "")[:80], reason) for c, reason in skipped],
        "selected_scores": [c["scores"]["total"] for c in selected],
        "elapsed": round(time.monotonic() - t0, 2),
    }
    log.info("fusion candidates=%d merged=%d selected=%d skipped=%d",
             len(candidates), len(dropped), len(selected), len(skipped))
    return selected, report


def deduplicate_refined(clips, iou_threshold=0.5, sim_threshold=0.92,
                        coverage_threshold=0.6):
    """Post-refinement dedup (boundaries can expand). Returns (kept, dropped).

    Drops a clip when its refined ranges are near-duplicates of a kept one:
    IoU >= threshold, or overlap covers >= coverage_threshold of the smaller
    clip, or moderate overlap + near-identical text. Keeps higher
    fusion_score (from clips[].provenance); marks the rest so callers can
    set refine_status=rejected with an explicit reason.
    Same-topic but distinct moments survive: only near-duplicates drop.
    """
    def _score(c):
        try:
            return float((c.get("provenance") or {}).get("fusion_score", 0.5))
        except (TypeError, ValueError):
            return 0.5

    def _ranges_of(c):
        if c.get("source_ranges"):
            return [(float(a), float(b)) for a, b in c["source_ranges"]]
        try:
            return [(float(c["start_time"]), float(c["end_time"]))]
        except (KeyError, TypeError, ValueError):
            return []

    ordered = sorted(clips, key=_score, reverse=True)
    texts = [f"{c.get('hook', '')} {c.get('reason', '')}" for c in ordered]
    sims = _content_similarity(texts) if len(ordered) > 1 else [[1.0]]
    kept, dropped = [], []
    for i, cand in enumerate(ordered):
        clash = None
        for j, sel in enumerate(kept):
            si = next(k for k, c in enumerate(ordered) if c is sel)
            # iou over refined ranges
            r1, r2 = _ranges_of(cand), _ranges_of(sel)
            inter = sum(max(0, min(b1, b2) - max(a1, a2))
                        for a1, b1 in r1 for a2, b2 in r2)
            union = (sum(b - a for a, b in r1) + sum(b - a for a, b in r2) - inter)
            iou = inter / union if union > 0 else 0.0
            small = min(sum(b - a for a, b in r1), sum(b - a for a, b in r2))
            coverage = inter / small if small > 0 else 0.0
            if (iou >= iou_threshold or coverage >= coverage_threshold
                    or (iou >= 0.3 and sims[i][si] >= sim_threshold)):
                clash = sel
                clash_info = (f"duplicate of clip {sel.get('clip_number')} "
                              f"(iou={iou:.2f}, cov={coverage:.2f})")
                break
        if clash is None:
            kept.append(cand)
        else:
            dropped.append((cand, clash_info))
    # restore clip-number order for downstream rendering
    kept.sort(key=lambda c: c.get("clip_number", 0))
    return kept, dropped


def temporal_overlap_seconds(c1, c2):
    r1, r2 = _ranges(c1), _ranges(c2)
    return sum(max(0, min(b1, b2) - max(a1, a2)) for a1, b1 in r1 for a2, b2 in r2)


def limit_stitch_ranges(primary, continuations, cfg=None):
    """Cap non-contiguous assembly (similarity stitching).

    primary: (start, end). continuations: [{start_time, end_time, ...}].
    Drops continuations starting more than max_stitch_gap_seconds after the
    primary end, and stops adding once output would exceed max_total_seconds.
    A distant passage is related context, never proof of continuation.
    Returns (kept, dropped) with explicit reasons.
    """
    cfg = cfg or {}
    s_cfg = cfg.get("similarity", {}) if isinstance(cfg, dict) else {}
    max_gap = float(s_cfg.get("max_stitch_gap_seconds", 45))
    max_total = float(s_cfg.get("max_total_seconds", 90))
    p0, p1 = float(primary[0]), float(primary[1])
    kept, dropped = [], []
    total = p1 - p0
    for seg in sorted(continuations,
                      key=lambda s: float(s.get("start_time", 0))):
        try:
            a, b = float(seg["start_time"]), float(seg["end_time"])
        except (KeyError, TypeError, ValueError):
            dropped.append((seg, "unreadable timestamps"))
            continue
        if b <= a:
            dropped.append((seg, "empty range"))
            continue
        gap = a - p1
        if gap > max_gap:
            dropped.append((seg, f"gap {gap:.1f}s > max {max_gap:.0f}s "
                                 "(related, not continuation)"))
            continue
        if total + (b - a) > max_total:
            dropped.append((seg, f"would exceed max_total {max_total:.0f}s"))
            continue
        kept.append(seg)
        total += b - a
        p1 = max(p1, b)
    return kept, dropped
