"""
pipeline/review.py

Review & correction operations for the Gradio workflow (Phase 6).

User corrections take precedence over generated choices. Every edit:
  - validates against the source timeline (bounds, order, min duration),
  - recomputes dependent artifacts (timeline, subtitles/ASS, edit plan),
  - bumps the edit-plan version,
  - re-renders ONLY the affected clip (via render_clips(clip_numbers=[n])).

Caption correction is cue-text editing with timings preserved: each cue keeps
its word count (words are reassigned in order, karaoke timings intact).
Boundary edits are supported for single-range clips; multi-range (stitched)
clips keep their ranges (edit the plan file directly for those).
"""
from logger import get_logger

log = get_logger("pipeline.review")


def list_review_clips(video_id: str) -> list:
    """Clips + candidates + latest plan version for review display."""
    from db.repositories import clip_repo, candidate_repo, editplan_repo
    clips = clip_repo.get_clips_for_video(video_id)
    out = []
    for c in clips:
        plan = None
        try:
            plan = editplan_repo.get_latest_plan(c["id"])
        except Exception:
            pass
        cands = []
        try:
            all_c = candidate_repo.get_candidates(video_id)
            cands = [k for k in all_c if k.get("clip_id") == c["id"]]
        except Exception:
            pass
        out.append({"clip_number": c["clip_number"], "id": c["id"],
                    "hook": c.get("hook"), "start_time": c.get("start_time"),
                    "end_time": c.get("end_time"),
                    "refine_status": c.get("refine_status"),
                    "refine_reason": c.get("refine_reason"),
                    "layout": c.get("layout"),
                    "output_path": c.get("output_path"),
                    "plan_version": (plan or {}).get("version"),
                    "candidate_scores": [k.get("scores") for k in cands]})
    return out


def list_candidates(video_id: str) -> list:
    from db.repositories import candidate_repo
    return candidate_repo.get_candidates(video_id)


def set_candidate_status(candidate_id: str, status: str, reason: str = ""):
    """Accept/reject a candidate (status recorded, never deleted)."""
    from db.repositories import candidate_repo
    if status not in ("selected", "rejected", "proposed", "shortlisted"):
        raise ValueError(f"bad candidate status {status!r}")
    candidate_repo.update_candidate_status(candidate_id, status, reason)
    log.info("candidate %s -> %s (%s)", candidate_id, status, reason)
    return True


def validate_boundary(start, end, video_duration=None):
    """Parse + validate manual boundaries. Returns (start, end) or raises."""
    try:
        s, e = float(start), float(end)
    except (TypeError, ValueError):
        raise ValueError("boundaries must be numbers (seconds)")
    if not (s >= 0 and e > s):
        raise ValueError("require 0 <= start < end")
    if e - s < 1.0:
        raise ValueError("clip must be at least 1s")
    if video_duration and e > float(video_duration) + 0.01:
        raise ValueError(f"end {e}s exceeds video duration {video_duration}s")
    return round(s, 3), round(e, 3)


def apply_boundary_edit(clip_id: str, start, end, video_duration=None):
    """User boundary override: validated, persisted, new plan version.

    Single-range clips only. Clears the clip's final so re-render runs.
    Returns the new source range.
    """
    from db.repositories import clip_repo
    from db.connection import get_conn, release_conn
    import json as _json
    s, e = validate_boundary(start, end, video_duration)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT source_ranges FROM clips WHERE id = %s", (clip_id,))
            row = cur.fetchone()
            if row and row[0] and len(row[0]) > 1:
                raise ValueError("multi-range (stitched) clips: edit the plan file directly")
            cur.execute("""
                UPDATE clips SET start_time=%s, end_time=%s, duration_seconds=%s,
                    source_ranges=%s::jsonb,
                    refine_status='manual', refine_reason='user boundary override'
                WHERE id=%s
            """, (s, e, e - s, _json.dumps([[s, e]]), clip_id))
            cur.execute("SELECT output_path FROM clips WHERE id = %s", (clip_id,))
            out = cur.fetchone()
            conn.commit()
    finally:
        release_conn(conn)
    _drop_final(out[0] if out else None)
    log.info("boundary override clip=%s range=[%.2f, %.2f]", clip_id, s, e)
    return [s, e]


def apply_layout_override(clip_id: str, layout: str):
    """Manual layout (+crop-mode) override. Returns validated layout."""
    from pipeline.framing import SUPPORTED_LAYOUTS, validate_layout
    from db.connection import get_conn, release_conn
    if str(layout or "").strip().lower() not in SUPPORTED_LAYOUTS:
        raise ValueError(f"layout must be one of {SUPPORTED_LAYOUTS}")
    layout = validate_layout(layout)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE clips SET layout=%s, layout_reason=%s WHERE id=%s",
                        (layout, "manual layout override", clip_id))
            cur.execute("SELECT output_path FROM clips WHERE id = %s", (clip_id,))
            out = cur.fetchone()
            conn.commit()
    finally:
        release_conn(conn)
    _drop_final(out[0] if out else None)
    return layout


def apply_caption_text(clip_id: str, cue_texts: list):
    """Replace cue display texts (word counts must match). Returns cue count.

    cue_texts: list of strings, one per cue in plan order. Each must contain
    exactly as many words as the stored cue (timings preserved).
    """
    from db.repositories import editplan_repo
    plan_row = editplan_repo.get_latest_plan(clip_id)
    if not plan_row:
        raise ValueError("no edit plan for clip (render it once first)")
    plan = plan_row["plan"] if isinstance(plan_row["plan"], dict) else {}
    import json as _json
    if isinstance(plan, str):
        plan = _json.loads(plan)
    cues = ((plan.get("captions") or {}).get("cues") or [])
    if len(cue_texts) != len(cues):
        raise ValueError(f"need {len(cues)} cue texts, got {len(cue_texts)}")
    for cue, text in zip(cues, cue_texts):
        new_words = str(text).split()
        old = cue.get("words", [])
        if len(new_words) != len(old):
            raise ValueError(
                f"cue word count must stay {len(old)} (got {len(new_words)}): "
                "timings are per-word and preserved")
        for w, nw in zip(old, new_words):
            w["display"] = nw
    # persist as a new plan version
    from db.repositories import editplan_repo as _er
    _er.save_plan(plan_row.get("video_id") or _clip_video(clip_id),
                  clip_id, plan)
    _drop_clip_final(clip_id)
    return len(cues)


def _clip_video(clip_id):
    from db.connection import get_conn, release_conn
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT video_id FROM clips WHERE id=%s", (clip_id,))
            row = cur.fetchone()
            return str(row[0]) if row else ""
    finally:
        release_conn(conn)


def _drop_final(path):
    if path:
        import os
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass


def _drop_clip_final(clip_id):
    from db.connection import get_conn, release_conn
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT output_path FROM clips WHERE id = %s", (clip_id,))
            row = cur.fetchone()
            conn.commit()
    finally:
        release_conn(conn)
    if row:
        _drop_final(row[0])


def get_manual_rois(clip_id: str) -> list:
    """Manual crop ROIs from the latest edit plan (authoritative at render)."""
    from db.repositories import editplan_repo
    import json as _json
    try:
        plan_row = editplan_repo.get_latest_plan(clip_id)
    except Exception:
        return []
    if not plan_row:
        return []
    plan = plan_row["plan"] if isinstance(plan_row["plan"], dict) else {}
    if isinstance(plan, str):
        try:
            plan = _json.loads(plan)
        except ValueError:
            return []
    rois = ((plan.get("framing") or {}).get("manual_rois") or [])
    return [r for r in rois if isinstance(r, dict)]


def apply_crop_override(clip_id: str, x, y, w, h,
                        t_start=None, t_end=None) -> dict:
    """Pin a manual 9:16 crop rectangle (source pixels) for a shot/time range.

    Persisted into the edit plan (new version) and authoritative at render:
    detection confidence never overrides it. Whole clip when t_start/t_end
    are omitted. Clears the clip final so re-render runs. Returns the ROI.
    """
    from pipeline.framing import validate_roi
    from db.repositories import editplan_repo
    roi = {"x": x, "y": y, "w": w, "h": h}
    if t_start is not None:
        roi["t_start"] = t_start
    if t_end is not None:
        roi["t_end"] = t_end
    roi = validate_roi(roi)  # raises on bad ratio / nonsense values
    plan_row = editplan_repo.get_latest_plan(clip_id)
    if not plan_row:
        raise ValueError("no edit plan for clip (render it once first)")
    plan = plan_row["plan"] if isinstance(plan_row["plan"], dict) else {}
    import json as _json
    if isinstance(plan, str):
        plan = _json.loads(plan)
    framing = plan.setdefault("framing", {})
    rois = framing.setdefault("manual_rois", [])
    # replace an identical time-range entry; otherwise append (last wins)
    replaced = False
    for i, old in enumerate(rois):
        if (old.get("t_start") == roi.get("t_start")
                and old.get("t_end") == roi.get("t_end")):
            rois[i] = roi
            replaced = True
            break
    if not replaced:
        rois.append(roi)
    framing["needs_review"] = False
    framing["review_note"] = "manual crop override active"
    editplan_repo.save_plan(plan_row.get("video_id") or _clip_video(clip_id),
                            clip_id, plan)
    _drop_clip_final(clip_id)
    log.info("crop override clip=%s roi=%s replaced=%s", clip_id, roi, replaced)
    return roi
