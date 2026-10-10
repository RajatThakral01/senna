"""campaign/delivery.py — manifest, posting text and zip for a job (CAMPAIGN_PIPELINE_PLAN C6).

    write_manifest(job_id)          → <job out dir>/manifest.json + posting.txt
    build_zip(job_id, approved_only) → <job out dir>/<campaign>_<job8>.zip

Posting text per deliverable: a title (clip title / hook in clip mode, else the
campaign name + input), the campaign's CTA / handle texts, and hashtags (the
clip's suggested hashtags + any #tags written in the brief).
"""
import json
import os
import re
import zipfile

from logger import get_logger

log = get_logger("campaign.delivery")


def _hashtags(*sources, limit=8):
    tags, seen = [], set()
    for src in sources:
        if not src:
            continue
        items = src if isinstance(src, (list, tuple)) else re.findall(r"#\w+", str(src))
        for t in items:
            t = "#" + str(t).lstrip("#").strip()
            if len(t) > 1 and t.lower() not in seen:
                seen.add(t.lower())
                tags.append(t)
    return tags[:limit]


def posting_text(deliverable, campaign, recipe, input_label=None) -> dict:
    qa = deliverable.get("qa") or {}
    clip = qa.get("clip") or {}
    title = (clip.get("title") or clip.get("hook")
             or f"{campaign['name']} — {input_label or 'video'}").strip()
    lines = [o["text"] for o in (recipe or {}).get("ops") or []
             if o.get("op") == "text_overlay" and o.get("role") in ("cta", "handle")
             and o.get("text")]
    tags = _hashtags(clip.get("hashtags"), campaign.get("brief"))
    caption = " ".join([title] + lines + tags).strip()
    return {"title": title, "caption": caption, "hashtags": tags}


def manifest(job_id) -> dict:
    from campaign import service
    from campaign.edit_mode import input_labels
    from campaign.recipe import describe_recipe
    from db.repositories import job_repo
    job = job_repo.get_job(job_id)
    campaign = service.get_campaign(job["campaign_id"])
    inputs = job_repo.list_inputs(job_id)
    labels = dict(zip([i["id"] for i in inputs], input_labels([i["source"] for i in inputs])))
    by_id = {i["id"]: i for i in inputs}
    from campaign.recipe import variants_of
    vrec = dict(variants_of(job["recipe"]))
    items = []
    for d in job_repo.list_deliverables(job_id):
        qa = d.get("qa") or {}
        inp = by_id.get(d["job_input_id"]) or {}
        items.append({
            "file": os.path.basename(d["path"] or ""), "path": d["path"],
            "platform": d["platform"], "variant": d["variant"], "status": d["status"],
            "input": inp.get("source"), "mode": inp.get("mode"),
            "duration": qa.get("duration"), "size": f"{qa.get('width')}x{qa.get('height')}",
            "compliance": qa.get("compliance"),
            "failed_checks": [c["label"] + ": " + c["detail"] for c in qa.get("checks") or []
                              if c["status"] == "fail"],
            "clip": qa.get("clip"),
            "posting": posting_text(d, campaign, vrec.get(d["variant"], job["recipe"]),
                                    labels.get(d["job_input_id"])),
        })
    return {"campaign": {"name": campaign["name"], "slug": campaign["slug"]},
            "job": {"id": job["id"], "status": job["status"], "created_at": str(job["created_at"]),
                    "recipe_version": job["recipe_version"]},
            "recipe": describe_recipe(job["recipe"]),
            "inputs": [{"source": i["source"], "mode": i["mode"], "status": i["status"],
                        "error": i["error"]} for i in inputs],
            "deliverables": items}


def _job_out_dir(job_id):
    from campaign import service
    from campaign.edit_mode import job_dirs
    from db.repositories import job_repo
    job = job_repo.get_job(job_id)
    return job_dirs(service.get_campaign(job["campaign_id"])["slug"], job_id)[0]


def write_manifest(job_id) -> str:
    m = manifest(job_id)
    out = _job_out_dir(job_id)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, default=str)
    with open(os.path.join(out, "posting.txt"), "w", encoding="utf-8") as f:
        for d in m["deliverables"]:
            f.write(f"{d['file']}  [{d['platform']}]\n{d['posting']['caption']}\n\n")
    return path


def build_zip(job_id, approved_only=False) -> str:
    """Zip deliverables (all rendered + approved, or approved only) + manifest + posting."""
    from db.repositories import job_repo
    m_path = write_manifest(job_id)
    out = os.path.dirname(m_path)
    keep = ("approved",) if approved_only else ("rendered", "approved")
    rows = [d for d in job_repo.list_deliverables(job_id)
            if d["status"] in keep and d["path"] and os.path.isfile(d["path"])]
    if not rows:
        raise ValueError("no approved deliverables to zip" if approved_only
                         else "no rendered deliverables to zip")
    m = json.load(open(m_path))
    zpath = os.path.join(out, f"{m['campaign']['slug']}_{job_id[:8]}.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:   # mp4 doesn't compress
        for d in rows:
            z.write(d["path"], os.path.basename(d["path"]))
        z.write(m_path, "manifest.json")
        z.write(os.path.join(out, "posting.txt"), "posting.txt")
    log.info("zip written job=%.8s files=%d path=%s", job_id, len(rows), zpath)
    return zpath


def set_status(deliverable_id, status) -> dict:
    """Review decision: approved / rejected (or back to rendered)."""
    from db.repositories import job_repo
    if status not in ("approved", "rejected", "rendered"):
        raise ValueError(f"bad status {status!r}")
    return job_repo.update_deliverable(deliverable_id, status=status)


def _video_transcript_segments(video_id):
    """Source transcript of a clip-mode video, without touching the active workspace
    (the job worker may be using it)."""
    from db.repositories import video_repo
    from pipeline.workspace import Workspace
    row = video_repo.get_video(video_id) or {}
    ws = Workspace(video_id, row.get("source_url") or row.get("raw_path"))
    with open(ws.transcript) as f:
        return json.load(f).get("segments") or []


def rerender(deliverable_id, recipe=None) -> dict:
    """Re-render ONE deliverable, optionally with a tweaked recipe (e.g. the logo
    moved), from the same composed input; QA runs again. Returns the updated row."""
    import shutil
    from campaign import service
    from campaign.edit_mode import job_dirs, qa_report, transcribe_input, _needs_transcript
    from campaign.qa import check_deliverable
    from campaign.recipe import validate_recipe
    from db.repositories import job_repo
    from pipeline.compositor import compose
    d = job_repo.get_deliverable(deliverable_id)
    if not d:
        raise KeyError(f"unknown deliverable {deliverable_id}")
    job = job_repo.get_job(d["job_id"])
    inp = next(i for i in job_repo.list_inputs(job["id"]) if i["id"] == d["job_input_id"])
    old = d.get("qa") or {}
    from campaign.recipe import variants_of
    base = dict(variants_of(job["recipe"])).get(d["variant"], job["recipe"])
    v = validate_recipe(recipe if recipe is not None else base,
                        service.asset_catalog(job["campaign_id"]))
    if not v["ok"]:
        raise ValueError("recipe has problems: " + "; ".join(v["errors"]))
    rec = dict(v["recipe"], export=[d["platform"]])
    src = old.get("compose_input") or inp.get("local_path")
    if not src or not os.path.isfile(src):
        raise FileNotFoundError(f"the file this was rendered from is gone ({src})")
    slug = service.get_campaign(job["campaign_id"])["slug"]
    work = os.path.join(job_dirs(slug, job["id"], inp["idx"])[1], f"rerender_{deliverable_id[:8]}")
    os.makedirs(work, exist_ok=True)
    segs = None
    clip = old.get("clip")
    if clip and inp.get("video_id"):
        from campaign.clip_mode import clip_segments
        segs = clip_segments(_video_transcript_segments(inp["video_id"]), clip["source_ranges"])
    elif _needs_transcript(rec):
        segs = transcribe_input(src, os.path.dirname(work))
    tmp = os.path.join(work, "out.mp4")
    r = compose(src, rec, tmp, platform=d["platform"], work_dir=work, transcript_segments=segs)
    shutil.move(tmp, d["path"])
    r["output_path"], r["compose_input"] = d["path"], src
    qa = qa_report(r)
    if clip:
        qa["clip"] = clip
    if recipe is not None:
        qa["recipe_override"] = rec       # this file no longer follows the job recipe
    row = job_repo.update_deliverable(deliverable_id, qa=qa, status="rendered")
    row = job_repo.update_deliverable(
        deliverable_id, qa=check_deliverable(row, rec,
                                             source_path=src if not clip else None))
    log.info("re-rendered deliverable %.8s → %s (%s)", deliverable_id, d["path"],
             row["qa"].get("compliance"))
    return row
