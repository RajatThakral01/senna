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
    post = qa.get("post") or {}
    if post.get("caption"):
        # AI-written with the brief's required tags / hashtags / song name
        title = ((qa.get("onscreen_text") or [None])[0] or clip.get("title") or "")
        title = title.replace("*", "").strip()
        return {"title": title, "caption": post["caption"],
                "hashtags": re.findall(r"#\w+", post["caption"])}
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
    from campaign.onscreen_text import has_auto, resolve_auto_text
    auto_used, auto_warn = [], []
    if has_auto(rec):
        camp = service.get_campaign(job["campaign_id"])
        rec, auto_used, auto_warn = resolve_auto_text(
            rec, camp.get("brief"), "", context=(old.get("clip") or {}).get("reason") or "",
            stored=old.get("onscreen_text"), focus=(rec.get("clips") or {}).get("focus"))
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
    if auto_used:
        r["onscreen_text"] = auto_used
    if old.get("post"):
        r["post"] = old["post"]
    if auto_warn:
        r["warnings"] = list(r.get("warnings") or []) + auto_warn
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


def reexport_job(job_id, rewrite_text=True, progress=None) -> dict:
    """Rebuild every file of a job from its already-framed sources with the job's recipe
    and today's export rules (one file per format, current text style, AI captions) —
    no re-analysis, no re-framing. Old rows/files are replaced; review decisions reset.
    rewrite_text=False reuses the on-screen lines written before.
    → {"before": n_old_files, "after": n_new_files, "failed": [...]}"""
    from campaign import service
    from campaign.clip_mode import clip_segments
    from campaign.edit_mode import job_dirs, qa_report
    from campaign.onscreen_text import resolve_auto_text, segments_text
    from campaign.qa import check_deliverable
    from campaign.recipe import variants_of
    from db.repositories import job_repo
    from pipeline.compositor import compose_all
    job = job_repo.get_job(job_id)
    camp = service.get_campaign(job["campaign_id"])
    inputs = {i["id"]: i for i in job_repo.list_inputs(job_id)}
    old = job_repo.list_deliverables(job_id)
    groups = {}
    for d in old:                       # one group per (video, clip, variant)
        groups.setdefault((d["job_input_id"], d.get("clip_id"), d["variant"]), []).append(d)
    vrecs = dict(variants_of(job["recipe"]))
    focus = (job["recipe"].get("clips") or {}).get("focus")
    seg_cache, texts_so_far, made, failed = {}, [], [], []
    for n, ((inp_id, clip_id, variant), rows) in enumerate(groups.items(), 1):
        first = rows[0]
        qa0 = first.get("qa") or {}
        inp = inputs.get(inp_id) or {}
        src = qa0.get("compose_input") or inp.get("local_path")
        name = os.path.basename(first["path"] or "clip")
        if progress:
            progress(f"re-exporting {n}/{len(groups)}", n / len(groups))
        try:
            if not src or not os.path.isfile(src):
                raise FileNotFoundError(f"source gone: {src}")
            clip = qa0.get("clip")
            segs = None
            if clip and inp.get("video_id"):
                if inp["video_id"] not in seg_cache:
                    seg_cache[inp["video_id"]] = _video_transcript_segments(inp["video_id"])
                segs = clip_segments(seg_cache[inp["video_id"]], clip["source_ranges"])
            vrec = vrecs.get(variant, job["recipe"])
            caps = []
            rec, used, warn = resolve_auto_text(
                vrec, camp.get("brief"), segments_text(segs),
                context=" ".join(x for x in ((clip or {}).get("hook"), (clip or {}).get("reason"))
                                 if x),
                avoid=texts_so_far, focus=focus, captions=caps,
                stored=None if rewrite_text else qa0.get("onscreen_text"))
            texts_so_far += used
            out_dir = os.path.dirname(first["path"])
            stem = re.sub(r"_(tiktok|reels|shorts|youtube|x|facebook|generic)"
                          r"([-+](tiktok|reels|shorts|youtube|x|facebook|generic))*$", "",
                          os.path.splitext(name)[0])
            plats = list(dict.fromkeys(p for d in rows for p in d["platform"].split("+")))
            slug = camp["slug"]
            work = os.path.join(job_dirs(slug, job_id, inp.get("idx", 0))[1],
                                f"reexport_{stem[-40:]}")
            results = compose_all(src, dict(rec, export=plats), out_dir, stem,
                                  work_dir=work, transcript_segments=segs)
            new_rows = []
            for r in results:
                r["compose_input"], r["variant"], r["onscreen_text"] = src, variant, used
                if caps:
                    r["post"] = {"caption": caps[0]}
                nd = job_repo.add_deliverable(job_id, inp_id, clip_id=clip_id, variant=variant,
                                              platform=r["platform"], path=r["output_path"],
                                              recipe_version=job.get("recipe_version"))
                q = qa_report(r)
                q["warnings"] = warn + (q.get("warnings") or [])
                if clip:
                    q["clip"] = clip
                row = job_repo.update_deliverable(nd["id"], qa=q, status="rendered")
                q = check_deliverable(row, vrec, source_path=None)
                new_rows.append(job_repo.update_deliverable(nd["id"], qa=q))
            keep = {r["path"] for r in new_rows}
            for d in rows:                       # drop the superseded rows + copies
                _delete_deliverable(d["id"])
                if d["path"] and d["path"] not in keep and os.path.isfile(d["path"]):
                    os.remove(d["path"])
            made += new_rows
        except Exception as e:
            log.exception("re-export failed for %s", name)
            failed.append(f"{name}: {e}")
    try:
        write_manifest(job_id)
    except Exception:
        log.exception("manifest failed for job %.8s", job_id)
    log.info("job %.8s re-exported: %d → %d files, %d failed", job_id, len(old), len(made),
             len(failed))
    return {"before": len(old), "after": len(made), "failed": failed}


def _delete_deliverable(deliverable_id):
    from db.connection import get_conn, release_conn
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM deliverables WHERE id = %s", (deliverable_id,))
        conn.commit()
    finally:
        release_conn(conn)
