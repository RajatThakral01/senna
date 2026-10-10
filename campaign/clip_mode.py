"""campaign/clip_mode.py — cut a long video into campaign deliverables (CAMPAIGN_PIPELINE_PLAN C5).

Clip mode = the existing pipeline for discovery + the campaign compositor for
the look:

  1. analysis  main.run_pipeline(render=False) with the recipe's clip spec
               (count, min/max duration) applied as temporary config overrides
               and clips.focus as the analysis focus. Earlier analysis of the
               same source is reused when it was made with the same spec.
  2. framing   main.render_clips for the top `count` clips only, with no
               captions / logo / music (the compositor adds the campaign's),
               written to the job's work dir (output/<video>/ is untouched).
  3. compose   per clip: 9:16 targets start from the face-tracked vertical
               render; other aspects (1:1, 4:5, 16:9, or reframe fit_blur)
               start from a cut of the source ranges. Captions use the
               source transcript mapped onto the clip's timeline, so the
               words are never re-transcribed.
"""
import contextlib
import json
import os
import subprocess

from logger import get_logger
from campaign.edit_mode import WORK_ROOT, job_dirs, output_basename, qa_report, variant_basename

log = get_logger("campaign.clip_mode")

VERTICAL_METHODS = ("auto", "face", "center")


# ── recipe → pipeline settings ───────────────────────────────────────────────

def clip_spec(recipe) -> dict:
    c = dict((recipe or {}).get("clips") or {})
    return {"count": int(c["count"]) if c.get("count") else None,
            "min_duration": float(c["min_duration"]) if c.get("min_duration") else None,
            "max_duration": float(c["max_duration"]) if c.get("max_duration") else None,
            "focus": (c.get("focus") or "").strip()}


def config_overrides(spec) -> dict:
    """{(section, key): value} applied to main.CONFIG for the analysis run."""
    o = {}
    if spec["count"]:
        o[("discovery", "target_clips")] = spec["count"] + 2   # headroom for rejects
    if spec["min_duration"]:
        o[("discovery", "min_span_seconds")] = spec["min_duration"]
        o[("refine", "min_duration")] = spec["min_duration"]
    if spec["max_duration"]:
        o[("discovery", "max_span_seconds")] = spec["max_duration"]
        o[("refine", "max_duration")] = spec["max_duration"]
    return o


@contextlib.contextmanager
def overridden(cfg, overrides):
    """Temporarily set nested config values (restored even on error)."""
    missing = object()
    saved = []
    for (sec, key), val in overrides.items():
        section = cfg.setdefault(sec, {})
        saved.append((section, key, section.get(key, missing)))
        section[key] = val
    try:
        yield cfg
    finally:
        for section, key, old in reversed(saved):
            if old is missing:
                section.pop(key, None)
            else:
                section[key] = old


def render_flags(recipe) -> dict:
    """campaign_config for render_clips: framing only, the compositor does the rest."""
    method = next((o.get("method") for o in recipe.get("ops") or [] if o.get("op") == "reframe"),
                  "auto")
    return {"subtitles": False, "logo": None, "music": None, "fade": True,
            "layout": "center_crop" if method == "center" else "auto"}


# ── analysis reuse ───────────────────────────────────────────────────────────

def _analysis_file(video_id):
    return os.path.join(WORK_ROOT, "_analysis", f"{str(video_id)[:8]}.json")


def _analysis_key(spec):
    return {k: spec[k] for k in ("count", "min_duration", "max_duration", "focus")}


def pick_clips(clips, spec) -> list:
    """Top clips (pipeline rank order) that passed refinement and fit the spec."""
    lo = (spec["min_duration"] or 0) - 2.0
    hi = (spec["max_duration"] or 1e9) + 2.0
    out = [c for c in clips if c.get("refine_status") != "rejected"
           and lo <= _clip_duration(c) <= hi]
    return out[:spec["count"]] if spec["count"] else out


def _clip_duration(c):
    rng = c.get("source_ranges") or [[c["start_time"], c["end_time"]]]
    try:
        return sum(float(b) - float(a) for a, b in rng)
    except (TypeError, ValueError):
        return float(c.get("duration_seconds") or 0)


def redo_stage(source, spec):
    """None (reuse / fresh run) or "analyze" when the cached clips don't match."""
    from db.repositories import clip_repo, run_repo, video_repo
    prev = video_repo.find_latest_by_source(source)
    if not prev or run_repo.get_stage_status(prev["id"], "refine") != "done":
        return None
    try:
        with open(_analysis_file(prev["id"])) as f:
            stored = json.load(f)
    except (OSError, ValueError):
        stored = None
    if stored is not None:
        return None if stored == _analysis_key(spec) else "analyze"
    if spec["focus"]:
        return "analyze"           # an earlier plain run never saw this focus
    have = pick_clips(clip_repo.get_clips_for_video(prev["id"]), spec)
    return None if len(have) >= (spec["count"] or 1) else "analyze"


# ── timeline / cutting ───────────────────────────────────────────────────────

def clip_segments(segments, ranges) -> list:
    """Transcript words inside the clip's source ranges, moved onto clip time."""
    out, t_out = [], 0.0
    for a, b in ranges:
        a, b = float(a), float(b)
        for s in segments or []:
            ws = [dict(w, start=w["start"] - a + t_out, end=min(w["end"], b) - a + t_out)
                  for w in s.get("words") or []
                  if "start" in w and "end" in w and a <= w["start"] < b]
            if ws:
                out.append({"start": ws[0]["start"], "end": ws[-1]["end"], "words": ws,
                            "text": " ".join(w.get("word", "") for w in ws)})
        t_out += b - a
    return out


def cut_ranges(src, ranges, out_path):
    """Frame-accurate cut + join of source ranges at full resolution."""
    from config import ffmpeg_path
    from pipeline import media
    has_audio = media.probe(src)["has_audio"]
    parts, labels = [], ""
    for i, (a, b) in enumerate(ranges):
        parts.append(f"[0:v]trim={float(a):.3f}:{float(b):.3f},setpts=PTS-STARTPTS[v{i}]")
        labels += f"[v{i}]"
        if has_audio:
            parts.append(f"[0:a]atrim={float(a):.3f}:{float(b):.3f},asetpts=PTS-STARTPTS[a{i}]")
            labels += f"[a{i}]"
    n = len(ranges)
    parts.append(f"{labels}concat=n={n}:v=1:a={1 if has_audio else 0}"
                 + ("[v][a]" if has_audio else "[v]"))
    cmd = [ffmpeg_path(), "-loglevel", "error", "-y", "-i", src,
           "-filter_complex", ";".join(parts), "-map", "[v]"]
    if has_audio:
        cmd += ["-map", "[a]", "-c:a", "aac", "-b:a", "192k"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "16", out_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"source cut failed: {r.stderr[-300:]}")
    return out_path


def platform_groups(recipe):
    """Split export platforms into those composed from the vertical render and
    those composed from a source cut."""
    from campaign.presets import preset
    ops = {o.get("op"): o for o in recipe.get("ops") or []}
    rf = ops.get("reframe") or {}
    vertical, other = [], []
    for plat in recipe.get("export") or ["generic"]:
        aspect = rf.get("aspect") or preset(plat)["aspect"]
        if aspect == "9:16" and (rf.get("method") or "auto") in VERTICAL_METHODS:
            vertical.append(plat)
        else:
            other.append(plat)
    return vertical, other


# ── job input ────────────────────────────────────────────────────────────────

def process_input(job, campaign, inp, label, progress=None) -> list:
    """Clip-mode run of one job input with DB bookkeeping. Returns deliverable rows."""
    import main
    from db.repositories import clip_repo, job_repo
    from pipeline import workspace
    recipe = job["recipe"]
    spec = clip_spec(recipe)
    out_dir, work_dir = job_dirs(campaign["slug"], job["id"], inp["idx"])
    os.makedirs(work_dir, exist_ok=True)
    job_repo.update_input(inp["id"], status="running", mode="clip", error=None)

    def say(msg, frac=None):
        if progress:
            progress(msg, frac)

    try:
        source = inp["source"]
        redo = redo_stage(source, spec)
        info = {}
        say("analysing" + (" (re-analysis for this clip spec)" if redo else ""))
        with overridden(main.CONFIG, config_overrides(spec)):
            main.run_pipeline(source, campaign_description=spec["focus"] or None,
                              campaign_config=render_flags(recipe), render=False,
                              from_stage=redo, info=info,
                              progress=(lambda f, desc="": say(desc, 0.5 * f)) if progress else None)
        video_id, raw = info["video_id"], info["raw_path"]
        from pipeline import media
        job_repo.update_input(inp["id"], local_path=raw, probe=media.probe(raw),
                              video_id=video_id)
        os.makedirs(os.path.dirname(_analysis_file(video_id)), exist_ok=True)
        with open(_analysis_file(video_id), "w") as f:
            json.dump(_analysis_key(spec), f)

        chosen = pick_clips(clip_repo.get_clips_for_video(video_id), spec)
        if not chosen:
            raise RuntimeError("the pipeline found no clips that fit the campaign's clip spec")
        warnings = []
        if spec["count"] and len(chosen) < spec["count"]:
            warnings.append(f"only {len(chosen)} of {spec['count']} requested clips found")

        vertical, other = platform_groups(recipe)
        finals = {}
        ws = workspace.current()
        if ws.video_id != str(video_id):
            ws = workspace.activate(video_id)
        if vertical:
            say(f"framing {len(chosen)} clips", 0.5)
            saved_out = ws.output_dir
            ws.output_dir = os.path.join(work_dir, "vertical")   # render_clips writes here
            try:
                # a finished standalone render would be "reused" (its finals have the
                # standalone captions/branding): force this campaign's own render
                main.invalidate_stage(video_id, "render")
                main.render_clips(video_id, raw, render_flags(recipe),
                                  clip_numbers=[c["clip_number"] for c in chosen])
            finally:
                ws.output_dir = saved_out
                # standalone runs of this video must re-render their own finals
                main.invalidate_stage(video_id, "render")
            fresh = {c["id"]: c for c in clip_repo.get_clips_for_video(video_id)}
            for c in chosen:
                p = (fresh.get(c["id"]) or {}).get("output_path")
                vdir = os.path.abspath(os.path.join(work_dir, "vertical"))
                if p and os.path.isfile(p) and os.path.abspath(p).startswith(vdir):
                    finals[c["id"]] = p
                    c.update(fresh[c["id"]])

        with open(ws.transcript) as f:
            segments = json.load(f).get("segments") or []

        job_repo.delete_deliverables(inp["id"])
        rows = []
        from pipeline.compositor import compose_all
        from campaign.recipe import variants_of
        for k, c in enumerate(chosen, 1):
            ranges = c.get("source_ranges") or [[c["start_time"], c["end_time"]]]
            segs = clip_segments(segments, ranges)
            base = output_basename((recipe.get("output") or {}).get("filename")
                                   or "{campaign}_{input}_{clip}_{platform}",
                                   campaign["slug"], label, clip=f"{k:02d}", n=inp["idx"] + 1)
            cwork = os.path.join(work_dir, f"clip_{k:02d}")
            os.makedirs(cwork, exist_ok=True)
            say(f"composing clip {k}/{len(chosen)}", 0.6 + 0.35 * (k - 1) / len(chosen))
            results = []
            groups = [(vertical, finals.get(c["id"])), (other, None)]
            for plats, src in groups:
                if not plats:
                    continue
                if src is None:
                    if plats is vertical:
                        warnings.append(f"clip {k}: vertical framing failed; composing "
                                        "from a source cut")
                    src = os.path.join(cwork, "source_cut.mp4")
                    if not os.path.isfile(src):
                        cut_ranges(raw, ranges, src)
                for vname, vrec in variants_of(recipe):
                    part = compose_all(src, dict(vrec, export=plats), out_dir,
                                       variant_basename(base, vname),
                                       work_dir=cwork if vname == "main"
                                       else os.path.join(cwork, vname),
                                       transcript_segments=segs)
                    for r in part:
                        r["compose_input"], r["variant"] = src, vname
                    results += part
            meta = {"number": c["clip_number"], "rank": k, "hook": c.get("hook"),
                    "title": c.get("suggested_title"), "hashtags": c.get("suggested_hashtags"),
                    "reason": c.get("reason"), "source_ranges": ranges,
                    "layout": c.get("layout")}
            for r in results:
                d = job_repo.add_deliverable(job["id"], inp["id"], clip_id=c["id"],
                                             variant=r.get("variant", "main"),
                                             platform=r["platform"],
                                             path=r["output_path"],
                                             recipe_version=job.get("recipe_version"))
                qa = qa_report(r)
                qa["warnings"] = warnings + (qa.get("warnings") or [])
                qa["clip"] = meta
                rows.append(job_repo.update_deliverable(d["id"], qa=qa, status="rendered"))
        job_repo.update_input(inp["id"], status="done")
        log.info("clip input done job=%.8s idx=%d clips=%d files=%d", job["id"], inp["idx"],
                 len(chosen), len(rows))
        return rows
    except Exception as e:
        log.exception("clip input failed job=%.8s idx=%d", job["id"], inp["idx"])
        job_repo.update_input(inp["id"], status="failed", error=str(e)[:2000])
        raise
