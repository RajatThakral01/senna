"""campaign/edit_mode.py — apply a campaign recipe to supplied videos (CAMPAIGN_PIPELINE_PLAN C4).

Edit mode: every input video IS the deliverable (no clip discovery). Per input:

  1. resolve   local path / URL → local file (input.input_handler), probe it
  2. transcript once per input, only when captions are on (Whisper models stay
                loaded across the batch — release_models() at the end)
  3. render    pipeline.compositor.compose_all → one file per export platform
                (same-aspect platforms share a render), named by the recipe's
                output.filename pattern
  4. record    a `deliverables` row per file, with the compose report as qa

render_input() is the DB-free core (tests, CLI); process_input() wraps it with
job_inputs / deliverables bookkeeping.

Layout:  output/campaigns/<slug>/<job8>/<file>.mp4
         work/campaigns/<slug>/<job8>/<idx>/   (segment cuts, ASS, transcript)
"""
import json
import os
import re
import subprocess
import time

from logger import get_logger
from pipeline import media

log = get_logger("campaign.edit_mode")

OUTPUT_ROOT = os.path.join("output", "campaigns")
WORK_ROOT = os.path.join("work", "campaigns")


# ── naming ───────────────────────────────────────────────────────────────────

def _safe(s, limit=40):
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s or "")).strip("_.")
    return s[:limit] or "x"


def input_labels(sources) -> list:
    """Short unique filename-safe label per input (file stem; _2, _3 on clashes)."""
    seen, out = {}, []
    for src in sources:
        base = str(src).rstrip("/").split("?")[0]
        stem = _safe(os.path.splitext(os.path.basename(base))[0] or "input")
        n = seen.get(stem, 0) + 1
        seen[stem] = n
        out.append(stem if n == 1 else f"{stem}_{n}")
    return out


def output_basename(pattern, campaign, input_label, clip="01", variant=None, n=1) -> str:
    """Fill every filename token except {platform} (compose_all fills that) and,
    when variant is None, {variant} (see variant_basename). Missing {platform} is
    appended so platform renders never overwrite each other."""
    tokens = {"campaign": _safe(campaign), "input": _safe(input_label), "clip": _safe(clip),
              "n": f"{int(n):02d}"}
    if variant is not None:
        tokens["variant"] = _safe(variant)
    out = re.sub(r"{(\w+)}", lambda m: tokens.get(m.group(1), m.group(0)), pattern)
    if "{platform}" not in out:
        out += "_{platform}"
    return out


def variant_basename(basename, variant):
    """Resolve {variant}; without the token, non-main variants get "_<name>" before
    the platform so variant files never overwrite the main ones."""
    if "{variant}" in basename:
        return basename.replace("{variant}", _safe(variant))
    if variant == "main":
        return basename
    return basename.replace("{platform}", f"{_safe(variant)}_{{platform}}")


def job_dirs(slug, job_id, idx=None):
    out = os.path.join(OUTPUT_ROOT, slug, str(job_id)[:8])
    work = os.path.join(WORK_ROOT, slug, str(job_id)[:8])
    if idx is not None:
        work = os.path.join(work, f"{int(idx):03d}")
    return out, work


# ── transcript ───────────────────────────────────────────────────────────────

def _needs_transcript(recipe) -> bool:
    ops = {o.get("op"): o for o in (recipe or {}).get("ops") or []}
    cap = ops.get("captions")
    if not cap or not cap.get("enabled", True):
        return False
    rep = ops.get("audio.replace")
    # captions then come from the replacement track (the compositor does that)
    return not (rep and rep.get("track_path") and not rep.get("keep_original"))


def transcribe_input(path, work_dir):
    """Whole-input WhisperX segments, cached as work_dir/transcript.json."""
    cache = os.path.join(work_dir, "transcript.json")
    if os.path.isfile(cache):
        with open(cache) as f:
            return json.load(f)
    from config import ffmpeg_path
    from pipeline.transcriber import transcribe_audio
    os.makedirs(work_dir, exist_ok=True)
    wav = os.path.join(work_dir, "input.wav")
    r = subprocess.run([ffmpeg_path(), "-loglevel", "error", "-y", "-i", path, "-vn", "-ac", "1",
                        "-ar", "16000", wav], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"audio extract failed: {r.stderr[-300:]}")
    _, result = transcribe_audio(wav, output_dir=work_dir, keep_models=True)
    segs = (result or {}).get("segments") or []
    with open(cache, "w") as f:
        json.dump(segs, f)
    return segs


# ── core ─────────────────────────────────────────────────────────────────────

def render_input(path, recipe, out_dir, work_dir, basename, progress=None) -> dict:
    """Render one supplied video for every export platform (no DB).

    Returns {probe, results: [compose report per platform], warnings, elapsed}.
    Raises ValueError for an input without a video stream.
    """
    from pipeline.compositor import compose_all
    t0 = time.monotonic()
    info = media.probe(path)
    if not info["exists"]:
        raise FileNotFoundError(path)
    if not info["has_video"]:
        raise ValueError(f"{os.path.basename(path)} has no video stream "
                         f"(detected {info['kind']})")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)
    from campaign.recipe import variants_of
    variants = variants_of(recipe)
    warnings, segs = [], None
    if any(_needs_transcript(r) for _, r in variants) and info["has_audio"]:
        if progress:
            progress("transcribing", 0.1)
        try:
            segs = transcribe_input(path, work_dir)
        except Exception as e:
            log.exception("transcription failed path=%s", path)
            warnings.append(f"transcription failed ({e}); captions will be retried per render")
    if progress:
        progress("rendering", 0.3)
    results = []
    for vname, vrec in variants:
        part = compose_all(path, vrec, out_dir, variant_basename(basename, vname),
                           work_dir=work_dir if vname == "main" else os.path.join(work_dir, vname),
                           transcript_segments=segs)
        for r in part:
            r["compose_input"], r["variant"] = path, vname
        results += part
    return {"probe": info, "results": results, "warnings": warnings,
            "elapsed": round(time.monotonic() - t0, 1)}


def qa_report(res: dict) -> dict:
    """The compose report fields worth keeping on the deliverable row."""
    keys = ("width", "height", "duration", "expected_duration", "source_start", "framing",
            "applied", "skipped", "warnings", "cut", "elapsed", "reused", "compose_input",
            "body_offset", "body_duration")
    return {k: res.get(k) for k in keys if k in res}


def process_input(job, campaign, inp, label, progress=None) -> list:
    """Edit-mode run of one job input with DB bookkeeping. Returns deliverable rows.
    Raises on failure after marking the input failed (the job runner isolates it)."""
    from db.repositories import job_repo
    from input.input_handler import handle_input
    recipe = job["recipe"]
    out_dir, work_dir = job_dirs(campaign["slug"], job["id"], inp["idx"])
    job_repo.update_input(inp["id"], status="running", mode="edit", error=None)
    try:
        path = inp.get("local_path") if inp.get("local_path") and os.path.isfile(
            inp["local_path"]) else handle_input(inp["source"])
        info = media.probe(path)
        job_repo.update_input(inp["id"], local_path=path, probe=info)
        base = output_basename((recipe.get("output") or {}).get("filename")
                               or "{campaign}_{input}_{clip}_{platform}",
                               campaign["slug"], label, clip="01", n=inp["idx"] + 1)
        rep = render_input(path, recipe, out_dir, work_dir, base, progress=progress)
        job_repo.delete_deliverables(inp["id"])
        rows = []
        for r in rep["results"]:
            d = job_repo.add_deliverable(job["id"], inp["id"], variant=r.get("variant", "main"),
                                         platform=r["platform"], path=r["output_path"],
                                         recipe_version=job.get("recipe_version"))
            qa = qa_report(r)
            qa["warnings"] = rep["warnings"] + (qa.get("warnings") or [])
            rows.append(job_repo.update_deliverable(d["id"], qa=qa, status="rendered"))
        job_repo.update_input(inp["id"], status="done")
        log.info("edit input done job=%.8s idx=%d files=%d elapsed=%.1fs", job["id"],
                 inp["idx"], len(rows), rep["elapsed"])
        return rows
    except Exception as e:
        log.exception("edit input failed job=%.8s idx=%d", job["id"], inp["idx"])
        job_repo.update_input(inp["id"], status="failed", error=str(e)[:2000])
        raise
