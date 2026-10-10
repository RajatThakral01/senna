"""campaign/jobs.py — campaign jobs: inputs → router → clip / edit mode → deliverables (C5).

    job = create_job(campaign_id, ["long.mp4", "https://youtu.be/…", "folder/"])
    run_job(job["id"])             # blocking; or submit(job["id"]) for the worker

A job freezes the campaign's recipe (and its version) when it is created, so
editing the campaign later never changes a running or finished job.

Router (per input): explicit per-input mode > job mode_policy > recipe.mode >
auto (longer than CLIP_THRESHOLD seconds → clip mode, else edit mode).

Inputs fail independently (the error is stored on the input, the rest carry
on). Re-running a job skips inputs that are already done, so a crash or a
fixed bad input resumes where it stopped. Final status: done (≥1 input
succeeded; failures listed in `error`), failed (none did) or cancelled.

One background worker thread runs queued jobs one at a time (CPU-bound
renders; Whisper / face models are shared across a batch).
"""
import os
import queue
import threading
import time

from logger import get_logger
from campaign import service
from db.repositories import job_repo

log = get_logger("campaign.jobs")

CLIP_THRESHOLD = 180.0          # seconds; longer inputs are cut into clips in auto mode
MODES = ("clip", "edit")
VIDEO_EXT = (".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi")


class JobCancelled(Exception):
    pass


# ── creation ─────────────────────────────────────────────────────────────────

def expand_sources(sources) -> list:
    """Links and files as given; folders become their video files (sorted)."""
    out = []
    for s in sources or []:
        s = str(s).strip()
        if not s:
            continue
        if os.path.isdir(s):
            out += sorted(os.path.join(s, f) for f in os.listdir(s)
                          if f.lower().endswith(VIDEO_EXT) and not f.startswith("."))
        else:
            out.append(s)
    return out


def create_job(campaign_id, sources, mode_policy="auto", modes=None) -> dict:
    """Queue a job over `sources`. modes: optional per-input overrides
    ({index: "clip"|"edit"} or a list aligned with the expanded sources)."""
    if mode_policy not in ("auto",) + MODES:
        raise ValueError(f"mode_policy must be auto, clip or edit (got {mode_policy!r})")
    recipe, version = service.ready_recipe(campaign_id)
    srcs = expand_sources(sources)
    if not srcs:
        raise ValueError("no inputs: give at least one video file, folder or link")
    if isinstance(modes, (list, tuple)):
        modes = dict(enumerate(modes))
    modes = modes or {}
    job = job_repo.insert_job(campaign_id, recipe, version, mode_policy)
    for i, s in enumerate(srcs):
        m = modes.get(i)
        job_repo.add_input(job["id"], i, s, m if m in MODES else None)
    log.info("job queued id=%.8s campaign=%.8s inputs=%d policy=%s", job["id"], campaign_id,
             len(srcs), mode_policy)
    return job


# ── routing ──────────────────────────────────────────────────────────────────

def route(probe, recipe, mode_policy="auto", override=None) -> tuple:
    """(mode, reason) for one input."""
    if override in MODES:
        return override, "chosen for this input"
    if mode_policy in MODES:
        return mode_policy, f"job runs everything in {mode_policy} mode"
    if (recipe or {}).get("mode") in MODES:
        return recipe["mode"], f"campaign recipe is {recipe['mode']} mode"
    dur = float((probe or {}).get("duration") or 0)
    if dur > CLIP_THRESHOLD:
        return "clip", f"{dur / 60:.1f} min long → cut into clips"
    return "edit", f"{dur:.0f}s long → edited as a whole"


def _resolve(inp):
    """Local file + probe for an input (downloads links once, cached)."""
    from input.input_handler import handle_input
    from pipeline import media
    path = inp.get("local_path")
    if not (path and os.path.isfile(path)):
        path = handle_input(inp["source"])
    return path, media.probe(path)


# ── running ──────────────────────────────────────────────────────────────────

def run_job(job_id, progress=None, cancel_event=None) -> dict:
    """Run (or resume) a job to the end. progress(fraction, message) optional."""
    from campaign import clip_mode, edit_mode
    job = job_repo.get_job(job_id)
    if not job:
        raise KeyError(f"unknown job {job_id}")
    campaign = service.get_campaign(job["campaign_id"])
    inputs = job_repo.list_inputs(job_id)
    labels = edit_mode.input_labels([i["source"] for i in inputs])
    job_repo.update_job(job_id, status="running", started_at=True, error=None,
                        message="starting")
    total = max(1, len(inputs))
    failures, t0 = [], time.monotonic()

    def report(i, msg, inner=None):
        frac = min(0.99, (i + min(max(inner or 0.0, 0.0), 1.0)) / total)
        text = f"[{i + 1}/{total}] {msg}" if i < total else msg
        job_repo.update_job(job_id, progress=frac, message=text[:500])
        if progress:
            progress(frac, text)

    try:
        for i, inp in enumerate(inputs):
            if cancel_event is not None and cancel_event.is_set():
                raise JobCancelled()
            if inp["status"] == "done":
                continue
            name = os.path.basename(str(inp["source"]).rstrip("/")) or inp["source"]
            try:
                report(i, f"{name}: resolving")
                path, probe = _resolve(inp)
                if not probe.get("has_video"):
                    raise ValueError(f"{name} has no video stream (detected {probe['kind']})")
                mode, why = route(probe, job["recipe"], job["mode_policy"], inp.get("mode"))
                inp = job_repo.update_input(inp["id"], local_path=path, probe=probe, mode=mode)
                log.info("job %.8s input %d → %s mode (%s)", job_id, i, mode, why)
                report(i, f"{name}: {mode} mode — {why}")
                runner = clip_mode if mode == "clip" else edit_mode
                rows = runner.process_input(
                    job, campaign, inp, labels[i],
                    progress=lambda m, f=None, i=i, n=name: report(i, f"{n}: {m}", f))
                report(i, f"{name}: checking {len(rows)} file(s)", 0.97)
                from campaign.qa import check_deliverable
                from campaign.recipe import variants_of
                vrec = dict(variants_of(job["recipe"]))
                n_fail = 0
                for row in rows:
                    qa = check_deliverable(row, vrec.get(row["variant"], job["recipe"]),
                                           source_path=path if mode == "edit" else None)
                    n_fail += qa["compliance"] == "fail"
                    job_repo.update_deliverable(row["id"], qa=qa)
                report(i + 1, f"{name}: {len(rows)} file(s) rendered"
                       + (f", {n_fail} failed QA" if n_fail else ""))
            except JobCancelled:
                raise
            except Exception as e:
                log.exception("job %.8s input %d failed", job_id, i)
                job_repo.update_input(inp["id"], status="failed", error=str(e)[:2000])
                failures.append(f"{name}: {e}")
    except JobCancelled:
        job_repo.update_job(job_id, status="cancelled", finished_at=True,
                            message="cancelled", error="; ".join(failures)[:4000] or None)
        log.info("job %.8s cancelled", job_id)
        return job_repo.get_job(job_id)
    finally:
        try:
            from pipeline.transcriber import release_models
            release_models()
        except Exception:
            pass
    done = sum(1 for x in job_repo.list_inputs(job_id) if x["status"] == "done")
    try:
        from campaign.delivery import write_manifest
        write_manifest(job_id)
    except Exception:
        log.exception("manifest failed for job %.8s", job_id)
    n_files = len(job_repo.list_deliverables(job_id))
    status = "done" if done else "failed"
    msg = (f"{done}/{len(inputs)} inputs done, {n_files} files, "
           f"{(time.monotonic() - t0) / 60:.1f} min")
    job_repo.update_job(job_id, status=status, progress=1.0 if done else 0.0, message=msg,
                        error="; ".join(failures)[:4000] or None, finished_at=True)
    log.info("job %.8s %s — %s", job_id, status, msg)
    if progress:
        progress(1.0, msg)
    return job_repo.get_job(job_id)


def status(job_id) -> dict:
    """Job row + inputs + deliverables (UI / CLI)."""
    job = job_repo.get_job(job_id)
    if not job:
        raise KeyError(f"unknown job {job_id}")
    return {"job": job, "inputs": job_repo.list_inputs(job_id),
            "deliverables": job_repo.list_deliverables(job_id)}


# ── background worker ────────────────────────────────────────────────────────

class Worker:
    """One thread, one job at a time. submit() queues; cancel() stops between inputs."""

    def __init__(self):
        self._q = queue.Queue()
        self._cancel = {}
        self._thread = None
        self._lock = threading.Lock()
        self.current = None

    def _ensure(self):
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._loop, name="campaign-jobs",
                                                daemon=True)
                self._thread.start()

    def submit(self, job_id):
        self._cancel[job_id] = threading.Event()
        job_repo.update_job(job_id, status="queued", message="waiting for the worker")
        self._q.put(job_id)
        self._ensure()

    def cancel(self, job_id):
        ev = self._cancel.get(job_id)
        if ev:
            ev.set()
        if job_repo.get_job(job_id)["status"] == "queued" and self.current != job_id:
            job_repo.update_job(job_id, status="cancelled", message="cancelled before start",
                                finished_at=True)

    def _loop(self):
        while True:
            job_id = self._q.get()
            ev = self._cancel.get(job_id)
            if (ev and ev.is_set()) or job_repo.get_job(job_id)["status"] == "cancelled":
                continue
            self.current = job_id
            try:
                run_job(job_id, cancel_event=ev)
            except Exception as e:
                log.exception("worker: job %.8s crashed", job_id)
                job_repo.update_job(job_id, status="failed", error=str(e)[:4000],
                                    finished_at=True)
            finally:
                self.current = None
                self._cancel.pop(job_id, None)


_worker = None


def worker() -> Worker:
    global _worker
    if _worker is None:
        _worker = Worker()
    return _worker


def submit(job_id):
    worker().submit(job_id)


def resume_interrupted() -> list:
    """Re-queue jobs left running/queued by a crash or restart (app start-up)."""
    ids = [j["id"] for st in ("running", "queued") for j in job_repo.list_jobs(status=st)]
    for jid in reversed(ids):
        submit(jid)
    return ids
