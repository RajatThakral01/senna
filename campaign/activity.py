"""campaign/activity.py — a live, human-readable activity feed per job.

While a job runs, the pipeline logs what it is doing (stage starts/ends, clips framed,
files composed, LLM progress). capture(job_id) copies the INFO lines logged by the
job's own thread into work/campaigns/_activity/<job_id>.log, rewritten in plain words;
the Progress tab tails that file every few seconds, so the user sees movement even
during steps that take 20+ minutes. Other threads (the UI, other jobs) are filtered out.
"""
import logging
import os
import re
import threading
import time
from contextlib import contextmanager

ROOT = os.path.join("work", "campaigns", "_activity")
_SKIP = ("db.", "httpx", "httpcore", "urllib3", "lightning", "pipeline.llm_client",
         "speechbrain", "faster_whisper", "numba", "matplotlib", "PIL")
_KEEP_PREFIX = ("pipeline", "main", "campaign", "input")

_STAGE_WORDS = {
    "input": "preparing the video", "campaign": "reading the campaign",
    "transcribe": "transcribing speech", "load_model": "loading the speech model",
    "align": "aligning words to the audio", "chunk": "splitting the transcript",
    "audio_events": "listening for laughs, music and reactions",
    "embed": "indexing the transcript", "outline": "understanding the video (AI outline)",
    "analyze": "finding the best moments (AI)", "similarity": "linking related moments",
    "boundaries": "polishing where clips start and end", "render": "framing clips",
}


def path(job_id) -> str:
    return os.path.join(ROOT, f"{job_id}.log")


def friendly(name, msg) -> str | None:
    """Plain-language version of a log line, or None to leave it out."""
    m = re.search(r"▶ START stage=(\w+)", msg)
    if m:
        return f"Started: {_STAGE_WORDS.get(m.group(1), m.group(1))}"
    m = re.search(r"✔ DONE stage=(\w+) elapsed=([\d.]+)s", msg)
    if m:
        secs = float(m.group(2))
        took = f"{secs / 60:.1f} min" if secs >= 90 else f"{secs:.0f} s"
        return f"Done: {_STAGE_WORDS.get(m.group(1), m.group(1))} ({took})"
    if "stage skipped (already done)" in msg or "detector ready" in msg \
            or msg.startswith(("loading ", "pool ready", "embedding model ready")):
        return None
    m = re.search(r"compose \S+ → (\S+) platform=(\w+).* dur=([\d.]+)s", msg)
    if m:
        return f"Exported {m.group(1)} for {m.group(2)} ({float(m.group(3)):.0f} s)"
    m = re.search(r"framing done .*layout=(\w+).*elapsed=([\d.]+)s", msg)
    if m:
        return f"Framed a clip for vertical ({m.group(1)}, {float(m.group(2)):.0f} s)"
    if msg.startswith("sampled faces"):
        return "Tracking faces in a clip…"
    m = re.search(r"discovery done candidates=(\d+)", msg)
    if m:
        return f"AI proposed {m.group(1)} candidate moment(s)"
    m = re.search(r"outline done method=(\w+) sections=(\d+)", msg)
    if m:
        return f"Outline ready: {m.group(2)} section(s)"
    return re.sub(r"\s+", " ", msg)[:180]


class _Handler(logging.Handler):
    def __init__(self, job_id, thread_id):
        super().__init__(logging.INFO)
        self.thread_id = thread_id
        self.file = path(job_id)

    def emit(self, record):
        try:
            if record.thread != self.thread_id or record.levelno < logging.INFO:
                return
            name = record.name or ""
            if name.startswith(_SKIP) or not name.startswith(_KEEP_PREFIX):
                return
            text = friendly(name, record.getMessage())
            if not text:
                return
            mark = "⚠️ " if record.levelno >= logging.WARNING else ""
            with open(self.file, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%H:%M:%S')}  {mark}{text}\n")
        except Exception:
            pass


@contextmanager
def capture(job_id):
    """Record the calling thread's pipeline activity for this job."""
    os.makedirs(ROOT, exist_ok=True)
    h = _Handler(job_id, threading.get_ident())
    root = logging.getLogger()
    root.addHandler(h)
    try:
        yield h.file
    finally:
        root.removeHandler(h)


def note(job_id, text):
    """Add a line to the feed directly (e.g. from a progress callback)."""
    try:
        os.makedirs(ROOT, exist_ok=True)
        with open(path(job_id), "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')}  {text}\n")
    except OSError:
        pass


def tail(job_id, n=14) -> list:
    try:
        with open(path(job_id), encoding="utf-8") as f:
            return [ln.rstrip("\n") for ln in f.readlines()[-n:]]
    except OSError:
        return []


def seconds_since_activity(job_id) -> float | None:
    try:
        return max(0.0, time.time() - os.path.getmtime(path(job_id)))
    except OSError:
        return None
