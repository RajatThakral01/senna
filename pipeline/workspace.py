"""pipeline/workspace.py

Per-video workspace paths, so runs of different videos never collide.

    downloads/<source-key>.mp4             cached source video (reused on re-runs)
    work/<label>/audio.wav                 extracted audio
    work/<label>/transcript.json|.txt      WhisperX transcript (+ glossary.json)
    work/<label>/clips/                    cuts, vertical renders, SRT, edit plans,
                                           debug videos (intermediates)
    output/<label>/clip_N_final.mp4        finished clips
    output/<label>/report.json

<label> is "<source-key>_<video_id[:8]>", e.g. "plN7JMbadRg_c9aa5363".

Before this, every run shared downloads/audio.wav, transcripts/transcript.json,
clips/ and output/: a second video was transcribed from the FIRST video's
cached audio, and render skipped clips whose output/clip_N_final.mp4 already
existed from the previous video.

The pipeline activates the workspace once the video row exists (activate());
modules ask current() for paths. With no active workspace the legacy global
paths are returned so standalone tools and unit tests keep working.
"""
import os
import re

from logger import get_logger

log = get_logger("pipeline.workspace")

WORK_ROOT = "work"
OUTPUT_ROOT = "output"
DOWNLOAD_ROOT = "downloads"

_YT_RE = re.compile(r"(?:[?&]v=|youtu\.be/|/shorts/|/live/|/embed/)([A-Za-z0-9_-]{11})")
_DRIVE_RE = re.compile(r"drive\.google\.com/.*?(?:/d/|[?&]id=)([A-Za-z0-9_-]{10,})")


def source_key(source):
    """Short filesystem-safe key for a source: YouTube id, Drive id, or file stem."""
    s = str(source or "")
    m = _YT_RE.search(s)
    if m:
        return m.group(1)
    m = _DRIVE_RE.search(s)
    if m:
        return "gd_" + m.group(1)[:24]
    stem = os.path.splitext(os.path.basename(s.split("?")[0].rstrip("/")))[0]
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", stem).strip("-")[:40]
    return stem or "video"


class Workspace:
    """Paths for one video (or the legacy global layout when video_id is None)."""

    def __init__(self, video_id=None, source=None):
        self.video_id = str(video_id) if video_id else None
        if self.video_id:
            self.label = f"{source_key(source)}_{self.video_id[:8]}"
            self.work_dir = os.path.join(WORK_ROOT, self.label)
            self.transcripts_dir = self.work_dir
            self.audio = os.path.join(self.work_dir, "audio.wav")
            self.clips_dir = os.path.join(self.work_dir, "clips")
            self.output_dir = os.path.join(OUTPUT_ROOT, self.label)
        else:  # legacy shared layout
            self.label = None
            self.work_dir = "."
            self.transcripts_dir = "transcripts"
            self.audio = os.path.join(DOWNLOAD_ROOT, "audio.wav")
            self.clips_dir = "clips"
            self.output_dir = OUTPUT_ROOT

    @property
    def transcript(self):
        return os.path.join(self.transcripts_dir, "transcript.json")

    @property
    def report(self):
        return os.path.join(self.output_dir, "report.json")

    def clip_file(self, n, suffix=".mp4"):
        """work/<label>/clips/clip_<n><suffix> (suffix e.g. ".mp4", "_vertical.mp4", ".srt")."""
        return os.path.join(self.clips_dir, f"clip_{n}{suffix}")

    def final(self, n):
        return os.path.join(self.output_dir, f"clip_{n}_final.mp4")

    def ensure(self):
        for d in (self.transcripts_dir, self.clips_dir, self.output_dir):
            os.makedirs(d, exist_ok=True)
        return self

    def __repr__(self):
        return f"Workspace({self.label or 'legacy'})"


_current = Workspace()


def activate(video_id, source=None):
    """Make video_id's workspace current. source defaults to the DB source_url."""
    global _current
    if source is None:
        try:
            from db.repositories import video_repo
            row = video_repo.get_video(video_id) or {}
            source = row.get("source_url") or row.get("raw_path")
        except Exception:
            log.debug("workspace source lookup failed", exc_info=True)
    _current = Workspace(video_id, source).ensure()
    log.info("workspace active video=%.8s work=%s output=%s",
             str(video_id), _current.work_dir, _current.output_dir)
    return _current


def current():
    return _current


def deactivate():
    global _current
    _current = Workspace()
