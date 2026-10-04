"""Centralized logging for Viral Clips Automator.

Usage:
    from logger import get_logger, log_stage, bind

    log = get_logger(__name__)
    log.info("transcribe started", extra={"video_id": vid, "stage": "transcribe"})

    with log_stage(__name__, "transcribe", video_id=vid):
        ...  # logs START/DONE (with elapsed) or FAILED (with exc)

    # Per-clip context:
    clog = bind(__name__, video_id=vid, stage="render", clip_number=n)
    clog.info("verticalize done")

Setup is idempotent and reads config.yaml [logging] + env overrides:
    LOG_LEVEL (DEBUG/INFO/WARNING/ERROR), LOG_FILE, LOG_TO_CONSOLE.
"""
import logging
import os
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler

_configured = False
_log_file = None
_log_level = None

_FORMAT = (
    "%(asctime)s | %(levelname)-7s | %(name)s"
    "%(ctx)s | %(message)s"
)


class _CtxFormatter(logging.Formatter):
    def format(self, record):
        parts = []
        for key in ("video_id", "stage", "clip_number"):
            val = getattr(record, key, None)
            if val is not None:
                short = str(val)[:8] if key == "video_id" and len(str(val)) > 8 else val
                parts.append(f" {key}={short}")
        record.ctx = "".join(parts)
        return super().format(record)


def _resolve_settings():
    level = os.getenv("LOG_LEVEL", "").upper() or None
    log_file = os.getenv("LOG_FILE", "") or None
    to_console = os.getenv("LOG_TO_CONSOLE", "").lower() not in ("0", "false", "no")
    if level is None or log_file is None:
        try:
            from config import get_config
            cfg = get_config().get("logging", {})
            level = level or str(cfg.get("level", "INFO")).upper()
            log_file = log_file or cfg.get("file", "logs/viral-clips.log")
            if "console" in cfg and not os.getenv("LOG_TO_CONSOLE"):
                to_console = bool(cfg["console"])
        except Exception:
            level = level or "INFO"
            log_file = log_file or "logs/viral-clips.log"
    return level, log_file, to_console


def setup_logging(level=None, log_file=None, to_console=True, force=False):
    """Configure root + viral-clips handlers once. Safe to call repeatedly."""
    global _configured, _log_file, _log_level
    if _configured and not force:
        return
    cfg_level, cfg_file, cfg_console = _resolve_settings()
    level = (level or cfg_level or "INFO").upper()
    log_file = log_file or cfg_file
    # explicit arg wins, else env/cfg
    if level is None:
        level = "INFO"
    numeric = getattr(logging, level, logging.INFO)

    root = logging.getLogger()
    root.setLevel(numeric)
    fmt = _CtxFormatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    # Avoid duplicate handlers on re-setup
    if force:
        for h in list(root.handlers):
            root.removeHandler(h)

    if to_console and cfg_console:
        ch = logging.StreamHandler()
        ch.setLevel(numeric)
        ch.setFormatter(fmt)
        root.addHandler(ch)

    if log_file:
        try:
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
            fh = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024,
                                     backupCount=3, encoding="utf-8")
            fh.setLevel(numeric)
            fh.setFormatter(fmt)
            root.addHandler(fh)
        except OSError:
            pass  # console-only fallback

    # Quiet noisy third-party libs (keep warnings/errors only)
    for noisy in ("urllib3", "requests", "httpx", "httpcore", "gradio", "uvicorn"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True
    _log_file = log_file
    _log_level = level


def get_logger(name):
    """Return a module logger, ensuring logging is configured."""
    if not _configured:
        setup_logging()
    return logging.getLogger(name)


class _BoundAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        extra = dict(self.extra)
        if "extra" in kwargs and kwargs["extra"]:
            extra.update(kwargs["extra"])
        kwargs["extra"] = extra
        return msg, kwargs


def bind(name_or_logger, video_id=None, stage=None, clip_number=None, **extra):
    """Return a LoggerAdapter pre-filled with pipeline context fields."""
    base = name_or_logger if isinstance(name_or_logger, logging.Logger) else get_logger(name_or_logger)
    ctx = {}
    if video_id is not None:
        ctx["video_id"] = video_id
    if stage is not None:
        ctx["stage"] = stage
    if clip_number is not None:
        ctx["clip_number"] = clip_number
    ctx.update(extra)
    return _BoundAdapter(base, ctx)


@contextmanager
def log_stage(name_or_logger, stage, video_id=None, clip_number=None, **fields):
    """Log stage entry/exit with elapsed time; FAILED on exception (re-raised)."""
    log = bind(name_or_logger, video_id=video_id, stage=stage,
               clip_number=clip_number, **fields)
    start = time.monotonic()
    log.info(f"▶ START stage={stage}")
    try:
        yield log
    except Exception:
        elapsed = time.monotonic() - start
        log.exception(f"✖ FAILED stage={stage} elapsed={elapsed:.1f}s")
        raise
    else:
        elapsed = time.monotonic() - start
        log.info(f"✔ DONE stage={stage} elapsed={elapsed:.1f}s")
