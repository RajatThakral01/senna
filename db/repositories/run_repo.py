"""
db/repositories/run_repo.py

CRUD operations for the `pipeline_runs` table.

Tracks stage-level progress so a crashed pipeline can resume from the last
successful stage rather than restarting from scratch.

Input:  video_id UUID string, stage name (str), error messages.
Output: Stage status strings; boolean helpers used by main.py.

Stage names: transcribe | chunk | embed | analyze | similarity | render
Status values: pending | running | done | failed
"""
from db.connection import get_conn, release_conn
import logging

log = logging.getLogger("db.run_repo")


def start_stage(video_id: str, stage: str) -> str:
    """
    Record that a pipeline stage has begun.
    Upserts: if a row for (video_id, stage) already exists with status != 'done',
    it is reset to 'running'. Returns the run row UUID.
    """
    log.debug("start_stage video=%.8s stage=%s", video_id, stage)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # Delete any non-done prior record to allow a clean restart
            cur.execute("""
                DELETE FROM pipeline_runs
                WHERE video_id = %s AND stage = %s AND status != 'done'
            """, (video_id, stage))
            cur.execute("""
                INSERT INTO pipeline_runs (video_id, stage, status, started_at)
                VALUES (%s, %s, 'running', NOW())
                RETURNING id
            """, (video_id, stage))
            run_id = cur.fetchone()[0]
            conn.commit()
            log.info("start_stage video=%.8s stage=%s", video_id, stage)
            return str(run_id)
    except Exception:
        log.exception("start_stage failed video=%.8s stage=%s", video_id, stage)
        raise
    finally:
        release_conn(conn)


def complete_stage(video_id: str, stage: str) -> None:
    """Mark a stage as successfully completed."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE pipeline_runs
                SET status = 'done', completed_at = NOW()
                WHERE video_id = %s AND stage = %s AND status = 'running'
            """, (video_id, stage))
            conn.commit()
            log.info("complete_stage video=%.8s stage=%s", video_id, stage)
    except Exception:
        log.exception("complete_stage failed video=%.8s stage=%s", video_id, stage)
        raise
    finally:
        release_conn(conn)


def fail_stage(video_id: str, stage: str, error_message: str) -> None:
    """Mark a stage as failed and store the error message."""
    log.error("fail_stage video=%.8s stage=%s err=%.200s", video_id, stage, error_message)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE pipeline_runs
                SET status = 'failed', error_message = %s, completed_at = NOW()
                WHERE video_id = %s AND stage = %s AND status = 'running'
            """, (error_message, video_id, stage))
            conn.commit()
    except Exception:
        log.exception("fail_stage write failed")
        raise
    finally:
        release_conn(conn)


def get_stage_status(video_id: str, stage: str) -> str | None:
    """
    Return the status of a stage for a given video.
    Returns None if no run record exists.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT status FROM pipeline_runs
                WHERE video_id = %s AND stage = %s
                ORDER BY started_at DESC LIMIT 1
            """, (video_id, stage))
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        release_conn(conn)
