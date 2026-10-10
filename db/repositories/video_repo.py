"""
db/repositories/video_repo.py

CRUD operations for the `videos` table.

Input:  Source URL, raw video path, optional campaign_id and duration.
Output: video UUID strings.
"""
from db.connection import get_conn, release_conn
import logging

log = logging.getLogger("db.video_repo")


def insert_video(source_url: str, raw_path: str, campaign_id: str = None,
                 duration_seconds: float = None) -> str:
    """Insert a new video row and return its UUID string."""
    log.debug("insert_video src=%.80s", source_url)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO videos (source_url, raw_path, campaign_id, duration_seconds)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (source_url, raw_path, campaign_id, duration_seconds))
            video_id = cur.fetchone()[0]
            conn.commit()
            log.info("insert_video done id=%.8s", str(video_id))
            return str(video_id)
    except Exception:
        log.exception("insert_video failed")
        raise
    finally:
        release_conn(conn)


def get_video(video_id: str) -> dict | None:
    """Return the video row as a dict, or None if the id does not exist."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, source_url, raw_path, duration_seconds,
                       status, campaign_id, created_at, updated_at
                FROM videos
                WHERE id = %s
            """, (video_id,))
            row = cur.fetchone()
            if row is None:
                return None
            cols = [d[0] for d in cur.description]
            d = dict(zip(cols, row))
            d["id"] = str(d["id"])
            return d
    except Exception:
        log.exception("get_video failed video=%.8s", video_id)
        raise
    finally:
        release_conn(conn)



def find_latest_by_source(source_url: str) -> dict | None:
    """Most recent video row for this source (resume target), or None."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id FROM videos WHERE source_url = %s
                ORDER BY created_at DESC LIMIT 1
            """, (source_url,))
            row = cur.fetchone()
    finally:
        release_conn(conn)
    return get_video(str(row[0])) if row else None


def update_raw_path(video_id: str, raw_path: str) -> None:
    """Point a video row at its (possibly re-located) source file."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE videos SET raw_path = %s, updated_at = NOW() WHERE id = %s",
                        (raw_path, video_id))
            conn.commit()
    finally:
        release_conn(conn)
