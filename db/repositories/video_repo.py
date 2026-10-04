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

