"""
db/repositories/video_repo.py

CRUD operations for the `videos` table.

Input:  Source URL, raw video path, optional campaign_id and duration.
Output: video UUID strings; status updates; row dicts from SELECT queries.
"""
from db.connection import get_conn, release_conn


def insert_video(source_url: str, raw_path: str, campaign_id: str = None,
                 duration_seconds: float = None) -> str:
    """Insert a new video row and return its UUID string."""
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
            return str(video_id)
    finally:
        release_conn(conn)


def update_status(video_id: str, status: str) -> None:
    """Update the processing status of a video row."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE videos SET status = %s, updated_at = NOW()
                WHERE id = %s
            """, (status, video_id))
            conn.commit()
    finally:
        release_conn(conn)


def update_duration(video_id: str, duration_seconds: float) -> None:
    """Store the video duration once it has been measured."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE videos SET duration_seconds = %s, updated_at = NOW()
                WHERE id = %s
            """, (duration_seconds, video_id))
            conn.commit()
    finally:
        release_conn(conn)


def get_video(video_id: str) -> dict | None:
    """Fetch a video row by UUID. Returns None if not found."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, source_url, raw_path, duration_seconds, status, campaign_id,
                       created_at, updated_at
                FROM videos WHERE id = %s
            """, (video_id,))
            row = cur.fetchone()
            if not row:
                return None
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))
    finally:
        release_conn(conn)


def get_video_by_source(source_url: str) -> dict | None:
    """Find a video by its source URL (useful for detecting re-runs of the same video)."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, source_url, raw_path, duration_seconds, status, campaign_id,
                       created_at, updated_at
                FROM videos WHERE source_url = %s
                ORDER BY created_at DESC LIMIT 1
            """, (source_url,))
            row = cur.fetchone()
            if not row:
                return None
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))
    finally:
        release_conn(conn)
