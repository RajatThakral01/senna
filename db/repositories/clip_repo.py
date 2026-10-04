"""
db/repositories/clip_repo.py

CRUD operations for the `clips` table.

Input:  Clip metadata produced by pipeline/analyzer.py (timestamps, hook, reason, etc.),
        embedding vectors, and rendered output paths.
Output: Clip UUID strings; clip row dicts for report generation.
"""
from db.connection import get_conn, release_conn


def insert_clip(video_id: str, clip_number: int, start_time: float, end_time: float,
                duration_seconds: float = None, hook: str = None, reason: str = None,
                suggested_title: str = None, suggested_hashtags=None,
                source_chunk_ids=None, embedding=None, **kwargs) -> str:
    """
    Insert a clip row and return its UUID string.

    suggested_hashtags: str (raw hashtag string) or list of str — stored as TEXT[]
    source_chunk_ids:   list of UUID strings
    embedding:          list of floats (pgvector)
    """
    # Normalise hashtags: split string "#a #b #c" into list, or use list as-is
    if isinstance(suggested_hashtags, str):
        hashtags_list = suggested_hashtags.split() if suggested_hashtags else []
    elif isinstance(suggested_hashtags, list):
        hashtags_list = suggested_hashtags
    else:
        hashtags_list = []

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO clips
                  (video_id, clip_number, start_time, end_time, duration_seconds,
                   hook, reason, suggested_title, suggested_hashtags,
                   source_chunk_ids, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::text[], %s::uuid[], %s)
                RETURNING id
            """, (
                video_id, clip_number, start_time, end_time, duration_seconds,
                hook, reason, suggested_title, hashtags_list,
                source_chunk_ids, embedding
            ))
            clip_id = cur.fetchone()[0]
            conn.commit()
            return str(clip_id)
    finally:
        release_conn(conn)


def update_output_path(clip_id: str, output_path: str) -> None:
    """Store the rendered .mp4 path once FFmpeg finishes a clip."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE clips SET output_path = %s WHERE id = %s",
                (output_path, clip_id)
            )
            conn.commit()
    finally:
        release_conn(conn)


def update_embedding(clip_id: str, embedding: list) -> None:
    """Write the embedding vector for a clip."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE clips SET embedding = %s WHERE id = %s",
                (embedding, clip_id)
            )
            conn.commit()
    finally:
        release_conn(conn)


def get_clips_for_video(video_id: str) -> list:
    """Return all clips for a video ordered by clip_number."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, video_id, clip_number, start_time, end_time, duration_seconds,
                       hook, reason, suggested_title, suggested_hashtags,
                       source_chunk_ids, output_path, created_at,
                       embedding::text as embedding
                FROM clips
                WHERE video_id = %s
                ORDER BY clip_number ASC
            """, (video_id,))
            cols = [d[0] for d in cur.description]
            rows = []
            import json
            for row in cur.fetchall():
                d = dict(zip(cols, row))
                d["id"] = str(d["id"])
                d["video_id"] = str(d["video_id"])
                if d.get("embedding"):
                    d["embedding"] = json.loads(d["embedding"])
                rows.append(d)
            return rows
    finally:
        release_conn(conn)
