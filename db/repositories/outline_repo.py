"""
db/repositories/outline_repo.py

CRUD for the `outlines` table (section- and video-level outlines).
"""
from db.connection import get_conn, release_conn
import logging

log = logging.getLogger("db.outline_repo")


def clear_outlines(video_id: str) -> int:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM outlines WHERE video_id = %s", (video_id,))
            conn.commit()
            return cur.rowcount
    finally:
        release_conn(conn)


def insert_outline(video_id: str, level: str, idx: int, start_time=None,
                   end_time=None, title=None, content=None) -> str:
    import json as _json
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO outlines (video_id, level, idx, start_time, end_time, title, content)
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                RETURNING id
            """, (video_id, level, idx, start_time, end_time, title,
                  _json.dumps(content or {})))
            oid = cur.fetchone()[0]
            conn.commit()
            return str(oid)
    finally:
        release_conn(conn)


def get_outlines(video_id: str, level: str = None) -> list:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            if level:
                cur.execute("""
                    SELECT id, level, idx, start_time, end_time, title, content
                    FROM outlines WHERE video_id = %s AND level = %s ORDER BY idx
                """, (video_id, level))
            else:
                cur.execute("""
                    SELECT id, level, idx, start_time, end_time, title, content
                    FROM outlines WHERE video_id = %s ORDER BY level, idx
                """, (video_id,))
            cols = [d[0] for d in cur.description]
            rows = []
            for row in cur.fetchall():
                d = dict(zip(cols, row))
                d["id"] = str(d["id"])
                rows.append(d)
            return rows
    finally:
        release_conn(conn)
