"""
db/repositories/event_repo.py

CRUD for the `audio_events` table. Events keep SOURCE timestamps: detection
runs on pre-normalization audio and offsets are never shifted by later edits.
"""
from db.connection import get_conn, release_conn
import logging

log = logging.getLogger("db.event_repo")


def clear_events(video_id: str) -> int:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM audio_events WHERE video_id = %s", (video_id,))
            conn.commit()
            return cur.rowcount
    finally:
        release_conn(conn)


def insert_event(video_id: str, start_time: float, end_time: float,
                 peak_time=None, energy_increase=None, confidence=None,
                 method="rms", version="v1", label=None,
                 label_confidence=None, config_fingerprint="") -> str:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO audio_events
                  (video_id, start_time, end_time, peak_time, energy_increase,
                   confidence, method, version, label, label_confidence,
                   config_fingerprint)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (video_id, float(start_time), float(end_time), peak_time,
                  energy_increase, confidence, method, version, label,
                  label_confidence, config_fingerprint or ""))
            eid = cur.fetchone()[0]
            conn.commit()
            return str(eid)
    finally:
        release_conn(conn)


def get_events(video_id: str) -> list:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, video_id, start_time, end_time, peak_time,
                       energy_increase, confidence, method, version,
                       label, label_confidence, config_fingerprint
                FROM audio_events WHERE video_id = %s ORDER BY start_time
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
