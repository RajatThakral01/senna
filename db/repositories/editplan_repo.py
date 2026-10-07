"""
db/repositories/editplan_repo.py

CRUD for the `edit_plans` table. Plans are versioned per clip; the latest
version drives preview and final export identically (Phase 6 re-renders from
the stored plan after user edits).
"""
from db.connection import get_conn, release_conn


def next_version(clip_id: str) -> int:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COALESCE(MAX(version), 0) + 1 FROM edit_plans WHERE clip_id = %s",
                        (clip_id,))
            return cur.fetchone()[0]
    finally:
        release_conn(conn)


def save_plan(video_id: str, clip_id: str, plan: dict):
    """Persist a new plan version. Returns (id, version).

    NOTE: the input dict is copied, not mutated; use the returned version.
    """
    import json as _json
    version = next_version(clip_id)
    stored = dict(plan or {})
    stored["version"] = version
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO edit_plans (video_id, clip_id, version, plan)
                VALUES (%s, %s, %s, %s::jsonb)
                RETURNING id
            """, (video_id, clip_id, version, _json.dumps(stored)))
            pid = cur.fetchone()[0]
            conn.commit()
            return str(pid), version
    finally:
        release_conn(conn)


def get_latest_plan(clip_id: str):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, video_id, clip_id, version, plan
                FROM edit_plans WHERE clip_id = %s
                ORDER BY version DESC LIMIT 1
            """, (clip_id,))
            row = cur.fetchone()
            if not row:
                return None
            cols = [d[0] for d in cur.description]
            d = dict(zip(cols, row))
            d["id"] = str(d["id"])
            return d
    finally:
        release_conn(conn)
