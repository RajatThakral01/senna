"""
db/repositories/candidate_repo.py

CRUD for the `candidates` table (the fused discovery pool; clips = selected set).
Statuses: proposed | shortlisted | selected | rejected (rejected/pending/failed
kept distinct from successful rows at all times).
"""
from db.connection import get_conn, release_conn
import logging

log = logging.getLogger("db.candidate_repo")


def clear_candidates(video_id: str, source: str = None) -> int:
    """Delete candidates for a video, optionally limited to one source."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            if source:
                cur.execute("DELETE FROM candidates WHERE video_id = %s AND source = %s",
                            (video_id, source))
            else:
                cur.execute("DELETE FROM candidates WHERE video_id = %s", (video_id,))
            conn.commit()
            return cur.rowcount
    finally:
        release_conn(conn)


def insert_candidate(video_id: str, source: str, sentence_ids,
                     source_ranges, hook=None, main_idea=None, payoff=None,
                     required_context=None, rationale=None, uncertainty=None,
                     scores=None, status="proposed", status_reason="") -> str:
    import json as _json
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO candidates
                  (video_id, source, sentence_ids, source_ranges, hook,
                   main_idea, payoff, required_context, rationale,
                   uncertainty, scores, status, status_reason)
                VALUES (%s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                RETURNING id
            """, (video_id, source, _json.dumps(sentence_ids or []),
                  _json.dumps(source_ranges or []), hook, main_idea, payoff,
                  required_context, rationale, uncertainty,
                  _json.dumps(scores or {}), status, status_reason or ""))
            cid = cur.fetchone()[0]
            conn.commit()
            return str(cid)
    finally:
        release_conn(conn)


def get_candidates(video_id: str, status: str = None) -> list:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            if status:
                cur.execute("""
                    SELECT id, video_id, source, sentence_ids, source_ranges, hook,
                           main_idea, payoff, required_context, rationale,
                           uncertainty, scores, status, status_reason, clip_id
                    FROM candidates WHERE video_id = %s AND status = %s
                    ORDER BY created_at
                """, (video_id, status))
            else:
                cur.execute("""
                    SELECT id, video_id, source, sentence_ids, source_ranges, hook,
                           main_idea, payoff, required_context, rationale,
                           uncertainty, scores, status, status_reason, clip_id
                    FROM candidates WHERE video_id = %s ORDER BY created_at
                """, (video_id,))
            cols = [d[0] for d in cur.description]
            rows = []
            for row in cur.fetchall():
                d = dict(zip(cols, row))
                d["id"] = str(d["id"])
                if d.get("clip_id"):
                    d["clip_id"] = str(d["clip_id"])
                rows.append(d)
            return rows
    finally:
        release_conn(conn)


def update_candidate_status(candidate_id: str, status: str, reason: str = "",
                            clip_id: str = None, scores=None) -> None:
    import json as _json
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            if scores is not None:
                cur.execute("""
                    UPDATE candidates SET status=%s, status_reason=%s,
                        clip_id=COALESCE(%s, clip_id), scores=%s::jsonb
                    WHERE id=%s
                """, (status, reason or "", clip_id, _json.dumps(scores), candidate_id))
            else:
                cur.execute("""
                    UPDATE candidates SET status=%s, status_reason=%s,
                        clip_id=COALESCE(%s, clip_id)
                    WHERE id=%s
                """, (status, reason or "", clip_id, candidate_id))
            conn.commit()
    finally:
        release_conn(conn)
