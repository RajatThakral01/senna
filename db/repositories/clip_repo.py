"""
db/repositories/clip_repo.py

CRUD operations for the `clips` table.

Input:  Clip metadata produced by pipeline/analyzer.py (timestamps, hook, reason, etc.),
        embedding vectors, and rendered output paths.
Output: Clip UUID strings; clip row dicts for report generation.
"""
from db.connection import get_conn, release_conn
import logging

log = logging.getLogger("db.clip_repo")


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
    log.debug("insert_clip video=%.8s no=%d %.1f-%.1f", video_id, clip_number, start_time, end_time)

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
            log.debug("insert_clip done id=%.8s no=%d", str(clip_id), clip_number)
            return str(clip_id)
    except Exception:
        log.exception("insert_clip failed no=%d", clip_number)
        raise
    finally:
        release_conn(conn)


def update_output_path(clip_id: str, output_path: str) -> None:
    """Store the rendered .mp4 path once FFmpeg finishes a clip."""
    log.debug("update_output clip=%.8s path=%.80s", str(clip_id), output_path)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE clips SET output_path = %s WHERE id = %s",
                (output_path, clip_id)
            )
            conn.commit()
    except Exception:
        log.exception("update_output failed clip=%.8s", str(clip_id))
        raise
    finally:
        release_conn(conn)


def get_clips_for_video(video_id: str) -> list:
    """Return all clips for a video ordered by clip_number."""
    log.debug("get_clips video=%.8s", video_id)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                    SELECT id, video_id, clip_number, start_time, end_time, duration_seconds,
                           hook, reason, suggested_title, suggested_hashtags,
                           source_chunk_ids, output_path, created_at,
                           embedding::text as embedding,
                           refine_status, refine_reason,
                           source_ranges, timeline, layout, layout_reason,
                           provenance
                    FROM clips
                    WHERE video_id = %s
                    ORDER BY clip_number ASC
                """, (video_id,))
            except Exception as e:
                if "does not exist" in str(e):
                    conn.rollback()
                    log.warning("v4 clip columns missing, reading legacy columns")
                    cur.execute("""
                        SELECT id, video_id, clip_number, start_time, end_time, duration_seconds,
                               hook, reason, suggested_title, suggested_hashtags,
                               source_chunk_ids, output_path, created_at,
                               embedding::text as embedding
                        FROM clips
                        WHERE video_id = %s
                        ORDER BY clip_number ASC
                    """, (video_id,))
                else:
                    raise
            cols = [d[0] for d in cur.description]
            rows = []
            import json
            for row in cur.fetchall():
                d = dict(zip(cols, row))
                d["id"] = str(d["id"])
                d["video_id"] = str(d["video_id"])
                if d.get("embedding"):
                    try:
                        d["embedding"] = json.loads(d["embedding"])
                    except Exception:
                        log.warning("clip embedding parse failed clip_no=%s", d.get("clip_number"))
                        d["embedding"] = None
                rows.append(d)
            log.debug("get_clips video=%.8s count=%d", video_id, len(rows))
            return rows
    except Exception:
        log.exception("get_clips failed video=%.8s", video_id)
        raise
    finally:
        release_conn(conn)


def get_confirmed_segments_for_video(video_id: str) -> dict:
    """Return {clip_id: [chunk dicts]} for LLM-confirmed (stitch) continuations.

    Chunk dicts carry id/start_time/end_time/text — exactly what
    pipeline.clipper.get_time_ranges_for_clip() reads from
    clip["related_segments"].
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT r.clip_id, ch.id, ch.start_time, ch.end_time, ch.text
                FROM related_segments r
                JOIN clips c  ON c.id = r.clip_id
                JOIN chunks ch ON ch.id = r.related_chunk_id
                WHERE c.video_id = %s AND r.confirmed_by_llm = TRUE
                ORDER BY ch.start_time ASC
            """, (video_id,))
            out = {}
            for clip_id, chunk_id, start, end, text in cur.fetchall():
                out.setdefault(str(clip_id), []).append({
                    "id": str(chunk_id),
                    "start_time": start,
                    "end_time": end,
                    "text": text,
                })
            log.debug("confirmed segments video=%.8s clips=%d", video_id, len(out))
            return out
    except Exception:
        log.exception("confirmed segments failed video=%.8s", video_id)
        raise
    finally:
        release_conn(conn)


def mark_segments_stitched(clip_id: str) -> None:
    """Flag confirmed segments of a clip as consumed by the FFmpeg stitch."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE related_segments
                SET stitched_into_clip = TRUE
                WHERE clip_id = %s AND confirmed_by_llm = TRUE
            """, (clip_id,))
            conn.commit()
    except Exception:
        log.exception("mark stitched failed clip=%.8s", str(clip_id))
        raise
    finally:
        release_conn(conn)


def update_refine_status(clip_id: str, status: str, reason: str = "") -> None:
    """Set refine_status/reason only (e.g. post-refine duplicate rejection)."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute("UPDATE clips SET refine_status=%s, refine_reason=%s WHERE id=%s",
                            (status, reason[:500] if reason else "", clip_id))
            except Exception as e:
                if "does not exist" in str(e):
                    conn.rollback()
                    log.warning("refine_status column missing, skip")
                    return
                raise
            conn.commit()
    finally:
        release_conn(conn)


def update_refinement(clip_id: str, start_time: float, end_time: float,
                      refine_status: str = "refined", refine_reason: str = "",
                      source_ranges=None, timeline=None,
                      layout: str = None, layout_reason: str = "") -> None:
    """Persist refined boundaries + timeline + layout.

    Uses v4 columns when the migration has been applied; falls back to
    start/end/duration update so render+report stay correct on old schemas.
    """
    import json as _json
    duration = (float(end_time) - float(start_time)) if end_time and start_time else 0
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                    UPDATE clips SET start_time=%s, end_time=%s, duration_seconds=%s,
                        refine_status=%s, refine_reason=%s,
                        source_ranges=%s::jsonb, timeline=%s::jsonb,
                        layout=%s, layout_reason=%s
                    WHERE id=%s
                """, (float(start_time), float(end_time), float(duration),
                      refine_status, refine_reason[:500] if refine_reason else "",
                      _json.dumps(source_ranges or []),
                      _json.dumps(timeline or []),
                      layout, (layout_reason or "")[:500], clip_id))
            except Exception as e:
                # UndefinedColumn on pre-migration DBs -> minimal update
                if "UndefinedColumn" in type(e).__name__ or "does not exist" in str(e):
                    conn.rollback()
                    log.warning("refine columns missing, falling back to times-only update")
                    cur.execute("""
                        UPDATE clips SET start_time=%s, end_time=%s, duration_seconds=%s
                        WHERE id=%s
                    """, (float(start_time), float(end_time), float(duration), clip_id))
                else:
                    raise
            conn.commit()
    except Exception:
        log.exception("update_refinement failed clip=%.8s", str(clip_id))
        raise
    finally:
        release_conn(conn)
