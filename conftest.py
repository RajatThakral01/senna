# conftest.py
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


def cleanup_video(vid):
    """Delete a throwaway video row + all derived rows (per-table, resilient).

    Each DELETE is independent so a missing table/row never aborts the rest
    (prevents orphan rows like the ones that once polluted the real DB).
    """
    from db.connection import get_conn, release_conn
    for table in ("edit_plans", "candidates", "outlines", "audio_events",
                  "pipeline_runs", "related_segments", "clips"):
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                if table == "related_segments":
                    cur.execute(
                        "DELETE FROM related_segments WHERE clip_id IN "
                        "(SELECT id FROM clips WHERE video_id = %s)", (vid,))
                else:
                    cur.execute(f"DELETE FROM {table} WHERE video_id = %s", (vid,))
                conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
        finally:
            release_conn(conn)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM clips WHERE video_id = %s", (vid,))
            cur.execute("DELETE FROM videos WHERE id = %s", (vid,))
            conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
    finally:
        release_conn(conn)