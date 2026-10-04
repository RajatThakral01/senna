"""
db/repositories/chunk_repo.py

CRUD operations for the `chunks` table, including pgvector embedding storage
and cosine-similarity nearest-neighbour search.

Input:  Chunk metadata from pipeline/chunker.py; embedding vectors from pipeline/embedder.py.
Output: Chunk UUID strings; chunk row dicts; similarity-ranked chunk lists.
"""
from db.connection import get_conn, release_conn


def insert_chunk(video_id: str, chunk_index: int, start_time: float, end_time: float,
                 text: str, token_count: int, is_overlap_tail: bool,
                 silence_gap_before: float) -> str:
    """Insert a chunk row and return its UUID string."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO chunks
                  (video_id, chunk_index, start_time, end_time, text,
                   token_count, is_overlap_tail, silence_gap_before)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (video_id, chunk_index, start_time, end_time, text,
                  token_count, is_overlap_tail, silence_gap_before))
            chunk_id = cur.fetchone()[0]
            conn.commit()
            return str(chunk_id)
    finally:
        release_conn(conn)


def update_embedding(chunk_id: str, embedding: list) -> None:
    """Write the embedding vector for a chunk (pgvector column)."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chunks SET embedding = %s WHERE id = %s",
                (embedding, chunk_id)
            )
            conn.commit()
    finally:
        release_conn(conn)


def get_chunks_for_video(video_id: str) -> list:
    """Return all chunks for a video ordered by chunk_index."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, chunk_index, start_time, end_time, text,
                       token_count, is_overlap_tail, silence_gap_before
                FROM chunks
                WHERE video_id = %s
                ORDER BY chunk_index ASC
            """, (video_id,))
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        release_conn(conn)


def find_similar_chunks(query_embedding: list, video_id: str,
                        top_k: int = 5, threshold: float = 0.80) -> list:
    """
    pgvector ANN search: find top_k chunks from the same video whose cosine
    similarity to query_embedding is above threshold.
    Excludes chunks with NULL embeddings.

    Returns a list of dicts with keys:
        id, chunk_index, start_time, end_time, text, similarity (float 0-1)
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, chunk_index, start_time, end_time, text,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM chunks
                WHERE video_id = %s
                  AND embedding IS NOT NULL
                  AND 1 - (embedding <=> %s::vector) >= %s
                ORDER BY embedding <=> %s::vector ASC
                LIMIT %s
            """, (query_embedding, video_id,
                  query_embedding, threshold,
                  query_embedding, top_k))
            cols = [d[0] for d in cur.description]
            rows = []
            for row in cur.fetchall():
                d = dict(zip(cols, row))
                # Ensure id is a plain string
                d["id"] = str(d["id"])
                rows.append(d)
            return rows
    finally:
        release_conn(conn)


def delete_chunks_for_video(video_id: str) -> None:
    """Delete all chunks for a video (used when re-processing a video from scratch)."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE video_id = %s", (video_id,))
            conn.commit()
    finally:
        release_conn(conn)
