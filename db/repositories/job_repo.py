"""
db/repositories/job_repo.py

CRUD for `jobs`, `job_inputs` and `deliverables` (migration v7).
"""
import json
import logging

from db.connection import get_conn, release_conn

log = logging.getLogger("db.job_repo")

_JOB_COLS = ("id, campaign_id, status, mode_policy, recipe_version, recipe, progress, message, "
             "error, created_at, started_at, finished_at")
_INPUT_COLS = "id, job_id, idx, source, local_path, probe, mode, video_id, status, error"
_DELIV_COLS = ("id, job_id, job_input_id, clip_id, variant, platform, path, recipe_version, qa, "
               "status, created_at, updated_at")
_UUIDS = ("id", "campaign_id", "job_id", "job_input_id", "clip_id", "video_id")


def _rows(cur):
    cols = [d[0] for d in cur.description]
    out = []
    for row in cur.fetchall():
        d = dict(zip(cols, row))
        for k in _UUIDS:
            if d.get(k) is not None:
                d[k] = str(d[k])
        out.append(d)
    return out


def _exec(sql, params=(), fetch="one"):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = _rows(cur) if cur.description else []
            conn.commit()
            if fetch == "one":
                return rows[0] if rows else None
            return rows
    finally:
        release_conn(conn)


def _update(table, cols, row_id, fields, allowed):
    sets, params = [], []
    for k, v in fields.items():
        if k not in allowed:
            raise ValueError(f"cannot update {table}.{k}")
        cast = allowed[k]
        if cast == "jsonb":
            sets.append(f"{k} = %s::jsonb")
            params.append(json.dumps(v))
        elif cast == "now":
            sets.append(f"{k} = NOW()")
        else:
            sets.append(f"{k} = %s")
            params.append(v)
    if not sets:
        return None
    params.append(row_id)
    return _exec(f"UPDATE {table} SET {', '.join(sets)} WHERE id = %s RETURNING {cols}",
                 tuple(params))


# ── jobs ─────────────────────────────────────────────────────────────────────

def insert_job(campaign_id: str, recipe: dict, recipe_version: int,
               mode_policy: str = "auto") -> dict:
    row = _exec(f"""
        INSERT INTO jobs (campaign_id, recipe, recipe_version, mode_policy)
        VALUES (%s, %s::jsonb, %s, %s) RETURNING {_JOB_COLS}
    """, (campaign_id, json.dumps(recipe or {}), recipe_version, mode_policy))
    log.info("job created id=%.8s campaign=%.8s", row["id"], campaign_id)
    return row


def get_job(job_id: str) -> dict | None:
    return _exec(f"SELECT {_JOB_COLS} FROM jobs WHERE id = %s", (job_id,))


def list_jobs(campaign_id: str = None, status: str = None, limit: int = 50) -> list:
    where, params = [], []
    if campaign_id:
        where.append("campaign_id = %s"); params.append(campaign_id)
    if status:
        where.append("status = %s"); params.append(status)
    sql = f"SELECT {_JOB_COLS} FROM jobs"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_at DESC LIMIT %s"
    params.append(limit)
    return _exec(sql, tuple(params), fetch="all")


def update_job(job_id: str, **fields) -> dict | None:
    """status / progress / message / error, started_at=True / finished_at=True → NOW()."""
    allowed = {"status": "", "progress": "", "message": "", "error": "",
               "started_at": "now", "finished_at": "now"}
    fields = {k: v for k, v in fields.items() if not (allowed.get(k) == "now" and not v)}
    return _update("jobs", _JOB_COLS, job_id, fields, allowed)


# ── job inputs ───────────────────────────────────────────────────────────────

def add_input(job_id: str, idx: int, source: str, mode: str = None) -> dict:
    return _exec(f"""
        INSERT INTO job_inputs (job_id, idx, source, mode) VALUES (%s, %s, %s, %s)
        RETURNING {_INPUT_COLS}
    """, (job_id, idx, source, mode))


def list_inputs(job_id: str) -> list:
    return _exec(f"SELECT {_INPUT_COLS} FROM job_inputs WHERE job_id = %s ORDER BY idx",
                 (job_id,), fetch="all")


def update_input(input_id: str, **fields) -> dict | None:
    allowed = {"local_path": "", "probe": "jsonb", "mode": "", "video_id": "",
               "status": "", "error": ""}
    return _update("job_inputs", _INPUT_COLS, input_id, fields, allowed)


# ── deliverables ─────────────────────────────────────────────────────────────

def add_deliverable(job_id: str, job_input_id: str = None, clip_id: str = None,
                    variant: str = "main", platform: str = "generic", path: str = None,
                    recipe_version: int = None) -> dict:
    return _exec(f"""
        INSERT INTO deliverables (job_id, job_input_id, clip_id, variant, platform, path,
                                  recipe_version)
        VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING {_DELIV_COLS}
    """, (job_id, job_input_id, clip_id, variant, platform, path, recipe_version))


def list_deliverables(job_id: str) -> list:
    return _exec(f"SELECT {_DELIV_COLS} FROM deliverables WHERE job_id = %s ORDER BY created_at",
                 (job_id,), fetch="all")


def update_deliverable(deliverable_id: str, **fields) -> dict | None:
    allowed = {"path": "", "qa": "jsonb", "status": "", "updated_at": "now"}
    fields = dict(fields, updated_at=True)
    return _update("deliverables", _DELIV_COLS, deliverable_id, fields, allowed)


def get_deliverable(deliverable_id: str) -> dict | None:
    return _exec(f"SELECT {_DELIV_COLS} FROM deliverables WHERE id = %s", (deliverable_id,))


def delete_deliverables(job_input_id: str) -> int:
    """Drop an input's deliverable rows before it is re-rendered (files are overwritten)."""
    rows = _exec("DELETE FROM deliverables WHERE job_input_id = %s RETURNING id",
                 (job_input_id,), fetch="all")
    return len(rows)
