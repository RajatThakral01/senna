"""
db/repositories/campaign_repo.py

CRUD for `campaigns` and `campaign_assets` (migration v7).
"""
import json
import logging

from db.connection import get_conn, release_conn

log = logging.getLogger("db.campaign_repo")

_CAMPAIGN_COLS = ("id, name, slug, brief, requirements, recipe, recipe_version, platforms, "
                  "status, created_at, updated_at")
_ASSET_COLS = "id, campaign_id, kind, name, path, meta, created_at"


def _rows(cur):
    cols = [d[0] for d in cur.description]
    out = []
    for row in cur.fetchall():
        d = dict(zip(cols, row))
        for k in ("id", "campaign_id"):
            if d.get(k) is not None:
                d[k] = str(d[k])
        out.append(d)
    return out


def _one(sql, params):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = _rows(cur) if cur.description else []
            conn.commit()
            return rows[0] if rows else None
    finally:
        release_conn(conn)


def _many(sql, params=()):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return _rows(cur)
    finally:
        release_conn(conn)


# ── campaigns ────────────────────────────────────────────────────────────────

def insert_campaign(name: str, slug: str, brief: str = "", platforms=None) -> dict:
    row = _one(f"""
        INSERT INTO campaigns (name, slug, brief, platforms)
        VALUES (%s, %s, %s, %s::jsonb)
        RETURNING {_CAMPAIGN_COLS}
    """, (name, slug, brief or "", json.dumps(list(platforms or []))))
    log.info("campaign created id=%.8s slug=%s", row["id"], slug)
    return row


def get_campaign(campaign_id: str) -> dict | None:
    return _one(f"SELECT {_CAMPAIGN_COLS} FROM campaigns WHERE id = %s", (campaign_id,))


def get_campaign_by_slug(slug: str) -> dict | None:
    return _one(f"SELECT {_CAMPAIGN_COLS} FROM campaigns WHERE slug = %s", (slug,))


def list_campaigns(include_archived: bool = False) -> list:
    where = "" if include_archived else "WHERE status <> 'archived'"
    return _many(f"SELECT {_CAMPAIGN_COLS} FROM campaigns {where} ORDER BY updated_at DESC")


def update_campaign(campaign_id: str, **fields) -> dict | None:
    """Update name / brief / platforms / status / requirements (only given fields)."""
    allowed = {"name": "%s", "brief": "%s", "status": "%s",
               "platforms": "%s::jsonb", "requirements": "%s::jsonb"}
    sets, params = [], []
    for k, v in fields.items():
        if k not in allowed:
            raise ValueError(f"cannot update campaign field {k!r}")
        sets.append(f"{k} = {allowed[k]}")
        params.append(json.dumps(v) if allowed[k].endswith("jsonb") else v)
    if not sets:
        return get_campaign(campaign_id)
    params.append(campaign_id)
    return _one(f"""
        UPDATE campaigns SET {', '.join(sets)}, updated_at = NOW()
        WHERE id = %s RETURNING {_CAMPAIGN_COLS}
    """, tuple(params))


def save_recipe(campaign_id: str, recipe: dict, requirements: dict = None) -> dict:
    """Store an approved recipe; bumps recipe_version. Returns the campaign row."""
    if requirements is None:
        sql = f"""
            UPDATE campaigns SET recipe = %s::jsonb, recipe_version = recipe_version + 1,
                   updated_at = NOW()
            WHERE id = %s RETURNING {_CAMPAIGN_COLS}"""
        params = (json.dumps(recipe), campaign_id)
    else:
        sql = f"""
            UPDATE campaigns SET recipe = %s::jsonb, requirements = %s::jsonb,
                   recipe_version = recipe_version + 1, updated_at = NOW()
            WHERE id = %s RETURNING {_CAMPAIGN_COLS}"""
        params = (json.dumps(recipe), json.dumps(requirements), campaign_id)
    row = _one(sql, params)
    log.info("campaign recipe saved id=%.8s version=%s", campaign_id, row and row["recipe_version"])
    return row


def delete_campaign(campaign_id: str) -> None:
    _one("DELETE FROM campaigns WHERE id = %s", (campaign_id,))


# ── assets ───────────────────────────────────────────────────────────────────

def upsert_asset(campaign_id: str, kind: str, name: str, path: str, meta: dict) -> dict:
    return _one(f"""
        INSERT INTO campaign_assets (campaign_id, kind, name, path, meta)
        VALUES (%s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (campaign_id, kind, name)
        DO UPDATE SET path = EXCLUDED.path, meta = EXCLUDED.meta
        RETURNING {_ASSET_COLS}
    """, (campaign_id, kind, name, path, json.dumps(meta or {})))


def list_assets(campaign_id: str, kind: str = None) -> list:
    if kind:
        return _many(f"""SELECT {_ASSET_COLS} FROM campaign_assets
                         WHERE campaign_id = %s AND kind = %s ORDER BY kind, name""",
                     (campaign_id, kind))
    return _many(f"""SELECT {_ASSET_COLS} FROM campaign_assets
                     WHERE campaign_id = %s ORDER BY kind, name""", (campaign_id,))


def get_asset(asset_id: str) -> dict | None:
    return _one(f"SELECT {_ASSET_COLS} FROM campaign_assets WHERE id = %s", (asset_id,))


def delete_asset(asset_id: str) -> None:
    _one("DELETE FROM campaign_assets WHERE id = %s", (asset_id,))
