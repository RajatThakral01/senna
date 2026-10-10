"""campaign/service.py — campaigns and their asset library (CAMPAIGN_PIPELINE_PLAN.md C1).

A campaign = brief (requirements as given) + an asset library (logos, audio,
fonts, intro/outro videos, images) + an approved edit recipe + platforms.
Assets are copied into assets/campaigns/<slug>/<kind>/<name> so a campaign is
self-contained and the brief parser can only ever reference files that exist.
"""
import os
import re
import shutil

from logger import get_logger
from db.repositories import campaign_repo
from pipeline import media

log = get_logger("campaign.service")

ASSET_ROOT = os.path.join("assets", "campaigns")
ASSET_KINDS = ("logo", "audio", "font", "video", "image")


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", str(name or "").lower()).strip("-")
    return s[:48] or "campaign"


def _unique_slug(name: str) -> str:
    base = slugify(name)
    slug, n = base, 2
    while campaign_repo.get_campaign_by_slug(slug):
        slug = f"{base}-{n}"
        n += 1
    return slug


def create_campaign(name: str, brief: str = "", platforms=None) -> dict:
    if not str(name or "").strip():
        raise ValueError("campaign name is required")
    c = campaign_repo.insert_campaign(name.strip(), _unique_slug(name), brief, platforms)
    os.makedirs(campaign_dir(c), exist_ok=True)
    return c


def get_campaign(campaign_id: str) -> dict:
    c = campaign_repo.get_campaign(campaign_id)
    if not c:
        raise KeyError(f"unknown campaign {campaign_id}")
    return c


def list_campaigns() -> list:
    return campaign_repo.list_campaigns()


def update_campaign(campaign_id: str, **fields) -> dict:
    return campaign_repo.update_campaign(campaign_id, **fields)


def campaign_dir(campaign: dict) -> str:
    return os.path.join(ASSET_ROOT, campaign["slug"])


def infer_asset_kind(path: str, probe: dict = None) -> str:
    """logo | audio | font | video | image from extension, probe and file name."""
    kind = (probe or {}).get("kind") or media.kind_from_ext(path)
    if kind == "image":
        name = os.path.basename(path).lower()
        if "logo" in name or (probe or {}).get("alpha"):
            return "logo"
        return "image"
    if kind in ("audio", "font", "video"):
        return kind
    raise ValueError(f"unsupported asset type: {os.path.basename(path)}")


def add_asset(campaign_id: str, src_path: str, kind: str = None, name: str = None) -> dict:
    """Copy a file into the campaign's library, probe it, register it.

    Re-adding the same kind+name replaces the previous file.
    """
    if not os.path.isfile(src_path):
        raise FileNotFoundError(src_path)
    campaign = get_campaign(campaign_id)
    info = media.probe(src_path)
    kind = kind or infer_asset_kind(src_path, info)
    if kind not in ASSET_KINDS:
        raise ValueError(f"asset kind must be one of {ASSET_KINDS}, got {kind!r}")
    name = name or os.path.basename(src_path)
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip() or f"asset{os.path.splitext(src_path)[1]}"
    dest_dir = os.path.join(campaign_dir(campaign), kind)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, name)
    if os.path.abspath(src_path) != os.path.abspath(dest):
        shutil.copy2(src_path, dest)
    meta = {k: info[k] for k in ("duration", "width", "height", "has_audio", "has_video",
                                 "alpha", "size_bytes", "video_codec", "audio_codec")}
    asset = campaign_repo.upsert_asset(campaign_id, kind, name, dest, meta)
    log.info("asset added campaign=%s kind=%s name=%s", campaign["slug"], kind, name)
    return asset


def list_assets(campaign_id: str, kind: str = None) -> list:
    return campaign_repo.list_assets(campaign_id, kind)


def remove_asset(asset_id: str) -> None:
    a = campaign_repo.get_asset(asset_id)
    if not a:
        return
    try:
        if os.path.isfile(a["path"]):
            os.remove(a["path"])
    except OSError:
        log.warning("could not delete asset file %s", a["path"])
    campaign_repo.delete_asset(asset_id)


def asset_catalog(campaign_id: str) -> list:
    """Compact asset list for the brief parser / UI: [{name, kind, path, duration?, size?}]."""
    out = []
    for a in list_assets(campaign_id):
        m = a.get("meta") or {}
        item = {"name": a["name"], "kind": a["kind"], "path": a["path"]}
        if m.get("duration"):
            item["duration"] = round(float(m["duration"]), 1)
        if m.get("width"):
            item["size"] = f'{m["width"]}x{m["height"]}'
        out.append(item)
    return out


# ── recipe ───────────────────────────────────────────────────────────────────

def parse_brief(campaign_id: str, brief: str = None, use_llm: bool = True) -> dict:
    """Parse the campaign's brief (or a new one) against its assets. Nothing is saved."""
    from campaign.brief_parser import parse_brief as _parse
    c = get_campaign(campaign_id)
    text = c.get("brief") if brief is None else brief
    return _parse(text or "", asset_catalog(campaign_id), use_llm=use_llm)


def save_recipe(campaign_id: str, recipe: dict, requirements: dict = None) -> dict:
    """Validate against the asset library and store a new recipe version.
    Returns {campaign, validation}; status becomes ready only when it validates."""
    from campaign.recipe import validate_recipe
    v = validate_recipe(recipe or {}, asset_catalog(campaign_id))
    campaign_repo.save_recipe(campaign_id, v["recipe"], requirements)
    c = campaign_repo.update_campaign(campaign_id, status="ready" if v["ok"] else "draft")
    return {"campaign": c, "validation": v}


def ready_recipe(campaign_id: str) -> tuple:
    """(recipe, version) for a job — re-validated against today's assets.
    Raises ValueError listing the problems when the recipe can't run."""
    from campaign.recipe import validate_recipe
    c = get_campaign(campaign_id)
    if not c.get("recipe"):
        raise ValueError(f"campaign {c['name']!r} has no recipe yet — parse the brief and save it")
    v = validate_recipe(c["recipe"], asset_catalog(campaign_id))
    if not v["ok"]:
        raise ValueError("campaign recipe is not ready: " + "; ".join(v["errors"]))
    return v["recipe"], c["recipe_version"]
