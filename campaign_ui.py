"""campaign_ui.py — campaign-first Gradio tabs (CAMPAIGN_PIPELINE_PLAN C7).

build_campaign_tabs() adds, inside an open gr.Blocks:
  1. Campaigns        create / pick a campaign, brief, platforms, asset library,
                      "Parse brief" → plain-English recipe + questions, JSON editor, save
  2. New job          pick campaign, paste links / paths / folders or upload files,
                      check inputs (duration, aspect, auto mode — overridable), run
  3. Jobs             live queue (auto-refresh), inputs per job, cancel / resume
  4. Review & deliver deliverables with player, compliance checklist, posting text,
                      approve / reject, re-render one file with a tweaked recipe, zip

The handlers are plain functions (tested without a browser); the Gradio wiring
is at the bottom. All work goes through campaign/service.py, campaign/jobs.py
and campaign/delivery.py, so a web frontend can replace this later.
"""
import json
import os

import gradio as gr

from logger import get_logger

log = get_logger("ui.campaigns")

KINDS = ["auto", "logo", "audio", "font", "video", "image"]
PLATFORM_CHOICES = ["tiktok", "reels", "shorts", "youtube", "x", "facebook", "generic"]
MODE_CHOICES = ["auto", "clip", "edit"]


def _err(e):
    log.exception("campaign UI action failed")
    return f"⚠️ {e}"


# ── campaigns ────────────────────────────────────────────────────────────────

def campaign_choices():
    from campaign import service
    try:
        return [(f"{c['name']}  ·  {c['status']}", c["id"]) for c in service.list_campaigns()]
    except Exception:
        log.exception("listing campaigns failed")
        return []


def job_choices(campaign_id=None):
    from db.repositories import job_repo
    from campaign import service
    names = {c["id"]: c["name"] for c in service.list_campaigns()}
    return [(f"{j['id'][:8]}  ·  {names.get(j['campaign_id'], '?')}  ·  {j['status']}  ·  "
             f"{j['created_at']:%d %b %H:%M}", j["id"])
            for j in job_repo.list_jobs(campaign_id=campaign_id or None, limit=100)]


def assets_table(campaign_id):
    from campaign import service
    if not campaign_id:
        return []
    return [[a["kind"], a["name"], a.get("duration") or "", a.get("size") or ""]
            for a in service.asset_catalog(campaign_id)]


def _recipe_view(validation, extra_questions=None):
    """(summary markdown, questions markdown) for a validation result."""
    from campaign.recipe import describe_recipe
    lines = [f"- {x}" for x in describe_recipe(validation["recipe"])]
    for w in validation.get("warnings") or []:
        lines.append(f"- ⚠️ {w}")
    for e in validation.get("errors") or []:
        lines.append(f"- ❌ {e}")
    qs = extra_questions if extra_questions is not None else validation.get("questions") or []
    qmd = "\n".join(
        f"- **{q['question']}**" + (f" — default: `{q['default']}`" if q.get("default") is not None
                                    else "")
        + (f" — options: {', '.join(map(str, q['options']))}" if q.get("options") else "")
        for q in qs) or "_No open questions._"
    status = "✅ **Ready to run**" if validation.get("ok") else \
        "📝 **Draft** — answer the questions (edit the brief or the JSON) and save again"
    return status + "\n\n" + "\n".join(lines), qmd


def load_campaign(campaign_id):
    """→ name, brief, platforms, assets, recipe json, summary, questions, status."""
    from campaign import service
    from campaign.recipe import validate_recipe
    if not campaign_id:
        return "", "", [], [], "{}", "", "", "Pick or create a campaign."
    c = service.get_campaign(campaign_id)
    rec = c.get("recipe") or {}
    if rec:
        v = validate_recipe(rec, service.asset_catalog(campaign_id))
        summary, qmd = _recipe_view(v)
    else:
        summary, qmd = "_No recipe yet — write the brief and press **Parse brief**._", ""
    return (c["name"], c.get("brief") or "", c.get("platforms") or [], assets_table(campaign_id),
            json.dumps(rec, indent=2), summary, qmd,
            f"Loaded **{c['name']}** (`{c['slug']}`), recipe v{c['recipe_version']}.")


def create_campaign(name, brief, platforms):
    from campaign import service
    try:
        c = service.create_campaign(name, brief or "", platforms or None)
        return gr.update(choices=campaign_choices(), value=c["id"]), \
            f"Created **{c['name']}**. Add assets, then parse the brief."
    except Exception as e:
        return gr.update(), _err(e)


def save_details(campaign_id, name, brief, platforms):
    from campaign import service
    if not campaign_id:
        return "Pick a campaign first."
    try:
        service.update_campaign(campaign_id, name=name, brief=brief, platforms=platforms or [])
        return "Saved name, brief and platforms."
    except Exception as e:
        return _err(e)


def add_assets(campaign_id, files, kind):
    from campaign import service
    if not campaign_id:
        return [], "Pick a campaign first."
    added, errors = [], []
    for f in files or []:
        path = f if isinstance(f, str) else getattr(f, "name", None)
        try:
            a = service.add_asset(campaign_id, path, kind=None if kind == "auto" else kind,
                                  name=os.path.basename(path))
            added.append(f"{a['kind']}: {a['name']}")
        except Exception as e:
            errors.append(f"{os.path.basename(str(path))}: {e}")
    msg = ("Added " + ", ".join(added) if added else "Nothing added") + \
        ("" if not errors else " — ⚠️ " + "; ".join(errors))
    return assets_table(campaign_id), msg


def remove_asset(campaign_id, name):
    from campaign import service
    if not campaign_id or not name:
        return assets_table(campaign_id), "Type the asset name to remove."
    hit = [a for a in service.list_assets(campaign_id) if a["name"] == name.strip()]
    if not hit:
        return assets_table(campaign_id), f"No asset named {name!r}."
    service.remove_asset(hit[0]["id"])
    return assets_table(campaign_id), f"Removed {name}."


def parse_brief(campaign_id, brief, use_llm):
    """→ recipe json, summary, questions, status (nothing saved yet)."""
    from campaign import service
    if not campaign_id:
        return "{}", "", "", "Pick a campaign first."
    try:
        r = service.parse_brief(campaign_id, brief, use_llm=bool(use_llm))
        summary, qmd = _recipe_view(r, r.get("questions"))
        return (json.dumps(r["recipe"], indent=2), summary, qmd,
                f"Parsed with **{r['method']}**. Review, then **Save recipe**.")
    except Exception as e:
        return gr.update(), gr.update(), gr.update(), _err(e)


def save_recipe(campaign_id, recipe_json, brief):
    """→ recipe json (normalised), summary, questions, status, campaign dropdown."""
    from campaign import service
    if not campaign_id:
        return gr.update(), gr.update(), gr.update(), "Pick a campaign first.", gr.update()
    try:
        raw = json.loads(recipe_json or "{}")
    except ValueError as e:
        return gr.update(), gr.update(), gr.update(), f"⚠️ The JSON is invalid: {e}", gr.update()
    try:
        out = service.save_recipe(campaign_id, raw, requirements={"brief": brief})
        v = out["validation"]
        summary, qmd = _recipe_view(v)
        return (json.dumps(v["recipe"], indent=2), summary, qmd,
                f"Saved recipe v{out['campaign']['recipe_version']} "
                f"({'ready' if v['ok'] else 'draft'}).",
                gr.update(choices=campaign_choices(), value=campaign_id))
    except Exception as e:
        return gr.update(), gr.update(), gr.update(), _err(e), gr.update()


# ── new job ──────────────────────────────────────────────────────────────────

def campaign_summary(campaign_id):
    """Markdown: what a job on this campaign will do (so the wrong campaign is obvious
    before anything renders), or why it can't run yet."""
    from campaign import service
    from campaign.recipe import validate_recipe
    if not campaign_id:
        return "_Pick the campaign this job belongs to._"
    try:
        c = service.get_campaign(campaign_id)
        rec = c.get("recipe") or {}
        if not rec:
            return f"❌ **{c['name']}** has no recipe yet — open **1 · Campaigns**, parse the brief and save."
        v = validate_recipe(rec, service.asset_catalog(campaign_id))
        summary, qmd = _recipe_view(v)
        head = f"### {c['name']} · recipe v{c['recipe_version']}\n"
        if not v.get("ok"):
            return head + summary + "\n\n**Open questions** (answer them in **1 · Campaigns**):\n" + qmd
        return head + summary
    except Exception as e:
        return _err(e)


def _sources(text, files):
    from campaign.jobs import expand_sources
    items = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    items += [f if isinstance(f, str) else getattr(f, "name", "") for f in files or []]
    return expand_sources(items)


def check_inputs(campaign_id, text, files, policy):
    """→ rows [idx, source, duration, aspect, auto mode, why, override]."""
    from campaign import jobs, service
    from pipeline import media
    rows = []
    recipe = {}
    if campaign_id:
        try:
            recipe = service.get_campaign(campaign_id).get("recipe") or {}
        except Exception:
            recipe = {}
    for i, s in enumerate(_sources(text, files)):
        if os.path.isfile(s):
            p = media.probe(s)
            if not p["has_video"]:
                rows.append([i, s, "", "", "-", f"not a video ({p['kind']})", "auto"])
                continue
            mode, why = jobs.route(p, recipe, policy)
            rows.append([i, s, f"{p['duration'] or 0:.0f}s",
                         media.aspect_label(p["width"], p["height"]), mode, why, "auto"])
        else:
            mode, why = (jobs.route({}, recipe, policy) if policy != "auto"
                         or recipe.get("mode") in jobs.MODES
                         else ("?", "link — length known after download"))
            rows.append([i, s, "?", "?", mode, why, "auto"])
    return rows, (f"{len(rows)} input(s). Change **override** to force clip/edit per input."
                  if rows else "No inputs yet.")


def _rows(table):
    if table is None:
        return []
    if hasattr(table, "values"):          # pandas DataFrame
        return table.values.tolist()
    return list(table)


def start_job(campaign_id, text, files, policy, table):
    from campaign import jobs
    if not campaign_id:
        return "Pick a campaign first.", gr.update()
    try:
        srcs = _sources(text, files)
        modes = {}
        for r in _rows(table):
            try:
                if str(r[6]).strip() in jobs.MODES:
                    modes[int(r[0])] = str(r[6]).strip()
            except (IndexError, ValueError, TypeError):
                pass
        job = jobs.create_job(campaign_id, srcs, mode_policy=policy, modes=modes)
        jobs.submit(job["id"])
        return (f"✅ Job `{job['id'][:8]}` queued with {len(srcs)} input(s). "
                "Follow it in **Jobs**."), gr.update(choices=job_choices(), value=job["id"])
    except Exception as e:
        return _err(e), gr.update()


# ── jobs ─────────────────────────────────────────────────────────────────────

def jobs_table():
    from db.repositories import job_repo
    from campaign import service
    names = {c["id"]: c["name"] for c in service.list_campaigns()}
    return [[j["id"][:8], names.get(j["campaign_id"], "?"), j["status"],
             f"{float(j['progress'] or 0) * 100:.0f}%", (j.get("message") or "")[:140],
             f"{j['created_at']:%d %b %H:%M}"] for j in job_repo.list_jobs(limit=30)]


def job_detail(job_id):
    from campaign import jobs
    if not job_id:
        return [], ""
    st = jobs.status(job_id)
    j = st["job"]
    rows = [[i["idx"], i["status"], i.get("mode") or "-", i["source"],
             (i.get("error") or "")[:200]] for i in st["inputs"]]
    md = (f"**{j['status']}** · {float(j['progress'] or 0) * 100:.0f}% · {j.get('message') or ''}"
          + (f"\n\n❌ {j['error']}" if j.get("error") else "")
          + f"\n\n{len(st['deliverables'])} file(s) rendered.")
    return rows, md


def refresh_jobs(job_id):
    rows, md = job_detail(job_id)
    return jobs_table(), rows, md


def cancel_job(job_id):
    from campaign import jobs
    if not job_id:
        return "Pick a job."
    jobs.worker().cancel(job_id)
    return "Cancel requested — the job stops after the current input."


def resume_job(job_id):
    from campaign import jobs
    if not job_id:
        return "Pick a job."
    jobs.submit(job_id)
    return "Re-queued: unfinished inputs will run again."


# ── progress (live view of a running job) ────────────────────────────────────

import html as _html
import re as _re
import time as _time
from datetime import datetime, timezone

CLIP_STEPS = [
    ("Prepare the video", ("resolving", "mode —", "normaliz", "registered", "parsing campaign",
                           "starting")),
    ("Transcribe speech", ("transcrib",)),
    ("Understand the video", ("chunking", "audio events", "embedding", "outlining")),
    ("Find the best moments (AI)", ("analyzing", "analysing", "related segments")),
    ("Polish where clips start & end", ("refining",)),
    ("Frame each clip for vertical", ("framing",)),
    ("Add text & export every platform", ("composing",)),
    ("Quality check", ("checking",)),
]
REEXPORT_STEPS = [
    ("Re-export every clip (text, captions, one file per format)", ("re-export",)),
    ("Quality check", ("checking",)),
]
EDIT_STEPS = [
    ("Prepare the video", ("resolving", "mode —", "starting")),
    ("Transcribe speech", ("transcrib",)),
    ("Edit & export every platform", ("rendering", "composing")),
    ("Quality check", ("checking",)),
]
_TYPICAL = ("Long videos usually take 30–60 minutes. Framing and exporting are the slowest "
            "steps and can stay on one line for several minutes — the activity feed below "
            "shows it is still working.")


def _parse_message(msg):
    """'[1/2] name.mp4: [8/11] Analyzing for clips...' → (1, 2, 'name.mp4', 'Analyzing for clips')."""
    m = _re.match(r"^\[(\d+)/(\d+)\]\s+(.*?):\s+(.*)$", msg or "")
    if not m:
        return None, None, None, (msg or "").strip()
    step = _re.sub(r"^\[\d+/\d+\]\s*", "", m.group(4)).rstrip(". ")
    return int(m.group(1)), int(m.group(2)), m.group(3), step


def _step_index(step_text, steps):
    t = (step_text or "").lower()
    for k, (_, keys) in enumerate(steps):
        if any(key in t for key in keys):
            return k
    return None


def _ago(secs):
    if secs is None:
        return "—"
    secs = int(secs)
    if secs < 60:
        return f"{secs} s ago"
    if secs < 3600:
        return f"{secs // 60} min ago"
    return f"{secs // 3600} h {secs % 3600 // 60} min ago"


def _duration(a, b=None):
    if not a:
        return "—"
    b = b or datetime.now(timezone.utc)
    s = max(0, int((b - a).total_seconds()))
    return f"{s // 3600} h {s % 3600 // 60} min" if s >= 3600 else \
        f"{s // 60} min {s % 60} s" if s >= 60 else f"{s} s"


_CHIP = {"running": ("#2563eb", "Running"), "queued": ("#a16207", "Waiting to start"),
         "done": ("#15803d", "Finished"), "failed": ("#b91c1c", "Failed"),
         "cancelled": ("#6b7280", "Cancelled")}


def progress_card(job_id):
    """HTML card: status, progress bar, step checklist, timing, what's happening now."""
    from campaign import activity, service
    from db.repositories import job_repo
    if not job_id:
        return ("<div style='padding:14px;opacity:.75'>Pick a job above — a running job is "
                "picked automatically.</div>")
    j = job_repo.get_job(job_id)
    if not j:
        return "<div style='padding:14px'>Job not found.</div>"
    camp = service.get_campaign(j["campaign_id"])
    inputs = job_repo.list_inputs(job_id)
    status = j["status"]
    pct = 100 if status == "done" else int(round(float(j["progress"] or 0) * 100))
    color, label = _CHIP.get(status, ("#6b7280", status))
    idx, total, name, step = _parse_message(j.get("message"))
    cur = next((i for i in inputs if i["status"] == "running"), None) or \
        (inputs[idx - 1] if idx and idx <= len(inputs) else None)
    steps = (REEXPORT_STEPS if "re-export" in (j.get("message") or "")
             else EDIT_STEPS if (cur or {}).get("mode") == "edit" else CLIP_STEPS)
    k = _step_index(step, steps) if status == "running" else None
    quiet = activity.seconds_since_activity(job_id)
    e = _html.escape
    rows = []
    for n, (title, _) in enumerate(steps):
        if status == "done" or (k is not None and n < k):
            icon, style = "✅", "opacity:.8"
        elif k is not None and n == k:
            icon, style = "⏳", "font-weight:600"
        else:
            icon, style = "○", "opacity:.45"
        extra = ""
        if k is not None and n == k:
            m = _re.search(r"(\d+)/(\d+)", step or "")
            extra = f" <span style='opacity:.75'>— {e(step)}</span>" if m or step else ""
        rows.append(f"<div style='{style};margin:3px 0'>{icon} {e(title)}{extra}</div>")
    done_inputs = sum(1 for i in inputs if i["status"] == "done")
    failed = [i for i in inputs if i["status"] == "failed"]
    files = len(job_repo.list_deliverables(job_id))
    if status == "running":
        now = (f"Video {idx} of {total}: <b>{e((name or '')[:70])}</b><br>{e(step or '')}"
               if idx else e(step or "starting"))
        hint = (f"<div style='margin-top:8px;opacity:.75'>Last activity {_ago(quiet)}. "
                + ("Still working — this step is long; nothing is wrong. "
                   if quiet is not None and quiet > 90 else "") + _TYPICAL + "</div>")
    elif status == "queued":
        now, hint = "Waiting for the worker to pick this job up.", ""
    elif status == "done":
        now = (f"{files} file(s) ready from {done_inputs} of {len(inputs)} video(s). "
               "Open <b>4 · Review &amp; download</b> to check and approve them.")
        hint = ""
    else:
        now, hint = e(j.get("message") or status), ""
    err = ""
    if failed:
        err = "<div style='margin-top:8px;color:#b91c1c'>" + "<br>".join(
            f"❌ Video {i['idx'] + 1} failed: {e((i.get('error') or '')[:240])}"
            for i in failed) + "<br>Use <b>Resume / retry failed videos</b> to try again.</div>"
    took_label = "Running for" if status in ("running", "queued") else "Last run took"
    return f"""
<div style="border:1px solid var(--border-color-primary);border-radius:12px;padding:16px;
            background:var(--block-background-fill)">
  <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap">
    <div style="font-size:1.1em"><b>{e(camp['name'])}</b>
      <span style="opacity:.6">· job {e(str(job_id)[:8])} · {len(inputs)} video(s)</span></div>
    <span style="background:{color};color:#fff;border-radius:999px;padding:3px 12px;
                 font-size:.9em">{label}</span>
  </div>
  <div style="margin:12px 0 6px;height:12px;border-radius:6px;background:var(--border-color-primary);
              overflow:hidden"><div style="width:{pct}%;height:100%;background:{color};
              transition:width .6s"></div></div>
  <div style="display:flex;justify-content:space-between;opacity:.8;font-size:.9em;flex-wrap:wrap;gap:8px">
    <span>{pct}% done</span>
    <span>{took_label} {_duration(j.get('started_at'),
                                   None if status in ('running', 'queued') else j.get('finished_at'))}</span>
  </div>
  <div style="margin-top:12px">{now}</div>
  <div style="margin-top:12px;columns:2;column-gap:24px">{''.join(rows)}</div>
  {hint}{err}
</div>"""


def activity_feed(job_id):
    from campaign import activity
    if not job_id:
        return ""
    lines = activity.tail(job_id, 16)
    if lines:
        return "\n".join(reversed(lines))
    return ("(no activity yet — it appears here as soon as the job starts. Jobs that ran "
            "before the live feed existed have no history.)")


def latest_active_job():
    from db.repositories import job_repo
    for st in ("running", "queued"):
        rows = job_repo.list_jobs(status=st, limit=1)
        if rows:
            return rows[0]["id"]
    return None


def status_banner():
    """One-line status shown above every tab (refreshed every few seconds)."""
    from campaign import activity, service
    from db.repositories import job_repo
    try:
        active = job_repo.list_jobs(status="running", limit=5) + \
            job_repo.list_jobs(status="queued", limit=5)
    except Exception:
        return ""
    if not active:
        return ("<div style='padding:8px 12px;border-radius:10px;"
                "background:var(--block-background-fill);opacity:.85'>✅ No jobs running. "
                "Set up a campaign in <b>1</b>, start a job in <b>2</b>.</div>")
    names = {c["id"]: c["name"] for c in service.list_campaigns()}
    parts = []
    for j in active:
        idx, total, _, step = _parse_message(j.get("message"))
        pct = int(round(float(j["progress"] or 0) * 100))
        where = (f"video {idx}/{total} · {step}" if idx else (step or "waiting"))
        q = activity.seconds_since_activity(j["id"])
        live = f" · active {_ago(q)}" if q is not None and j["status"] == "running" else ""
        icon = "⏳" if j["status"] == "running" else "🕓"
        parts.append(f"{icon} <b>{_html.escape(names.get(j['campaign_id'], '?'))}</b> "
                     f"({str(j['id'])[:8]}) — {pct}% — {_html.escape(where[:90])}{live}")
    return ("<div style='padding:8px 12px;border-radius:10px;border:1px solid #2563eb55;"
            "background:var(--block-background-fill)'>" + "<br>".join(parts) +
            " · <i>details in <b>3 · Progress</b></i></div>")


def refresh_progress(job_id):
    """Progress tab refresh: auto-picks the running job when none is selected.
    → job dropdown, card html, inputs rows, activity text, queue rows."""
    jid = job_id or latest_active_job()
    pick = gr.update(choices=job_choices(), value=jid) if (jid and not job_id) else gr.update()
    rows, _ = job_detail(jid) if jid else ([], "")
    return pick, progress_card(jid), rows, activity_feed(jid), jobs_table()


def review_counts(job_id):
    from db.repositories import job_repo
    if not job_id:
        return ""
    rows = job_repo.list_deliverables(job_id)
    if not rows:
        return "_No files yet — the job is still running or produced nothing._"
    c = {s: sum(1 for r in rows if r["status"] == s) for s in ("rendered", "approved", "rejected")}
    fails = sum(1 for r in rows if (r.get("qa") or {}).get("compliance") == "fail")
    return (f"**{len(rows)} files** · 🕓 {c['rendered']} to review · ✅ {c['approved']} approved "
            f"· ❌ {c['rejected']} rejected" + (f" · ⚠️ {fails} failed QA" if fails else ""))


def start_reexport(job_id, rewrite_text=True):
    """Re-export a finished job in the background, shown as a running job (banner,
    Progress card, activity feed). → message."""
    import threading
    from campaign import activity
    from campaign.delivery import reexport_job
    from db.repositories import job_repo
    if not job_id:
        return "Pick a job first."
    j = job_repo.get_job(job_id)
    if j["status"] in ("running", "queued"):
        return "⚠️ This job is still running — wait until it finishes."

    def work():
        job_repo.update_job(job_id, status="running", started_at=True, progress=0.0,
                            message="re-exporting files with the current settings")
        try:
            with activity.capture(job_id):
                activity.note(job_id, "▸ re-export started (current text style, one file per "
                                      "format, fresh AI text + captions)")
                res = reexport_job(job_id, rewrite_text=rewrite_text,
                                   progress=lambda m, f: job_repo.update_job(
                                       job_id, progress=min(0.99, f), message=m))
            msg = f"re-exported: {res['before']} → {res['after']} files" + \
                (f", {len(res['failed'])} failed" if res["failed"] else "")
            activity.note(job_id, f"▸ {msg}")
            job_repo.update_job(job_id, status="done", progress=1.0, finished_at=True,
                                message=msg, error="; ".join(res["failed"])[:4000] or None)
        except Exception as e:
            log.exception("re-export crashed")
            job_repo.update_job(job_id, status="failed", finished_at=True, error=str(e)[:4000],
                                message="re-export failed")

    threading.Thread(target=work, daemon=True, name=f"reexport-{job_id[:8]}").start()
    return ("🔄 Re-export started — follow it in **3 · Progress**. Files are replaced one clip "
            "at a time; approvals reset.")


def approve_all_passing(job_id):
    """Approve every not-yet-reviewed file whose QA passed. → table, message."""
    from campaign.delivery import set_status
    from db.repositories import job_repo
    if not job_id:
        return [], "Pick a job first."
    n = 0
    for r in job_repo.list_deliverables(job_id):
        if r["status"] == "rendered" and (r.get("qa") or {}).get("compliance") == "pass":
            set_status(r["id"], "approved")
            n += 1
    return deliverables_table(job_id), f"✅ Approved {n} file(s) that passed every check."


# ── review & deliver ─────────────────────────────────────────────────────────

_ICON = {"pass": "✅", "warn": "⚠️", "fail": "❌", None: "·"}


def deliverables_table(job_id):
    from db.repositories import job_repo
    if not job_id:
        return []
    rows = []
    for n, d in enumerate(job_repo.list_deliverables(job_id), 1):
        qa = d.get("qa") or {}
        from campaign.presets import platform_label
        rows.append([n, os.path.basename(d["path"] or ""),
                     platform_label(d["platform"]) + ("" if d["variant"] == "main"
                                                      else f" / {d['variant']}"),
                     f"{qa.get('duration') or 0:.1f}s",
                     f"{_ICON.get(qa.get('compliance'), '·')} {qa.get('compliance') or '-'}",
                     d["status"]])
    return rows


def _deliverable(job_id, n):
    from db.repositories import job_repo
    rows = job_repo.list_deliverables(job_id) if job_id else []
    try:
        return rows[int(n) - 1]
    except (IndexError, TypeError, ValueError):
        return None


def show_deliverable(job_id, n):
    """→ video, checks rows, posting text, recipe json, status."""
    from campaign import service
    from campaign.delivery import posting_text
    from campaign.presets import platform_label
    from db.repositories import job_repo
    d = _deliverable(job_id, n)
    if not d:
        return None, [], "", "{}", "Pick a row number from the table."
    qa = d.get("qa") or {}
    job = job_repo.get_job(job_id)
    camp = service.get_campaign(job["campaign_id"])
    checks = [[_ICON.get(c["status"]), c["label"], c["detail"]] for c in qa.get("checks") or []]
    from campaign.recipe import variants_of
    recipe = qa.get("recipe_override") or dict(variants_of(job["recipe"])).get(
        d["variant"], job["recipe"])
    if qa.get("onscreen_text"):
        # show the line the AI wrote for this file, so it can be edited and re-rendered
        from campaign.onscreen_text import resolve_auto_text
        recipe = resolve_auto_text(recipe, None, None, stored=qa["onscreen_text"],
                                   generate=False)[0]
    post = posting_text(d, camp, recipe)
    total = len(job_repo.list_deliverables(job_id))
    return (d["path"] if d["path"] and os.path.isfile(d["path"]) else None, checks,
            f"{post['title']}\n\n{post['caption']}", json.dumps(recipe, indent=2),
            f"**#{n} of {total}** · {os.path.basename(d['path'] or '')} · "
            f"{platform_label(d['platform'])} · "
            f"variant {d['variant']} · **{d['status']}**"
            + (" · re-rendered with its own recipe" if qa.get("recipe_override") else ""))


def decide(job_id, n, status):
    """Approve / reject row n, then move to the next row (the last row stays put).
    → table, message, next row number."""
    from campaign.delivery import set_status
    d = _deliverable(job_id, n)
    if not d:
        return deliverables_table(job_id), "⚠️ Pick a row first (click it in the table).", n
    set_status(d["id"], status)
    table = deliverables_table(job_id)
    icon = "✅" if status == "approved" else "❌" if status == "rejected" else "↩️"
    nxt = int(n) + 1 if int(n) < len(table) else int(n)
    msg = f"{icon} #{int(n)} **{status}** — {os.path.basename(d['path'] or '')}"
    msg += f" · showing #{nxt}" if nxt != int(n) else " · that was the last file"
    return table, msg, nxt


def step_row(job_id, n, delta):
    """Row number after Prev (-1) / Next (+1), clamped to the table."""
    total = len(deliverables_table(job_id)) if job_id else 0
    if not total:
        return 1
    try:
        cur = int(n)
    except (TypeError, ValueError):
        cur = 1
    return min(max(cur + int(delta), 1), total)


def row_from_select(evt: gr.SelectData):
    """Row number of the clicked table cell (1-based)."""
    idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    return int(idx) + 1


def rerender_one(job_id, n, recipe_json):
    from campaign.delivery import rerender
    d = _deliverable(job_id, n)
    if not d:
        return deliverables_table(job_id), None, [], "Pick a row number first."
    try:
        recipe = json.loads(recipe_json) if recipe_json and recipe_json.strip() else None
        row = rerender(d["id"], recipe)
        qa = row.get("qa") or {}
        checks = [[_ICON.get(c["status"]), c["label"], c["detail"]] for c in qa.get("checks") or []]
        return deliverables_table(job_id), row["path"], checks, \
            f"Re-rendered #{n} — QA {qa.get('compliance')}."
    except Exception as e:
        return deliverables_table(job_id), gr.update(), gr.update(), _err(e)


def make_zip(job_id, approved_only):
    from campaign.delivery import build_zip
    if not job_id:
        return None, "Pick a job."
    try:
        z = build_zip(job_id, approved_only=bool(approved_only))
        return z, f"Zip ready: {os.path.basename(z)}"
    except Exception as e:
        return None, _err(e)


# ── layout ───────────────────────────────────────────────────────────────────

def start_and_follow(campaign_id, text, files, policy, table):
    """Run job, then jump to the Progress tab on that job. → message, job dropdown, tabs."""
    msg, dd = start_job(campaign_id, text, files, policy, table)
    ok = isinstance(dd, dict) and dd.get("value")
    return msg, dd, (gr.Tabs(selected="progress") if ok else gr.update())


def build_campaign_tabs(tabs=None):
    """Create the four campaign tabs inside the current gr.Blocks (and gr.Tabs `tabs`,
    which lets "Run job" / "Open in Review" switch tabs)."""
    with gr.Tab("1 · Set up campaign", id="setup") as tab1:
        gr.Markdown(
            "A **campaign** is one client or brand: its brief, its files (logo, music…) and "
            "the edit steps that every video gets. Do this once, then start jobs in **2**.")
        with gr.Row():
            # value=None: Gradio 6 otherwise preselects the first entry, so a job could
            # silently run on whichever campaign happened to be listed first
            camp = gr.Dropdown(choices=campaign_choices(), value=None,
                               label="Open an existing campaign (or fill in step A to create one)",
                               scale=3, allow_custom_value=False)
            camp_refresh = gr.Button("↻", scale=0, min_width=40)
        c_status = gr.Markdown()
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### A · Name and brief")
                c_name = gr.Textbox(label="Campaign name")
                c_brief = gr.Textbox(label="Brief — paste the requirements exactly as the "
                                           "client wrote them", lines=8)
                c_platforms = gr.CheckboxGroup(PLATFORM_CHOICES, label="Platforms")
                with gr.Row():
                    c_new = gr.Button("Create as new campaign", variant="primary")
                    c_save = gr.Button("Save changes")
                gr.Markdown("### B · Brand files (optional)\nLogos, music, fonts, intro/outro "
                            "videos — upload them **before** step C so the brief can use them.")
                c_files = gr.File(label="Upload files", file_count="multiple")
                with gr.Row():
                    c_kind = gr.Dropdown(KINDS, value="auto", label="Kind", scale=1)
                    c_add = gr.Button("Add files", scale=1)
                c_assets = gr.Dataframe(headers=["kind", "name", "duration", "size"],
                                        interactive=False, label="Files in this campaign")
                with gr.Accordion("Remove a file", open=False):
                    with gr.Row():
                        c_rm_name = gr.Textbox(label="File name to remove", scale=3)
                        c_rm = gr.Button("Remove", scale=1)
            with gr.Column(scale=1):
                gr.Markdown("### C · Turn the brief into edit steps\nThe AI reads the brief "
                            "and lists exactly what will be done to every video.")
                with gr.Row():
                    c_parse = gr.Button("Read the brief", variant="primary")
                    c_llm = gr.Checkbox(value=True, label="Use AI (off = simple keyword reader)")
                c_summary = gr.Markdown()
                gr.Markdown("### D · Answer open questions, then save")
                c_questions = gr.Markdown()
                with gr.Accordion("Advanced: edit the steps as JSON", open=False):
                    c_recipe = gr.Code(label="Recipe (editable JSON)", language="json",
                                       value="{}")
                c_save_recipe = gr.Button("Save edit steps", variant="primary")

        camp.change(load_campaign, [camp], [c_name, c_brief, c_platforms, c_assets, c_recipe,
                                            c_summary, c_questions, c_status])
        camp_refresh.click(lambda: gr.update(choices=campaign_choices()), None, [camp])
        c_new.click(create_campaign, [c_name, c_brief, c_platforms], [camp, c_status])
        c_save.click(save_details, [camp, c_name, c_brief, c_platforms], [c_status])
        c_add.click(add_assets, [camp, c_files, c_kind], [c_assets, c_status])
        c_rm.click(remove_asset, [camp, c_rm_name], [c_assets, c_status])
        c_parse.click(parse_brief, [camp, c_brief, c_llm],
                      [c_recipe, c_summary, c_questions, c_status])
        c_save_recipe.click(save_recipe, [camp, c_recipe, c_brief],
                            [c_recipe, c_summary, c_questions, c_status, camp])

    with gr.Tab("2 · Start a job", id="start") as tab2:
        gr.Markdown("Pick the campaign, add the videos, press **Run job**. You'll be taken "
                    "to **3 · Progress** to follow it.")
        with gr.Row():
            j_camp = gr.Dropdown(choices=campaign_choices(), value=None,
                                 label="1. Campaign", scale=3)
            j_camp_refresh = gr.Button("↻", scale=0, min_width=40)
        j_summary = gr.Markdown("_Pick the campaign this job belongs to._")
        j_text = gr.Textbox(label="2. Videos — one per line: YouTube / Drive / direct links, "
                                  "file paths or folders", lines=4)
        j_files = gr.File(label="…or upload video files", file_count="multiple")
        with gr.Accordion("Options: clip or edit mode, check videos first", open=False):
            j_policy = gr.Radio(MODE_CHOICES, value="auto", label="Mode for all videos "
                                "(auto = videos over 3 min are cut into clips, shorter ones "
                                "are edited whole)")
            j_check = gr.Button("Check videos")
            j_table = gr.Dataframe(headers=["#", "source", "duration", "aspect", "mode", "why",
                                            "override"],
                                   interactive=True,
                                   label="Videos (type clip / edit in override to force a mode)")
        j_run = gr.Button("3. Run job ▶", variant="primary", size="lg")
        j_status = gr.Markdown()

    with gr.Tab("3 · Progress", id="progress") as tab3:
        gr.Markdown("Updates every few seconds. A running job is opened automatically.")
        with gr.Row():
            q_job = gr.Dropdown(choices=job_choices(), value=None, label="Job", scale=3)
            q_refresh = gr.Button("↻ Refresh", scale=0)
        q_card = gr.HTML()
        q_feed = gr.Textbox(label="What's happening (newest first)", lines=10, max_lines=16,
                            interactive=False)
        with gr.Row():
            q_review = gr.Button("Open in Review ▶", variant="primary")
            q_cancel = gr.Button("Cancel job")
            q_resume = gr.Button("Resume / retry failed videos")
        q_msg = gr.Markdown()
        with gr.Accordion("Details: videos in this job and all jobs", open=False):
            q_inputs = gr.Dataframe(headers=["#", "status", "mode", "source", "error"],
                                    interactive=False, label="Videos in this job")
            q_table = gr.Dataframe(headers=["job", "campaign", "status", "progress", "message",
                                            "created"], interactive=False, label="All jobs")
        q_detail = gr.Markdown(visible=False)
        q_timer = gr.Timer(4.0)

    with gr.Tab("4 · Review & download", id="review") as tab4:
        gr.Markdown("Watch each file, then **Approve** or **Reject** — it moves to the next one "
                    "by itself. Download the approved files as a zip at the bottom.")
        with gr.Row():
            r_job = gr.Dropdown(choices=job_choices(), value=None, label="Job", scale=3)
            r_refresh = gr.Button("↻", scale=0, min_width=40)
        r_counts = gr.Markdown()
        # capped height: with 40+ files the preview used to sit far below the table
        r_table = gr.Dataframe(headers=["#", "file", "post to", "duration", "QA", "status"],
                               interactive=False, label="Files (click a row to open it)",
                               max_height=260)
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    r_prev = gr.Button("◀ Prev", scale=1, min_width=80)
                    r_n = gr.Number(label="Row #", value=1, precision=0, minimum=1, scale=1)
                    r_next = gr.Button("Next ▶", scale=1, min_width=80)
                r_status = gr.Markdown()
                r_video = gr.Video(label="Preview")
                with gr.Row():
                    r_ok = gr.Button("✅ Approve", variant="primary")
                    r_no = gr.Button("❌ Reject", variant="stop")
                r_decision = gr.Markdown()
            with gr.Column(scale=1):
                r_checks = gr.Dataframe(headers=["", "check", "detail"], interactive=False,
                                        label="Quality checks")
                r_post = gr.Textbox(label="Posting text (title + caption) — copy when you post",
                                    lines=4)
                with gr.Accordion("Change this file (text, logo, timing…) and re-render",
                                  open=False):
                    r_recipe = gr.Code(label="Edit steps for THIS file only", language="json",
                                       value="{}")
                    r_rerender = gr.Button("Re-render this file")
        gr.Markdown("### Download")
        with gr.Row():
            r_all = gr.Button("Approve all files that passed QA")
            r_approved = gr.Checkbox(value=True, label="Approved files only")
            r_zip = gr.Button("Build zip", variant="primary")
        r_file = gr.File(label="Download")
        r_zip_status = gr.Markdown()
        with gr.Accordion("Re-export all files with the campaign's current settings", open=False):
            gr.Markdown("Rebuilds every file of this job from the already-framed clips: current "
                        "text style, one file per format, fresh AI on-screen text and captions. "
                        "No re-analysis. Takes about 1–2 min per clip; approvals reset.")
            with gr.Row():
                r_rewrite = gr.Checkbox(value=True, label="Write new on-screen text")
                r_reexport = gr.Button("Re-export all files")

    # dropdowns are filled when the app starts: refresh them whenever a tab is opened so
    # campaigns / jobs created since then are listed (otherwise a stale list made the
    # wrong campaign easy to pick)
    tab1.select(lambda: gr.update(choices=campaign_choices()), None, [camp])
    tab2.select(lambda: gr.update(choices=campaign_choices()), None, [j_camp])
    tab3.select(lambda: gr.update(choices=job_choices()), None, [q_job])\
        .then(refresh_progress, [q_job], [q_job, q_card, q_inputs, q_feed, q_table])
    tab4.select(lambda: gr.update(choices=job_choices()), None, [r_job])

    # start a job → follow it
    j_camp_refresh.click(lambda: gr.update(choices=campaign_choices()), None, [j_camp])
    j_camp.change(campaign_summary, [j_camp], [j_summary])
    j_check.click(check_inputs, [j_camp, j_text, j_files, j_policy], [j_table, j_status])
    run_outputs = [j_status, q_job] + ([tabs] if tabs is not None else [])
    j_run.click((lambda *a: start_and_follow(*a)) if tabs is not None
                else (lambda *a: start_job(*a)),
                [j_camp, j_text, j_files, j_policy, j_table], run_outputs)

    # progress
    prog_out = [q_job, q_card, q_inputs, q_feed, q_table]
    q_refresh.click(lambda j: (gr.update(choices=job_choices()),) + refresh_progress(j)[1:],
                    [q_job], prog_out)
    q_job.change(refresh_progress, [q_job], prog_out)
    q_timer.tick(refresh_progress, [q_job], prog_out)
    q_cancel.click(cancel_job, [q_job], [q_msg])
    q_resume.click(resume_job, [q_job], [q_msg])
    q_review.click(lambda j: ((gr.update(choices=job_choices(), value=j),
                               gr.Tabs(selected="review")) if tabs is not None
                              else (gr.update(choices=job_choices(), value=j),)),
                   [q_job], [r_job] + ([tabs] if tabs is not None else []))

    # review
    r_refresh.click(lambda j: (gr.update(choices=job_choices()), deliverables_table(j),
                               review_counts(j)), [r_job], [r_job, r_table, r_counts])
    r_n.change(show_deliverable, [r_job, r_n], [r_video, r_checks, r_post, r_recipe, r_status])
    r_table.select(row_from_select, None, [r_n])
    r_prev.click(lambda j, n: step_row(j, n, -1), [r_job, r_n], [r_n])
    r_next.click(lambda j, n: step_row(j, n, 1), [r_job, r_n], [r_n])
    # opening a job: table, row 1 and a cleared message in ONE update, then show row 1
    # (separate handlers raced a table click and could jump the row back to 1)
    r_job.change(lambda j: (deliverables_table(j), 1, "", review_counts(j)), [r_job],
                 [r_table, r_n, r_decision, r_counts])\
        .then(show_deliverable, [r_job, r_n], [r_video, r_checks, r_post, r_recipe, r_status])
    r_ok.click(lambda j, n: decide(j, n, "approved"), [r_job, r_n], [r_table, r_decision, r_n])\
        .then(show_deliverable, [r_job, r_n], [r_video, r_checks, r_post, r_recipe, r_status])\
        .then(review_counts, [r_job], [r_counts])
    r_no.click(lambda j, n: decide(j, n, "rejected"), [r_job, r_n], [r_table, r_decision, r_n])\
        .then(show_deliverable, [r_job, r_n], [r_video, r_checks, r_post, r_recipe, r_status])\
        .then(review_counts, [r_job], [r_counts])
    r_all.click(approve_all_passing, [r_job], [r_table, r_zip_status])\
        .then(review_counts, [r_job], [r_counts])
    r_rerender.click(rerender_one, [r_job, r_n, r_recipe], [r_table, r_video, r_checks, r_status])
    r_zip.click(make_zip, [r_job, r_approved], [r_file, r_zip_status])
    r_reexport.click(start_reexport, [r_job, r_rewrite], [r_zip_status])
