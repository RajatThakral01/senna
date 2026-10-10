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


# ── review & deliver ─────────────────────────────────────────────────────────

_ICON = {"pass": "✅", "warn": "⚠️", "fail": "❌", None: "·"}


def deliverables_table(job_id):
    from db.repositories import job_repo
    if not job_id:
        return []
    rows = []
    for n, d in enumerate(job_repo.list_deliverables(job_id), 1):
        qa = d.get("qa") or {}
        rows.append([n, os.path.basename(d["path"] or ""), f"{d['platform']} / {d['variant']}",
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
    post = posting_text(d, camp, recipe)
    return (d["path"] if d["path"] and os.path.isfile(d["path"]) else None, checks,
            f"{post['title']}\n\n{post['caption']}", json.dumps(recipe, indent=2),
            f"#{n} · {d['platform']} · variant {d['variant']} · {d['status']}"
            + (" · re-rendered with its own recipe" if qa.get("recipe_override") else ""))


def decide(job_id, n, status):
    from campaign.delivery import set_status
    d = _deliverable(job_id, n)
    if not d:
        return deliverables_table(job_id), "Pick a row number first."
    set_status(d["id"], status)
    return deliverables_table(job_id), f"#{n} {status}."


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

def build_campaign_tabs():
    """Create the four campaign tabs inside the current gr.Blocks."""
    with gr.Tab("1 · Campaigns"):
        with gr.Row():
            camp = gr.Dropdown(choices=campaign_choices(), label="Campaign", scale=3,
                               allow_custom_value=False)
            camp_refresh = gr.Button("↻", scale=0, min_width=40)
        with gr.Row():
            with gr.Column(scale=1):
                c_name = gr.Textbox(label="Name")
                c_brief = gr.Textbox(label="Brief (requirements as the client wrote them)",
                                     lines=8)
                c_platforms = gr.CheckboxGroup(PLATFORM_CHOICES, label="Platforms")
                with gr.Row():
                    c_new = gr.Button("Create as new campaign")
                    c_save = gr.Button("Save details")
                gr.Markdown("#### Asset library")
                c_files = gr.File(label="Upload logos / audio / fonts / videos / images",
                                  file_count="multiple")
                with gr.Row():
                    c_kind = gr.Dropdown(KINDS, value="auto", label="Kind")
                    c_add = gr.Button("Add assets")
                c_assets = gr.Dataframe(headers=["kind", "name", "duration", "size"],
                                        interactive=False, label="Assets")
                with gr.Row():
                    c_rm_name = gr.Textbox(label="Asset name to remove", scale=3)
                    c_rm = gr.Button("Remove", scale=1)
            with gr.Column(scale=1):
                with gr.Row():
                    c_llm = gr.Checkbox(value=True, label="Use the LLM (off = keyword parser)")
                    c_parse = gr.Button("Parse brief", variant="primary")
                c_summary = gr.Markdown()
                gr.Markdown("#### Questions")
                c_questions = gr.Markdown()
                c_recipe = gr.Code(label="Recipe (editable JSON)", language="json", value="{}")
                c_save_recipe = gr.Button("Save recipe", variant="primary")
        c_status = gr.Markdown()

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

    with gr.Tab("2 · New job"):
        with gr.Row():
            j_camp = gr.Dropdown(choices=campaign_choices(), label="Campaign", scale=3)
            j_camp_refresh = gr.Button("↻", scale=0, min_width=40)
        j_text = gr.Textbox(label="Inputs — one per line: YouTube / Drive / direct links, "
                                  "local file paths or folders", lines=5)
        j_files = gr.File(label="…or upload videos", file_count="multiple")
        j_policy = gr.Radio(MODE_CHOICES, value="auto", label="Mode for all inputs "
                            "(auto = long videos are cut into clips, short ones edited)")
        j_check = gr.Button("Check inputs")
        j_table = gr.Dataframe(headers=["#", "source", "duration", "aspect", "mode", "why",
                                        "override"],
                               interactive=True, label="Inputs (set override to clip / edit)")
        j_run = gr.Button("Run job", variant="primary")
        j_status = gr.Markdown()

    with gr.Tab("3 · Jobs"):
        with gr.Row():
            q_job = gr.Dropdown(choices=job_choices(), label="Job", scale=3)
            q_refresh = gr.Button("↻ Refresh", scale=0)
        q_table = gr.Dataframe(headers=["job", "campaign", "status", "progress", "message",
                                        "created"], interactive=False, label="Queue")
        q_detail = gr.Markdown()
        q_inputs = gr.Dataframe(headers=["#", "status", "mode", "source", "error"],
                                interactive=False, label="Inputs")
        with gr.Row():
            q_cancel = gr.Button("Cancel job")
            q_resume = gr.Button("Resume / retry failed inputs")
        q_msg = gr.Markdown()
        q_timer = gr.Timer(4.0)

    with gr.Tab("4 · Review & deliver"):
        with gr.Row():
            r_job = gr.Dropdown(choices=job_choices(), label="Job", scale=3)
            r_refresh = gr.Button("↻", scale=0, min_width=40)
        r_table = gr.Dataframe(headers=["#", "file", "platform", "duration", "QA", "status"],
                               interactive=False, label="Deliverables")
        with gr.Row():
            with gr.Column(scale=1):
                r_n = gr.Number(label="Row #", value=1, precision=0)
                r_video = gr.Video(label="Preview")
                with gr.Row():
                    r_ok = gr.Button("Approve", variant="primary")
                    r_no = gr.Button("Reject", variant="stop")
            with gr.Column(scale=1):
                r_checks = gr.Dataframe(headers=["", "check", "detail"], interactive=False,
                                        label="Compliance")
                r_post = gr.Textbox(label="Posting text (title + caption)", lines=4)
                r_recipe = gr.Code(label="Recipe for THIS file (tweak, then re-render)",
                                   language="json", value="{}")
                r_rerender = gr.Button("Re-render this file")
        with gr.Row():
            r_approved = gr.Checkbox(value=True, label="Approved files only")
            r_zip = gr.Button("Build zip", variant="primary")
        r_file = gr.File(label="Download")
        r_status = gr.Markdown()

    # cross-tab wiring
    j_camp_refresh.click(lambda: gr.update(choices=campaign_choices()), None, [j_camp])
    j_check.click(check_inputs, [j_camp, j_text, j_files, j_policy], [j_table, j_status])
    j_run.click(start_job, [j_camp, j_text, j_files, j_policy, j_table], [j_status, q_job])

    q_refresh.click(lambda j: (gr.update(choices=job_choices()),) + refresh_jobs(j), [q_job],
                    [q_job, q_table, q_inputs, q_detail])
    q_job.change(refresh_jobs, [q_job], [q_table, q_inputs, q_detail])
    q_timer.tick(refresh_jobs, [q_job], [q_table, q_inputs, q_detail])
    q_cancel.click(cancel_job, [q_job], [q_msg])
    q_resume.click(resume_job, [q_job], [q_msg])

    r_refresh.click(lambda j: (gr.update(choices=job_choices()), deliverables_table(j)), [r_job],
                    [r_job, r_table])
    r_job.change(deliverables_table, [r_job], [r_table])
    r_n.change(show_deliverable, [r_job, r_n], [r_video, r_checks, r_post, r_recipe, r_status])
    r_job.change(show_deliverable, [r_job, r_n], [r_video, r_checks, r_post, r_recipe, r_status])
    r_ok.click(lambda j, n: decide(j, n, "approved"), [r_job, r_n], [r_table, r_status])
    r_no.click(lambda j, n: decide(j, n, "rejected"), [r_job, r_n], [r_table, r_status])
    r_rerender.click(rerender_one, [r_job, r_n, r_recipe], [r_table, r_video, r_checks, r_status])
    r_zip.click(make_zip, [r_job, r_approved], [r_file, r_status])
