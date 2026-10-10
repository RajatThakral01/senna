"""campaign/cli.py — `python main.py campaign …` / `python main.py job …` (C5).

  campaign create --name "Acme Summer" --brief "…" [--brief-file f.txt]
                  [--asset logo.png --asset track.mp3 …] [--platform tiktok …] [--no-llm]
  campaign list
  campaign show  <campaign id | slug>
  campaign brief <campaign> [--brief "…"] [--no-llm]      re-parse + save the recipe
  job run    --campaign <c> --input a.mp4 --input https://… [--input folder/]
             [--mode auto|clip|edit]                     create + run (blocking)
  job resume <job id>                                    re-run unfinished inputs
  job status [<job id>]                                  one job, or the latest jobs
"""
import argparse
import json
import sys


def _campaign(ref):
    from campaign import service
    from db.repositories import campaign_repo
    c = campaign_repo.get_campaign_by_slug(ref)
    if c:
        return c
    try:
        return service.get_campaign(ref)
    except Exception:
        raise SystemExit(f"no campaign with id or slug {ref!r}")


def _job(ref):
    from db.repositories import job_repo
    if len(ref) < 36:   # allow the 8-char prefix shown in listings
        for j in job_repo.list_jobs(limit=500):
            if j["id"].startswith(ref):
                return j
    else:
        j = job_repo.get_job(ref)
        if j:
            return j
    raise SystemExit(f"no job {ref!r}")


def _print_recipe(validation):
    from campaign.recipe import describe_recipe
    print("\nRecipe:")
    for line in describe_recipe(validation["recipe"]):
        print("  •", line)
    for w in validation.get("warnings") or []:
        print("  ! ", w)
    for e in validation.get("errors") or []:
        print("  ✗ ", e)
    for q in validation.get("questions") or []:
        extra = f" (default: {q['default']})" if q.get("default") is not None else ""
        print("  ? ", q["question"] + extra)
    print("Status:", "ready" if validation.get("ok") else
          "draft — answer the questions / fix the errors, then `campaign brief` again")


def _parse_and_save(c, brief, use_llm):
    from campaign import service
    parsed = service.parse_brief(c["id"], brief, use_llm=use_llm)
    saved = service.save_recipe(c["id"], parsed["recipe"],
                                requirements={"brief": brief if brief is not None else c["brief"],
                                              "method": parsed["method"],
                                              "summary": parsed.get("brief_summary")})
    print(f"Brief parsed ({parsed['method']}).")
    v = saved["validation"]
    v["questions"] = parsed.get("questions") or v["questions"]
    _print_recipe(v)


def _progress(frac, msg):
    print(f"  {frac * 100:5.1f}%  {msg}", flush=True)


def _print_status(st):
    j = st["job"]
    print(f"job {j['id'][:8]}  {j['status']}  {float(j['progress'] or 0) * 100:.0f}%  "
          f"{j.get('message') or ''}")
    if j.get("error"):
        print("  errors:", j["error"])
    for i in st["inputs"]:
        print(f"  [{i['idx']}] {i['status']:<8} {i.get('mode') or '-':<5} {i['source']}"
              + (f"  — {i['error'][:160]}" if i.get("error") else ""))
    for d in st["deliverables"]:
        qa = d.get("qa") or {}
        print(f"      → {d['platform']:<8} {qa.get('width')}x{qa.get('height')} "
              f"{qa.get('duration') or 0:.1f}s  {d['path']}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="main.py", description="Campaign jobs")
    sub = p.add_subparsers(dest="group", required=True)

    pc = sub.add_parser("campaign").add_subparsers(dest="cmd", required=True)
    c_create = pc.add_parser("create")
    c_create.add_argument("--name", required=True)
    c_create.add_argument("--brief", default=None)
    c_create.add_argument("--brief-file", default=None)
    c_create.add_argument("--asset", action="append", default=[])
    c_create.add_argument("--platform", action="append", default=[])
    c_create.add_argument("--no-llm", action="store_true")
    pc.add_parser("list")
    c_show = pc.add_parser("show")
    c_show.add_argument("campaign")
    c_brief = pc.add_parser("brief")
    c_brief.add_argument("campaign")
    c_brief.add_argument("--brief", default=None)
    c_brief.add_argument("--no-llm", action="store_true")

    pj = sub.add_parser("job").add_subparsers(dest="cmd", required=True)
    j_run = pj.add_parser("run")
    j_run.add_argument("--campaign", required=True)
    j_run.add_argument("--input", action="append", required=True)
    j_run.add_argument("--mode", default="auto", choices=["auto", "clip", "edit"])
    j_resume = pj.add_parser("resume")
    j_resume.add_argument("job")
    j_status = pj.add_parser("status")
    j_status.add_argument("job", nargs="?")

    a = p.parse_args(argv)
    from campaign import jobs, service

    if a.group == "campaign":
        if a.cmd == "create":
            brief = a.brief
            if a.brief_file:
                with open(a.brief_file, encoding="utf-8") as f:
                    brief = f.read()
            c = service.create_campaign(a.name, brief or "", a.platform or None)
            print(f"campaign {c['slug']} ({c['id'][:8]}) created")
            for path in a.asset:
                asset = service.add_asset(c["id"], path)
                print(f"  asset {asset['kind']:<5} {asset['name']}")
            if brief:
                _parse_and_save(c, brief, not a.no_llm)
        elif a.cmd == "list":
            for c in service.list_campaigns():
                print(f"{c['id'][:8]}  {c['slug']:<32} {c['status']:<6} v{c['recipe_version']}  "
                      f"{c['name']}")
        elif a.cmd == "show":
            c = _campaign(a.campaign)
            print(json.dumps({k: c[k] for k in ("id", "name", "slug", "status", "recipe_version",
                                                "platforms", "brief")}, indent=2, default=str))
            for x in service.asset_catalog(c["id"]):
                print(f"  asset {x['kind']:<5} {x['name']}")
            if c.get("recipe"):
                from campaign.recipe import validate_recipe
                _print_recipe(validate_recipe(c["recipe"], service.asset_catalog(c["id"])))
        elif a.cmd == "brief":
            c = _campaign(a.campaign)
            if a.brief is not None:
                c = service.update_campaign(c["id"], brief=a.brief)
            _parse_and_save(c, a.brief, not a.no_llm)
        return 0

    if a.cmd == "run":
        c = _campaign(a.campaign)
        try:
            job = jobs.create_job(c["id"], a.input, mode_policy=a.mode)
        except ValueError as e:
            raise SystemExit(str(e))
        from db.repositories import job_repo
        print(f"job {job['id'][:8]} — {len(job_repo.list_inputs(job['id']))} input(s)")
        jobs.run_job(job["id"], progress=_progress)
        _print_status(jobs.status(job["id"]))
    elif a.cmd == "resume":
        j = _job(a.job)
        jobs.run_job(j["id"], progress=_progress)
        _print_status(jobs.status(j["id"]))
    elif a.cmd == "status":
        if a.job:
            _print_status(jobs.status(_job(a.job)["id"]))
        else:
            from db.repositories import job_repo
            for j in job_repo.list_jobs(limit=20):
                print(f"{j['id'][:8]}  {j['status']:<9} {float(j['progress'] or 0) * 100:3.0f}%  "
                      f"{j['created_at']:%Y-%m-%d %H:%M}  {j.get('message') or ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
