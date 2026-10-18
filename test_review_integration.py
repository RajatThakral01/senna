# test_review_integration.py — integration (Docker DB + ffmpeg, real renders) for the
# 15 Oct fixes: AI-written on-screen text through a real job, review navigation,
# approve/reject auto-advance persisted in the DB, re-render keeps the written text,
# duplicate-job guard, New-job campaign summary, Clip-review video picker.
# The LLM is stubbed (deterministic, no API spend); everything else is real.
import json
import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

import campaign_ui as U  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(not FFMPEG, reason="ffmpeg not installed")

BRIEF = ("Clip the footage. Text on screen is required for every post — write your own, "
         "something people want to reply to. No negativity. Two versions: one with our "
         "logo and one without. TikTok.")


def _ff(*args):
    r = subprocess.run([FFMPEG, "-loglevel", "error", "-y", *args], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]


@pytest.fixture
def env(tmp_path, monkeypatch):
    from campaign import edit_mode, service, onscreen_text
    monkeypatch.setattr(service, "ASSET_ROOT", str(tmp_path / "assets"))
    monkeypatch.setattr(edit_mode, "OUTPUT_ROOT", str(tmp_path / "output"))
    monkeypatch.setattr(edit_mode, "WORK_ROOT", str(tmp_path / "work"))
    calls = []

    def fake_write(brief, transcript, context="", avoid=(), focus=None):
        calls.append({"brief": brief, "avoid": list(avoid)})
        return {"text": f"Line {len(calls)}: name a better moment",
                "caption": f"Caption {len(calls)} @brand #brand"}
    monkeypatch.setattr(onscreen_text, "write_post", fake_write)
    _ff("-f", "lavfi", "-i", "testsrc2=size=360x640:rate=30:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=300:duration=3", "-shortest",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(tmp_path / "promo.mp4"))
    _ff("-f", "lavfi", "-i", "color=c=0x2040F0:size=300x120,format=rgba", "-frames:v", "1",
        str(tmp_path / "brand_logo.png"))
    made = []
    yield tmp_path, made, calls
    from db.repositories import campaign_repo
    for cid in made:
        campaign_repo.delete_campaign(cid)


def _campaign(tmp, made, recipe):
    from campaign import service
    c = service.create_campaign("Review Integration Test", BRIEF, ["tiktok"])
    made.append(c["id"])
    service.add_asset(c["id"], str(tmp / "brand_logo.png"), kind="logo")
    out = service.save_recipe(c["id"], recipe, requirements={"brief": BRIEF})
    assert out["validation"]["ok"], out["validation"]
    return c["id"]


RECIPE = {"mode": "edit",
          "ops": [{"op": "logo", "asset": "brand_logo.png", "position": "top-left"},
                  {"op": "text_overlay", "text": "{auto}", "role": "hook"}],
          "export": ["tiktok"],
          "variants": [{"name": "no-logo", "ops": [], "remove": ["logo"]}]}


def _run(cid, src):
    from campaign import jobs
    job = jobs.create_job(cid, [src])
    jobs.run_job(job["id"])
    return job["id"]


def test_auto_text_review_flow(env):
    tmp, made, calls = env
    cid = _campaign(tmp, made, RECIPE)

    # New job tab shows what will happen before anything renders
    md = U.campaign_summary(cid)
    assert "Ready to run" in md and "written by AI" in md and "brand_logo.png" in md

    job_id = _run(cid, str(tmp / "promo.mp4"))
    from db.repositories import job_repo
    rows = job_repo.list_deliverables(job_id)

    # live progress: the run wrote a plain-language activity feed, the card renders
    from campaign import activity
    feed = "\n".join(activity.tail(job_id, 50))
    assert "Exported" in feed and "▸ [1/1]" in feed, feed
    card = U.progress_card(job_id)
    assert "Finished" in card and "100% done" in card and "2 file(s) ready" in card
    assert "2 files" in U.review_counts(job_id)
    assert len(rows) == 2                                         # main + no-logo
    texts = [r["qa"]["onscreen_text"] for r in rows]
    assert texts == [["Line 1: name a better moment"], ["Line 2: name a better moment"]]
    assert calls[0]["brief"] == BRIEF and calls[1]["avoid"] == ["Line 1: name a better moment"]
    assert all("text_overlay" in (r["qa"].get("applied") or []) for r in rows)

    # review: the written line is shown in the editable recipe, row info is clear
    video, checks, post, rjson, st = U.show_deliverable(job_id, 1)
    assert os.path.isfile(video)
    assert "Caption 1 @brand #brand" in post                        # AI caption shown
    ops = json.loads(rjson)["ops"]
    assert next(o for o in ops if o["op"] == "text_overlay")["text"] == "Line 1: name a better moment"
    assert "#1 of 2" in st

    # navigation + approve → auto-advance, persisted
    assert U.step_row(job_id, 1, 1) == 2 and U.step_row(job_id, 2, 1) == 2
    table, msg, nxt = U.decide(job_id, 1, "approved")
    assert nxt == 2 and table[0][5] == "approved"
    table, msg, nxt = U.decide(job_id, 2, "rejected")
    assert nxt == 2 and table[1][5] == "rejected" and "last file" in msg
    assert [r["status"] for r in job_repo.list_deliverables(job_id)] == ["approved", "rejected"]

    # re-render with the job recipe keeps the approved wording (no new LLM call)
    n_calls = len(calls)
    table, video2, checks2, st2 = U.rerender_one(job_id, 1, "")
    assert "Re-rendered #1" in st2 and len(calls) == n_calls
    assert job_repo.list_deliverables(job_id)[0]["qa"]["onscreen_text"] == \
        ["Line 1: name a better moment"]

    # editing the line by hand and re-rendering uses exactly that text
    r = json.loads(U.show_deliverable(job_id, 1)[3])
    for o in r["ops"]:
        if o["op"] == "text_overlay":
            o["text"] = "Edited by hand"
    U.rerender_one(job_id, 1, json.dumps(r))
    shown = json.loads(U.show_deliverable(job_id, 1)[3])
    assert next(o for o in shown["ops"] if o["op"] == "text_overlay")["text"] == "Edited by hand"
    assert len(calls) == n_calls


def test_duplicate_job_refused_while_active(env):
    tmp, made, _ = env
    cid = _campaign(tmp, made, RECIPE)
    from campaign import jobs
    from db.repositories import job_repo
    first = jobs.create_job(cid, [str(tmp / "promo.mp4")])         # queued, not run
    with pytest.raises(ValueError, match="already queued"):
        jobs.create_job(cid, [str(tmp / "promo.mp4")])
    msg, _ = U.start_job(cid, str(tmp / "promo.mp4"), None, "auto", [])
    assert "already queued" in msg
    job_repo.update_job(first["id"], status="done")
    assert jobs.create_job(cid, [str(tmp / "promo.mp4")])["id"] != first["id"]


def test_draft_campaign_cannot_run(env):
    tmp, made, _ = env
    from campaign import service
    c = service.create_campaign("Review Draft Test", "text on screen", ["tiktok"])
    made.append(c["id"])
    service.save_recipe(c["id"], {"mode": "edit", "ops": [{"op": "text_overlay", "text": None}],
                                  "export": ["tiktok"]}, requirements={})
    assert "Open questions" in U.campaign_summary(c["id"])
    msg, _ = U.start_job(c["id"], str(tmp / "promo.mp4"), None, "auto", [])
    assert "not ready" in msg


def test_recent_videos_picker():
    import ui
    from db.repositories import video_repo
    vids = video_repo.list_recent(5)
    choices = ui._recent_video_choices()
    assert len(choices) == len(vids)
    if vids:
        assert choices[0][1] == vids[0]["id"] and "clips" in choices[0][0]


def test_reexport_merges_platform_copies_into_one_file(env):
    # 16 Oct: old jobs had one byte-identical copy per platform (42 files for 14 clips)
    import shutil as _sh
    tmp, made, calls = env
    cid = _campaign(tmp, made, dict(RECIPE, variants=[], export=["tiktok"]))
    job_id = _run(cid, str(tmp / "promo.mp4"))
    from db.repositories import job_repo
    from campaign.delivery import reexport_job
    d = job_repo.list_deliverables(job_id)[0]
    copy = d["path"].replace("_tiktok.mp4", "_reels.mp4")
    _sh.copy2(d["path"], copy)                                    # the old-style duplicate
    dup = job_repo.add_deliverable(job_id, d["job_input_id"], variant="main", platform="reels",
                                   path=copy, recipe_version=1)
    job_repo.update_deliverable(dup["id"], qa=d["qa"], status="approved")
    assert len(job_repo.list_deliverables(job_id)) == 2
    n_calls = len(calls)

    res = reexport_job(job_id)
    rows = job_repo.list_deliverables(job_id)
    assert res == {"before": 2, "after": 1, "failed": []}
    assert len(rows) == 1 and rows[0]["platform"] == "tiktok+reels"
    assert rows[0]["path"].endswith("_tiktok-reels.mp4") and os.path.isfile(rows[0]["path"])
    assert not os.path.exists(copy) and not os.path.exists(d["path"])
    assert rows[0]["status"] == "rendered" and rows[0]["qa"]["compliance"] == "pass"
    assert len(calls) == n_calls + 1                              # fresh line + caption
    assert rows[0]["qa"]["post"]["caption"].startswith("Caption")
    assert U.deliverables_table(job_id)[0][2] == "TikTok · Reels"
