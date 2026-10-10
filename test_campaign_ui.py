# test_campaign_ui.py — campaign pipeline C7: the UI handlers end to end
# (create → assets → parse → save → check inputs → run → review → re-render → zip),
# called exactly as the Gradio buttons call them. Needs the Docker DB + ffmpeg.
import json
import os
import shutil
import subprocess
import sys
import zipfile

import pytest

sys.path.insert(0, os.path.dirname(__file__))

import campaign_ui as U  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(not FFMPEG, reason="ffmpeg not installed")


def _ff(*args):
    r = subprocess.run([FFMPEG, "-loglevel", "error", "-y", *args], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]


@pytest.fixture
def env(tmp_path, monkeypatch):
    from campaign import edit_mode, service
    monkeypatch.setattr(service, "ASSET_ROOT", str(tmp_path / "assets"))
    monkeypatch.setattr(edit_mode, "OUTPUT_ROOT", str(tmp_path / "output"))
    monkeypatch.setattr(edit_mode, "WORK_ROOT", str(tmp_path / "work"))
    _ff("-f", "lavfi", "-i", "testsrc2=size=360x640:rate=30:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=300:duration=3", "-shortest",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(tmp_path / "promo.mp4"))
    _ff("-f", "lavfi", "-i", "color=c=0x2040F0:size=300x120,format=rgba", "-frames:v", "1",
        str(tmp_path / "acme_logo.png"))
    made = []
    yield tmp_path, made
    from db.repositories import campaign_repo
    for cid in made:
        campaign_repo.delete_campaign(cid)


def test_full_flow(env):
    from campaign import jobs
    tmp, made = env
    dd, msg = U.create_campaign("UI Flow Test", "Add the logo bottom left and the text "
                                "\"Shop now\" for the last 2 seconds. TikTok. #acme", ["tiktok"])
    cid = dd["value"]
    made.append(cid)
    assert "Created" in msg
    table, msg = U.add_assets(cid, [str(tmp / "acme_logo.png")], "auto")
    assert table == [["logo", "acme_logo.png", "", "300x120"]]

    brief = U.load_campaign(cid)[1]
    recipe_json, summary, questions, status = U.parse_brief(cid, brief, use_llm=False)
    assert "keyword" in status and "Logo acme_logo.png bottom-left" in summary
    rec = json.loads(recipe_json)
    out = U.save_recipe(cid, json.dumps(rec), brief)
    assert "ready" in out[3] and "Ready to run" in out[1]

    rows, _ = U.check_inputs(cid, str(tmp / "promo.mp4"), None, "auto")
    assert rows[0][2:5] == ["3s", "9:16", "edit"]
    # start_job queues on the worker; run it synchronously here instead
    queued = []
    orig = jobs.submit
    jobs.submit = queued.append
    try:
        msg, dd = U.start_job(cid, str(tmp / "promo.mp4"), None, "auto", rows)
    finally:
        jobs.submit = orig
    job_id = dd["value"]
    assert "queued" in msg and queued == [job_id]
    jobs.run_job(job_id)

    q, inputs, md = U.refresh_jobs(job_id)
    assert inputs[0][1] == "done" and "**done**" in md
    table = U.deliverables_table(job_id)
    assert table[0][2] == "tiktok / main" and "pass" in table[0][4]
    video, checks, post, rjson, st = U.show_deliverable(job_id, 1)
    assert os.path.isfile(video) and "Shop now" in post and "#acme" in post
    assert all(c[0] == "✅" for c in checks), checks

    # tweak: move the logo top-right, re-render this file only
    r = json.loads(rjson)
    next(o for o in r["ops"] if o["op"] == "logo")["position"] = "top-right"
    table, video2, checks2, st = U.rerender_one(job_id, 1, json.dumps(r))
    assert "QA pass" in st, (st, checks2)
    assert "re-rendered with its own recipe" in U.show_deliverable(job_id, 1)[4]

    assert U.make_zip(job_id, True)[0] is None            # nothing approved yet
    U.decide(job_id, 1, "approved")
    z, st = U.make_zip(job_id, True)
    assert len(zipfile.ZipFile(z).namelist()) == 3


def test_bad_json_and_no_campaign(env):
    assert "invalid" in U.save_recipe("00000000-0000-0000-0000-000000000000", "{oops", "")[3]
    assert U.check_inputs(None, "", None, "auto")[0] == []
    assert "Pick a campaign" in U.start_job(None, "x", None, "auto", [])[0]


def test_app_builds():
    import gradio as gr
    with gr.Blocks() as app:
        U.build_campaign_tabs()
    assert app is not None
