# test_ui_e2e.py — browser automation (Playwright/Chromium) against the RUNNING app.
#
# Seeds its own campaign + job + 3 small rendered files inside the app container, then
# clicks through the real UI: select a row by clicking the table, Approve / Reject with
# auto-advance, ◀ / ▶ navigation, statuses persisted in the DB, new campaigns showing up
# in the New-job dropdown without a reload, the campaign summary, the Clip-review picker.
# Cleans up after itself.
#
# Run from the host (not inside the container):
#   pip install playwright && python -m playwright install chromium
#   E2E_URL=http://localhost:7862 pytest -q test_ui_e2e.py
# Skips when Playwright, docker or the app is not available.
import json
import os
import shutil
import subprocess
import time
import urllib.request

import pytest

URL = os.getenv("E2E_URL", "http://localhost:7862")
CONTAINER = os.getenv("E2E_CONTAINER", "senna-dev-app-1")

pw = pytest.importorskip("playwright.sync_api")


def _app_up():
    try:
        return urllib.request.urlopen(URL, timeout=5).status == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not (shutil.which("docker") and _app_up()),
                                reason=f"needs docker and the app running at {URL}")

SEED = r'''
import json, os, subprocess, uuid
from campaign import service, jobs
from db.repositories import job_repo
tag = "E2E " + uuid.uuid4().hex[:6]
c = service.create_campaign(tag, "Add the text Hello for the first second. TikTok.", ["tiktok"])
service.save_recipe(c["id"], {"mode": "edit", "export": ["tiktok"],
    "ops": [{"op": "text_overlay", "text": "Hello", "role": "hook", "start": 0, "end": 1}]}, {})
job = job_repo.insert_job(c["id"], {"ops": []}, 1, "auto")
inp = job_repo.add_input(job["id"], 0, "/tmp/e2e.mp4", "edit")
os.makedirs("output/e2e", exist_ok=True)
ids = []
for i in range(3):
    p = f"output/e2e/{tag.replace(' ', '_')}_{i + 1}.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    f"testsrc2=size=360x640:rate=30:duration=1", "-pix_fmt", "yuv420p", p],
                   check=True)
    d = job_repo.add_deliverable(job["id"], inp["id"], variant="main", platform="tiktok",
                                 path=p, recipe_version=1)
    job_repo.update_deliverable(d["id"], status="rendered",
                                qa={"compliance": "pass", "duration": 1.0, "checks": []})
    ids.append(d["id"])
job_repo.update_job(job["id"], status="done", progress=1.0)
print(json.dumps({"campaign": c["id"], "name": tag, "job": job["id"], "deliverables": ids}))
'''

CLEAN = r'''
import glob, os, sys
from db.repositories import campaign_repo
from db.connection import get_conn, release_conn
cid, job = sys.argv[1], sys.argv[2]
conn = get_conn()
with conn.cursor() as cur:
    jobs = "SELECT id FROM jobs WHERE campaign_id = %s"
    cur.execute(f"DELETE FROM deliverables WHERE job_id IN ({jobs})", (cid,))
    cur.execute(f"DELETE FROM job_inputs WHERE job_id IN ({jobs})", (cid,))
    cur.execute("DELETE FROM jobs WHERE campaign_id = %s", (cid,))
    conn.commit()
release_conn(conn)
campaign_repo.delete_campaign(cid)
for p in glob.glob("output/e2e/E2E_*"):
    os.remove(p)
'''


def _py(code, *args):
    r = subprocess.run(["docker", "exec", "-i", CONTAINER, "python", "-", *args],
                       input=code, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-1500:]
    return r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""


def _statuses(deliverable_ids):
    code = ("import json,sys\nfrom db.repositories import job_repo\n"
            "print(json.dumps([job_repo.get_deliverable(i)['status'] for i in sys.argv[1:]]))")
    return json.loads(_py(code, *deliverable_ids))


@pytest.fixture(scope="module")
def seed():
    s = json.loads(_py(SEED))
    yield s
    _py(CLEAN, s["campaign"], s["job"])


@pytest.fixture(scope="module")
def page():
    with pw.sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1400, "height": 1600})
        pg.goto(URL, wait_until="networkidle")
        yield pg
        b.close()


def _visible(loc):
    return [loc.nth(i) for i in range(loc.count()) if loc.nth(i).is_visible()][0]


def _pick(page, label, starts_with):
    _visible(page.get_by_label(label)).click()
    page.get_by_role("option").filter(has_text=starts_with).first.click()


def _cell(page, seed, n):
    """The visible table cell holding deliverable n's file name."""
    name = f"{seed['name'].replace(' ', '_')}_{n}.mp4"
    page.get_by_text(name).first.wait_for(state="attached")
    return _visible(page.get_by_text(name, exact=True))


def _row_no(page):
    return int(float(_visible(page.get_by_label("Row #")).input_value()))


def _wait_row(page, n, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        if _row_no(page) == n:
            return
        time.sleep(0.2)
    assert _row_no(page) == n


def test_review_select_approve_reject_navigate(page, seed):
    page.get_by_role("tab", name="4 · Review & download").click()
    _pick(page, "Job", seed["job"][:8])          # list refreshed on tab open: new job is there
    page.get_by_text(f"{seed['name'].replace(' ', '_')}_3.mp4").first.wait_for()

    # clicking a table row selects it
    _cell(page, seed, 3).click()
    _wait_row(page, 3)
    page.get_by_text("#3 of 3").first.wait_for()

    # approve the last one: stays on it, says so
    page.get_by_role("button", name="✅ Approve", exact=True).click()
    page.get_by_text("that was the last file").first.wait_for()
    assert _statuses(seed["deliverables"]) == ["rendered", "rendered", "approved"]

    # ◀ Prev → 2, reject → auto-advance to 3
    page.get_by_role("button", name="◀ Prev", exact=True).click()
    _wait_row(page, 2)
    page.get_by_role("button", name="❌ Reject", exact=True).click()
    page.get_by_text("#2 **rejected**").or_(page.get_by_text("#2 rejected")).first.wait_for()
    _wait_row(page, 3)

    # click row 1, approve → moves to 2; Next ▶ → 3
    _cell(page, seed, 1).click()
    _wait_row(page, 1)
    page.get_by_role("button", name="✅ Approve", exact=True).click()
    _wait_row(page, 2)
    page.get_by_role("button", name="Next ▶", exact=True).click()
    _wait_row(page, 3)
    assert _statuses(seed["deliverables"]) == ["approved", "rejected", "approved"]


def test_new_campaign_visible_in_new_job_tab(page, seed):
    # the page was loaded BEFORE this campaign existed; opening the tab must list it
    page.get_by_role("tab", name="2 · Start a job").click()
    _pick(page, "Campaign", seed["name"])
    page.get_by_text(f"{seed['name']} · recipe v").first.wait_for()
    page.get_by_text("Ready to run").first.wait_for()


RUNNING = r'''
import json, sys
from campaign import activity
from db.repositories import job_repo
cid = sys.argv[1]
job = job_repo.insert_job(cid, {"ops": []}, 1, "auto")
job_repo.add_input(job["id"], 0, "/tmp/talk_show.mp4", "clip")
job_repo.update_job(job["id"], status="running", started_at=True, progress=0.55,
                    message="[1/1] talk_show.mp4: framing clip 4/13")
activity.note(job["id"], "Tracking faces in a clip…")
print(json.dumps({"job": job["id"]}))
'''


def test_running_job_is_visible_everywhere(page, seed):
    # a job in a long step: banner on every tab + Progress card + live feed
    j = json.loads(_py(RUNNING, seed["campaign"]))["job"]
    try:
        page.reload(wait_until="networkidle")
        page.get_by_text("framing clip 4/13").first.wait_for(timeout=15000)     # banner
        page.get_by_role("tab", name="3 · Progress").click()
        _pick(page, "Job", j[:8])
        page.get_by_text("⏳ Frame each clip for vertical").first.wait_for(timeout=15000)
        page.get_by_text("55% done").first.wait_for()
        feed = _visible(page.get_by_label("What's happening (newest first)"))
        end = time.time() + 15
        while time.time() < end and "Tracking faces" not in feed.input_value():
            time.sleep(0.5)
        assert "Tracking faces" in feed.input_value()
    finally:
        _py("import sys\nfrom db.repositories import job_repo\n"
            "job_repo.update_job(sys.argv[1], status='cancelled', finished_at=True)", j)


def test_clip_review_has_video_picker(page):
    page.get_by_role("tab", name="Fix a clip (advanced)").click()
    _visible(page.get_by_label("Pick a recent video (fills the ID)")).click()
    opts = page.get_by_role("option")
    if opts.count() == 0:
        pytest.skip("no videos in the database yet")
    opts.first.click()
    vid = _visible(page.get_by_label("Video ID (UUID)"))
    end = time.time() + 10
    while time.time() < end and len(vid.input_value()) != 36:
        time.sleep(0.2)
    assert len(vid.input_value()) == 36
