# test_progress_ui.py — unit tests (no DB) for the 16 Oct UI rework: live progress card,
# step checklist, activity feed (per-thread capture + plain-language lines), status
# banner, auto-picking the running job, review counts, approve-all-passing.
import logging
import os
import sys
import threading
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(__file__))

import campaign_ui as U  # noqa: E402
from campaign import activity as A  # noqa: E402

def _job(status="running", msg="[1/2] talk.mp4: [8/11] Analyzing for clips...", prog=0.3):
    now = datetime.now(timezone.utc)          # per call: the suite can run for minutes
    return {"id": "6641082f-aaaa", "campaign_id": "c1", "status": status, "progress": prog,
            "message": msg, "started_at": now - timedelta(minutes=12, seconds=5),
            "finished_at": None if status == "running" else now, "error": None}


INPUTS = [{"idx": 0, "status": "running", "mode": "clip", "source": "talk.mp4", "error": None},
          {"idx": 1, "status": "queued", "mode": None, "source": "song.mp4", "error": None}]


def _patch_job(job, inputs=INPUTS, deliverables=(), quiet=5.0):
    return [patch("db.repositories.job_repo.get_job", return_value=job),
            patch("db.repositories.job_repo.list_inputs", return_value=inputs),
            patch("db.repositories.job_repo.list_deliverables", return_value=list(deliverables)),
            patch("campaign.service.get_campaign", return_value={"name": "u2"}),
            patch("campaign.activity.seconds_since_activity", return_value=quiet)]


def _run(patches, fn, *a):
    for p in patches:
        p.start()
    try:
        return fn(*a)
    finally:
        for p in patches:
            p.stop()


class TestMessageParsing:
    def test_parse(self):
        assert U._parse_message("[1/2] talk.mp4: [8/11] Analyzing for clips...") == \
            (1, 2, "talk.mp4", "Analyzing for clips")
        assert U._parse_message("[2/2] a b.mp4: composing clip 3/13") == \
            (2, 2, "a b.mp4", "composing clip 3/13")
        assert U._parse_message("starting") == (None, None, None, "starting")

    def test_step_index(self):
        assert U._step_index("Transcribing", U.CLIP_STEPS) == 1
        assert U._step_index("framing clip 4/13", U.CLIP_STEPS) == 5
        assert U._step_index("composing clip 2/13", U.CLIP_STEPS) == 6
        assert U._step_index("rendering", U.EDIT_STEPS) == 2
        assert U._step_index("something new", U.CLIP_STEPS) is None


class TestProgressCard:
    def test_running_card_shows_step_bar_and_reassurance(self):
        html = _run(_patch_job(_job(), quiet=300), U.progress_card, "6641082f-aaaa")
        assert "u2" in html and "Running" in html and "30% done" in html
        assert "⏳ Find the best moments (AI)" in html          # current step
        assert "✅ Transcribe speech" in html                    # earlier step ticked
        assert "○ Frame each clip for vertical" in html          # later step pending
        assert "Video 1 of 2" in html and "talk.mp4" in html
        assert "Still working" in html and "5 min ago" in html   # long quiet step
        assert "Running for 12 min" in html

    def test_active_step_shows_clip_counter(self):
        html = _run(_patch_job(_job(msg="[1/2] talk.mp4: framing clip 4/13", prog=0.55)),
                    U.progress_card, "j")
        assert "⏳ Frame each clip for vertical" in html and "framing clip 4/13" in html

    def test_done_card_points_to_review(self):
        html = _run(_patch_job(_job("done", "2/2 inputs done, 42 files", 1.0),
                               inputs=[dict(INPUTS[0], status="done"),
                                       dict(INPUTS[1], status="done")],
                               deliverables=[{}] * 42), U.progress_card, "j")
        assert "Finished" in html and "100% done" in html and "42 file(s) ready" in html
        assert "Review" in html and "⏳" not in html and "Last run took" in html

    def test_failed_video_is_explained(self):
        inputs = [dict(INPUTS[0], status="done"),
                  dict(INPUTS[1], status="failed", error="could not convert string")]
        html = _run(_patch_job(_job("done", "1/2 inputs done", 1.0), inputs=inputs),
                    U.progress_card, "j")
        assert "Video 2 failed" in html and "Resume" in html

    def test_reexport_uses_its_own_steps_and_running_time(self):
        j = _job(msg="re-exporting 3/14", prog=0.2)
        j["finished_at"] = j["started_at"] - timedelta(hours=3)   # stale from the first run
        html = _run(_patch_job(j), U.progress_card, "j")
        assert "⏳ Re-export every clip" in html and "re-exporting 3/14" in html
        assert "Running for 12 min" in html and "-" not in html.split("Running for")[1][:6]

    def test_no_job(self):
        assert "Pick a job" in U.progress_card(None)

    def test_message_is_html_escaped(self):
        html = _run(_patch_job(_job(msg="[1/1] <b>x</b>.mp4: Transcribing")),
                    U.progress_card, "j")
        assert "<b>x</b>" not in html and "&lt;b&gt;" in html


class TestBannerAndAutoPick:
    def test_banner_idle(self):
        with patch("db.repositories.job_repo.list_jobs", return_value=[]):
            assert "No jobs running" in U.status_banner()

    def test_banner_running(self):
        def jobs(status=None, limit=5):
            return [_job(msg="[1/2] talk.mp4: composing clip 2/13", prog=0.6)] \
                if status == "running" else []
        with patch("db.repositories.job_repo.list_jobs", side_effect=jobs), \
                patch("campaign.service.list_campaigns",
                      return_value=[{"id": "c1", "name": "u2"}]), \
                patch("campaign.activity.seconds_since_activity", return_value=8):
            html = U.status_banner()
        assert "u2" in html and "60%" in html and "composing clip 2/13" in html
        assert "active 8 s ago" in html and "3 · Progress" in html

    def test_refresh_picks_running_job_when_none_selected(self):
        with patch.object(U, "latest_active_job", return_value="J1"), \
                patch.object(U, "job_choices", return_value=[("J1", "J1")]), \
                patch.object(U, "job_detail", return_value=([[0, "running"]], "")), \
                patch.object(U, "progress_card", return_value="CARD"), \
                patch.object(U, "activity_feed", return_value="FEED"), \
                patch.object(U, "jobs_table", return_value=[]):
            pick, card, rows, feed, table = U.refresh_progress(None)
            assert pick["value"] == "J1" and card == "CARD" and feed == "FEED"
            pick2 = U.refresh_progress("J1")[0]
        assert "value" not in pick2                              # user's choice kept


class TestReview:
    ROWS = [{"id": "a", "status": "rendered", "qa": {"compliance": "pass"}},
            {"id": "b", "status": "rendered", "qa": {"compliance": "fail"}},
            {"id": "c", "status": "approved", "qa": {"compliance": "pass"}},
            {"id": "d", "status": "rejected", "qa": {"compliance": "pass"}}]

    def test_counts(self):
        with patch("db.repositories.job_repo.list_deliverables", return_value=self.ROWS):
            md = U.review_counts("j")
        assert "4 files" in md and "2 to review" in md and "1 approved" in md
        assert "1 rejected" in md and "1 failed QA" in md

    def test_approve_all_passing_only_touches_unreviewed_passes(self):
        with patch("db.repositories.job_repo.list_deliverables", return_value=self.ROWS), \
                patch.object(U, "deliverables_table", return_value=[]), \
                patch("campaign.delivery.set_status") as st:
            _, msg = U.approve_all_passing("j")
        st.assert_called_once_with("a", "approved")
        assert "Approved 1" in msg


class TestActivity:
    def test_friendly_lines(self):
        assert A.friendly("main", "▶ START stage=transcribe") == "Started: transcribing speech"
        assert A.friendly("main", "✔ DONE stage=render elapsed=1560.2s") == \
            "Done: framing clips (26.0 min)"
        assert A.friendly("main", "stage skipped (already done)") is None
        assert A.friendly("pipeline.compositor",
                          "compose clip_1_final.mp4 → u2_x_01_tiktok.mp4 platform=tiktok "
                          "1080x1920 dur=28.2s framing=scale ops=[]") == \
            "Exported u2_x_01_tiktok.mp4 for tiktok (28 s)"

    def test_capture_keeps_only_the_job_thread(self, tmp_path, monkeypatch):
        monkeypatch.setattr(A, "ROOT", str(tmp_path))
        log = logging.getLogger("pipeline.test_activity")
        log.setLevel(logging.INFO)
        with A.capture("job1"):
            log.info("▶ START stage=outline")
            t = threading.Thread(target=lambda: log.info("other thread line"))
            t.start()
            t.join()
            logging.getLogger("db.run_repo").info("noise")
            log.warning("⚠ careful")
        log.info("after capture")
        lines = A.tail("job1")
        assert len(lines) == 2
        assert lines[0].endswith("Started: understanding the video (AI outline)")
        assert "⚠️" in lines[1]
        assert A.seconds_since_activity("job1") < 5
        A.note("job1", "▸ manual")
        assert A.tail("job1")[-1].endswith("▸ manual")
        assert A.tail("nope") == [] and A.seconds_since_activity("nope") is None

    def test_feed_newest_first(self):
        with patch("campaign.activity.tail", return_value=["10:00 a", "10:01 b"]):
            assert U.activity_feed("j") == "10:01 b\n10:00 a"


def test_app_builds_with_tabs():
    import gradio as gr
    import ui
    ids = [b.id for b in ui.app.blocks.values() if isinstance(b, gr.Tab)]
    labels = [b.label for b in ui.app.blocks.values() if isinstance(b, gr.Tab)]
    assert labels[:4] == ["1 · Set up campaign", "2 · Start a job", "3 · Progress",
                          "4 · Review & download"]
    assert "progress" in ids and "review" in ids
