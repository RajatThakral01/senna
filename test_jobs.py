# test_jobs.py — campaign pipeline C5: router, jobs (DB), worker, clip-mode
# helpers, CLI. Renders are real ffmpeg (edit mode); the long-video pipeline is
# mocked here and exercised by real runs instead.
import os
import shutil
import subprocess
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from campaign import clip_mode as CM  # noqa: E402
from campaign import jobs as J  # noqa: E402
from pipeline import media  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(not FFMPEG, reason="ffmpeg not installed")


def _ff(*args):
    r = subprocess.run([FFMPEG, "-loglevel", "error", "-y", *args], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]


@pytest.fixture(scope="module")
def m(tmp_path_factory):
    d = tmp_path_factory.mktemp("jobs")
    (d / "batch").mkdir()
    if FFMPEG:
        for name, size in (("a.mp4", "360x640"), ("b.mov", "640x360")):
            _ff("-f", "lavfi", "-i", f"testsrc2=size={size}:rate=30:duration=2",
                "-f", "lavfi", "-i", "sine=frequency=300:duration=2", "-shortest",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(d / "batch" / name))
        _ff("-f", "lavfi", "-i", "color=c=red@0.6:size=300x120,format=rgba", "-frames:v", "1",
            str(d / "logo.png"))
        _ff("-f", "lavfi", "-i", "sine=frequency=440:duration=1", str(d / "song.wav"))
    (d / "batch" / "notes.txt").write_text("not a video")
    return d


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Campaign with a logo + saved recipe; outputs under tmp_path."""
    from campaign import edit_mode, service
    from db.repositories import campaign_repo
    monkeypatch.setattr(service, "ASSET_ROOT", str(tmp_path / "assets"))
    monkeypatch.setattr(edit_mode, "OUTPUT_ROOT", str(tmp_path / "output"))
    monkeypatch.setattr(edit_mode, "WORK_ROOT", str(tmp_path / "work"))
    made = []

    def make(recipe=None, name="Jobs Test"):
        c = service.create_campaign(name)
        made.append(c["id"])
        return c
    yield service, make
    for cid in made:
        campaign_repo.delete_campaign(cid)


class TestRouter:
    def test_precedence(self):
        long_, short = {"duration": 900}, {"duration": 30}
        assert J.route(short, {"mode": "auto"}, "auto", "clip")[0] == "clip"
        assert J.route(long_, {"mode": "clip"}, "edit")[0] == "edit"
        assert J.route(short, {"mode": "clip"}, "auto")[0] == "clip"
        assert J.route(long_, {"mode": "auto"})[0] == "clip"
        assert J.route(short, {"mode": "auto"})[0] == "edit"
        assert J.route({}, {})[0] == "edit"

    def test_expand_folder(self, m):
        out = J.expand_sources([str(m / "batch"), "https://youtu.be/x", " "])
        assert [os.path.basename(x) for x in out[:2]] == ["a.mp4", "b.mov"]
        assert out[2] == "https://youtu.be/x" and len(out) == 3


class TestClipModeHelpers:
    def test_spec_and_overrides(self):
        spec = CM.clip_spec({"clips": {"count": 5, "max_duration": 45, "focus": " money "}})
        assert spec == {"count": 5, "min_duration": None, "max_duration": 45.0, "focus": "money"}
        o = CM.config_overrides(spec)
        assert o[("discovery", "target_clips")] == 7 and o[("refine", "max_duration")] == 45.0
        assert ("refine", "min_duration") not in o

    def test_overridden_restores(self):
        cfg = {"discovery": {"target_clips": 15}}
        with pytest.raises(RuntimeError):
            with CM.overridden(cfg, {("discovery", "target_clips"): 3, ("refine", "x"): 1}):
                assert cfg["discovery"]["target_clips"] == 3 and cfg["refine"]["x"] == 1
                raise RuntimeError
        assert cfg == {"discovery": {"target_clips": 15}, "refine": {}}

    def test_clip_segments_maps_stitched_ranges(self):
        words = [{"word": w, "start": s, "end": s + 0.4} for w, s in
                 (("a", 10.0), ("b", 11.0), ("skip", 15.0), ("c", 30.2), ("late", 40.0))]
        segs = CM.clip_segments([{"words": words}], [[9.5, 12.0], [30.0, 31.0]])
        flat = [(w["word"], round(w["start"], 2)) for s in segs for w in s["words"]]
        assert flat == [("a", 0.5), ("b", 1.5), ("c", 2.7)]

    def test_pick_clips(self):
        clips = [{"clip_number": 1, "refine_status": "rejected", "source_ranges": [[0, 30]]},
                 {"clip_number": 2, "source_ranges": [[0, 30]]},
                 {"clip_number": 3, "source_ranges": [[0, 100]]},
                 {"clip_number": 4, "source_ranges": [[0, 20], [50, 70]]},
                 {"clip_number": 5, "start_time": 0, "end_time": 25}]
        got = CM.pick_clips(clips, {"count": 2, "min_duration": None, "max_duration": 60,
                                    "focus": ""})
        assert [c["clip_number"] for c in got] == [2, 4]

    def test_platform_groups(self):
        v, o = CM.platform_groups({"export": ["tiktok", "youtube", "reels"], "ops": []})
        assert v == ["tiktok", "reels"] and o == ["youtube"]
        v, o = CM.platform_groups({"export": ["tiktok"],
                                   "ops": [{"op": "reframe", "aspect": "9:16",
                                            "method": "fit_blur"}]})
        assert v == [] and o == ["tiktok"]

    def test_render_flags(self):
        f = CM.render_flags({"ops": [{"op": "reframe", "method": "center"}]})
        assert f["subtitles"] is False and f["logo"] is None and f["layout"] == "center_crop"

    @needs_ffmpeg
    def test_cut_ranges_joins(self, m, tmp_path):
        out = CM.cut_ranges(str(m / "batch" / "b.mov"), [[0.2, 0.7], [1.0, 1.6]],
                            str(tmp_path / "cut.mp4"))
        p = media.probe(out)
        assert p["has_audio"] and p["width"] == 640 and abs(p["duration"] - 1.1) < 0.1


@needs_ffmpeg
class TestJobs:
    def _ready(self, service, make, m, ops=None, export=("tiktok",)):
        c = make()
        service.add_asset(c["id"], str(m / "logo.png"), kind="logo")
        r = service.save_recipe(c["id"], {"mode": "auto", "export": list(export),
                                          "ops": ops or [{"op": "logo", "asset": "logo.png"}]})
        assert r["validation"]["ok"] and r["campaign"]["status"] == "ready"
        return r["campaign"]

    def test_needs_ready_recipe(self, env, m):
        service, make = env
        c = make()
        with pytest.raises(ValueError, match="no recipe"):
            J.create_job(c["id"], [str(m / "batch" / "a.mp4")])
        r = service.save_recipe(c["id"], {"ops": [{"op": "logo", "asset": "missing.png"}]})
        assert r["campaign"]["status"] == "draft"
        with pytest.raises(ValueError, match="not ready"):
            J.create_job(c["id"], [str(m / "batch" / "a.mp4")])

    def test_run_isolates_failures_and_resumes(self, env, m, monkeypatch):
        from db.repositories import job_repo
        service, make = env
        c = self._ready(service, make, m)
        job = J.create_job(c["id"], [str(m / "batch"), str(m / "song.wav"),
                                     str(m / "nope.mp4")])
        seen = []
        res = J.run_job(job["id"], progress=lambda f, msg: seen.append((f, msg)))
        st = J.status(job["id"])
        assert res["status"] == "done" and res["finished_at"]
        assert [i["status"] for i in st["inputs"]] == ["done", "done", "failed", "failed"]
        assert [i["mode"] for i in st["inputs"][:2]] == ["edit", "edit"]
        assert "no video stream" in st["inputs"][2]["error"]
        assert "song.wav" in res["error"] and "nope.mp4" in res["error"]
        assert len(st["deliverables"]) == 2 and seen[-1][0] == 1.0
        assert st["deliverables"][0]["path"].endswith("_a_01_tiktok.mp4")
        # resume: done inputs are skipped, failed ones retried
        calls = []
        from campaign import edit_mode
        monkeypatch.setattr(edit_mode, "process_input",
                            lambda *a, **k: calls.append(a[2]["idx"]) or [])
        J.run_job(job["id"])
        assert calls == [] and len(job_repo.list_deliverables(job["id"])) == 2

    def test_router_sends_long_input_to_clip_mode(self, env, m, monkeypatch):
        service, make = env
        c = self._ready(service, make, m)
        job = J.create_job(c["id"], [str(m / "batch" / "a.mp4"), str(m / "batch" / "b.mov")],
                           modes={1: "edit"})
        long_probe = dict(media.probe(str(m / "batch" / "a.mp4")), duration=1200.0)
        monkeypatch.setattr(J, "_resolve", lambda inp: (inp["source"], long_probe))
        ran = []
        from campaign import edit_mode
        monkeypatch.setattr(CM, "process_input", lambda *a, **k: ran.append(("clip", a[3])) or [])
        monkeypatch.setattr(edit_mode, "process_input",
                            lambda *a, **k: ran.append(("edit", a[3])) or [])
        J.run_job(job["id"])
        assert ran == [("clip", "a"), ("edit", "b")]

    def test_worker_runs_and_cancels(self, env, m):
        service, make = env
        c = self._ready(service, make, m)
        job = J.create_job(c["id"], [str(m / "batch" / "a.mp4")])
        w = J.Worker()
        w.submit(job["id"])
        for _ in range(120):
            if J.status(job["id"])["job"]["status"] in ("done", "failed"):
                break
            time.sleep(0.5)
        assert J.status(job["id"])["job"]["status"] == "done"
        j2 = J.create_job(c["id"], [str(m / "batch" / "b.mov")])
        w2 = J.Worker()
        w2._ensure = lambda: None          # keep it queued
        w2.submit(j2["id"])
        w2.cancel(j2["id"])
        assert J.status(j2["id"])["job"]["status"] == "cancelled"

    def test_cli_create_and_run(self, env, m, capsys):
        from campaign import cli
        from db.repositories import campaign_repo
        service, _ = env
        cli.main(["campaign", "create", "--name", "CLI Test Campaign", "--no-llm",
                  "--asset", str(m / "logo.png"),
                  "--brief", "add the logo top left, export for TikTok and YouTube"])
        out = capsys.readouterr().out
        c = next(x for x in service.list_campaigns() if x["name"] == "CLI Test Campaign")
        try:
            assert "Logo logo.png top-left" in out and "Status: ready" in out
            cli.main(["job", "run", "--campaign", c["slug"], "--input",
                      str(m / "batch" / "a.mp4")])
            out = capsys.readouterr().out
            assert "1/1 inputs done, 2 files" in out and "youtube" in out
        finally:
            campaign_repo.delete_campaign(c["id"])
