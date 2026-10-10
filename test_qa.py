# test_qa.py — campaign pipeline C6: compliance checks, manifest, posting text, zip.
# Real renders through the compositor; each check is shown to pass on a correct
# file and to fail on a file that breaks it.
import json
import os
import shutil
import subprocess
import sys
import zipfile

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from campaign import qa as Q  # noqa: E402
from campaign.recipe import validate_recipe  # noqa: E402
from pipeline import compositor as C  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(not FFMPEG, reason="ffmpeg not installed")


def _ff(*args):
    r = subprocess.run([FFMPEG, "-loglevel", "error", "-y", *args], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]


@pytest.fixture(scope="module")
def m(tmp_path_factory):
    d = tmp_path_factory.mktemp("qa")
    _ff("-f", "lavfi", "-i", "testsrc2=size=360x640:rate=30:duration=4",
        "-f", "lavfi", "-i", "sine=frequency=300:duration=4", "-shortest",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(d / "in.mp4"))
    # a track with a rhythm (so its loudness envelope is distinctive)
    _ff("-f", "lavfi", "-i", "sine=frequency=660:duration=6", "-af",
        "volume='if(lt(mod(t,0.5),0.25),1,0.05)':eval=frame", str(d / "beat.wav"))
    _ff("-f", "lavfi", "-i", "color=c=0x2040F0:size=300x120,format=rgba", "-frames:v", "1",
        str(d / "logo.png"))
    cat = [{"name": "logo.png", "kind": "logo", "path": str(d / "logo.png")},
           {"name": "beat.wav", "kind": "audio", "path": str(d / "beat.wav")}]
    return d, cat


def _recipe(cat, ops, export=("tiktok",)):
    v = validate_recipe({"mode": "edit", "ops": ops, "export": list(export)}, cat)
    assert v["ok"], v["errors"]
    return v["recipe"]


def _status(res, cid):
    return next(c for c in res["checks"] if c["id"] == cid)["status"]


class TestChecks:
    def test_good_file_passes_everything(self, m, tmp_path):
        d, cat = m
        rec = _recipe(cat, [{"op": "logo", "asset": "logo.png", "position": "bottom-left"},
                            {"op": "audio.replace", "track": "beat.wav"},
                            {"op": "text_overlay", "text": "Link in bio", "role": "cta"}])
        out = str(tmp_path / "o.mp4")
        rep = C.compose(str(d / "in.mp4"), rec, out, platform="tiktok")
        res = Q.check_file(out, rec, "tiktok", rep, source_path=str(d / "in.mp4"))
        assert res["status"] == "pass", res["checks"]
        assert {c["id"] for c in res["checks"]} >= {"duration", "format", "size", "audio",
                                                    "replaced", "logo0", "text"}

    def test_missing_logo_and_original_audio_fail(self, m, tmp_path):
        d, cat = m
        rec = _recipe(cat, [{"op": "logo", "asset": "logo.png"},
                            {"op": "audio.replace", "track": "beat.wav"}])
        plain = str(tmp_path / "plain.mp4")      # rendered WITHOUT the recipe's edits
        rep = C.compose(str(d / "in.mp4"), _recipe(cat, []), plain, platform="tiktok")
        res = Q.check_file(plain, rec, "tiktok", rep, source_path=str(d / "in.mp4"))
        assert _status(res, "logo0") == "fail" and _status(res, "replaced") == "fail"
        assert res["status"] == "fail"

    def test_wrong_size_and_too_long(self, m, tmp_path):
        d, cat = m
        rec = _recipe(cat, [{"op": "trim", "max_duration": 2}])
        res = Q.check_file(str(d / "in.mp4"), rec, "tiktok", {})
        assert _status(res, "format") == "fail" and _status(res, "duration") == "fail"

    def test_platform_limit(self, m, tmp_path, monkeypatch):
        d, cat = m
        from campaign import presets
        monkeypatch.setitem(presets.PLATFORMS["tiktok"], "max_duration", 3)
        out = str(tmp_path / "o.mp4")
        rep = C.compose(str(d / "in.mp4"), _recipe(cat, []), out, platform="tiktok")
        res = Q.check_file(out, _recipe(cat, []), "tiktok", rep)
        assert _status(res, "duration") == "fail" and "TikTok limit" in \
            next(c for c in res["checks"] if c["id"] == "duration")["detail"]

    def test_loudness_measured(self, m, tmp_path):
        d, _ = m
        lufs = Q.loudness(str(d / "in.mp4"))
        assert lufs is not None and -30 < lufs < 0

    def test_logo_geometry_matches_compositor(self):
        assert Q.logo_geometry(1080, 1920, {"position": "top-right", "size_pct": 12},
                               300, 120) == (1080 - 130 - 32, 32, 130, 52)
        x, y, w, h = Q.logo_geometry(1920, 1080, {"position": "bottom", "size_pct": 10,
                                                  "margin_pct": 0}, 100, 100)
        assert (x, y, w, h) == ((1920 - 192) // 2, 1080 - 192, 192, 192)


class TestDelivery:
    def test_posting_text(self):
        from campaign.delivery import posting_text
        d = {"qa": {"clip": {"title": "He beat the fastest woman", "hashtags": ["mrbeast",
                                                                               "#race"]}}}
        rec = {"ops": [{"op": "text_overlay", "role": "cta", "text": "Link in bio"},
                       {"op": "text_overlay", "role": "hook", "text": "Wait…"}]}
        p = posting_text(d, {"name": "Acme", "brief": "use #AcmeSummer and #race"}, rec)
        assert p["title"] == "He beat the fastest woman"
        assert p["hashtags"] == ["#mrbeast", "#race", "#AcmeSummer"]
        assert p["caption"] == "He beat the fastest woman Link in bio #mrbeast #race #AcmeSummer"

    def test_job_manifest_qa_and_zip(self, m, tmp_path, monkeypatch):
        from campaign import delivery, edit_mode, jobs, service
        from db.repositories import campaign_repo, job_repo
        d, _ = m
        monkeypatch.setattr(service, "ASSET_ROOT", str(tmp_path / "assets"))
        monkeypatch.setattr(edit_mode, "OUTPUT_ROOT", str(tmp_path / "output"))
        monkeypatch.setattr(edit_mode, "WORK_ROOT", str(tmp_path / "work"))
        c = service.create_campaign("QA Test", brief="Post with #acme")
        try:
            service.add_asset(c["id"], str(d / "logo.png"), kind="logo")
            service.save_recipe(c["id"], {"mode": "edit", "export": ["tiktok"],
                                          "ops": [{"op": "logo", "asset": "logo.png"},
                                                  {"op": "text_overlay", "role": "cta",
                                                   "text": "Shop now"}]})
            job = jobs.create_job(c["id"], [str(d / "in.mp4")])
            jobs.run_job(job["id"])
            rows = job_repo.list_deliverables(job["id"])
            assert rows[0]["qa"]["compliance"] == "pass", rows[0]["qa"]["checks"]
            man = json.load(open(os.path.join(os.path.dirname(rows[0]["path"]),
                                              "manifest.json")))
            item = man["deliverables"][0]
            assert item["compliance"] == "pass" and item["posting"]["hashtags"] == ["#acme"]
            assert "Shop now" in item["posting"]["caption"]
            with pytest.raises(ValueError, match="no approved"):
                delivery.build_zip(job["id"], approved_only=True)
            delivery.set_status(rows[0]["id"], "approved")
            z = delivery.build_zip(job["id"], approved_only=True)
            names = zipfile.ZipFile(z).namelist()
            assert sorted(names) == sorted([os.path.basename(rows[0]["path"]), "manifest.json",
                                            "posting.txt"])
        finally:
            campaign_repo.delete_campaign(c["id"])
