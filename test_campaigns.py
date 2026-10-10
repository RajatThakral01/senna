# test_campaigns.py — campaign pipeline C1: media probe, campaigns + asset
# library, jobs / inputs / deliverables (needs the Docker DB + ffmpeg).
import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from pipeline import media  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(not FFMPEG, reason="ffmpeg not installed")


def _ff(*args):
    r = subprocess.run([FFMPEG, "-loglevel", "error", "-y", *args], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]


@pytest.fixture(scope="module")
def mediadir(tmp_path_factory):
    d = tmp_path_factory.mktemp("media")
    if FFMPEG:
        _ff("-f", "lavfi", "-i", "testsrc=size=1920x1080:rate=25:duration=2",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            str(d / "clip.mp4"))
        _ff("-f", "lavfi", "-i", "testsrc=size=1080x1920:rate=30:duration=1", "-an",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(d / "vertical.mp4"))
        _ff("-f", "lavfi", "-i", "sine=frequency=220:duration=3", str(d / "track.wav"))
        _ff("-f", "lavfi", "-i", "color=c=red@0.5:size=200x80,format=rgba", "-frames:v", "1",
            str(d / "brand.png"))
        _ff("-f", "lavfi", "-i", "color=c=blue:size=640x480", "-frames:v", "1",
            str(d / "photo.jpg"))
    (d / "Brand.ttf").write_bytes(b"\x00\x01\x00\x00fake font")
    return d


class TestMediaProbe:
    @needs_ffmpeg
    def test_video(self, mediadir):
        p = media.probe(str(mediadir / "clip.mp4"))
        assert p["kind"] == "video" and p["has_video"] and p["has_audio"]
        assert (p["width"], p["height"]) == (1920, 1080) and abs(p["duration"] - 2) < 0.2
        assert media.aspect_label(p["width"], p["height"]) == "16:9"

    @needs_ffmpeg
    def test_vertical_without_audio(self, mediadir):
        p = media.probe(str(mediadir / "vertical.mp4"))
        assert p["has_video"] and not p["has_audio"]
        assert media.aspect_label(p["width"], p["height"]) == "9:16"

    @needs_ffmpeg
    def test_audio_and_images(self, mediadir):
        a = media.probe(str(mediadir / "track.wav"))
        assert a["kind"] == "audio" and a["has_audio"] and not a["has_video"]
        logo = media.probe(str(mediadir / "brand.png"))
        assert logo["kind"] == "image" and logo["alpha"] is True
        assert media.probe(str(mediadir / "photo.jpg"))["alpha"] is False

    def test_font_and_missing(self, mediadir):
        assert media.probe(str(mediadir / "Brand.ttf"))["kind"] == "font"
        missing = media.probe(str(mediadir / "nope.mp4"))
        assert missing["exists"] is False and missing["duration"] == 0.0

    def test_aspect_labels(self):
        assert media.aspect_label(1080, 1350) == "4:5"
        assert media.aspect_label(1000, 1000) == "1:1"
        assert media.aspect_label(1000, 700) == "1000x700"


@pytest.fixture
def campaign_env(tmp_path, monkeypatch):
    """A real campaign in the DB with its asset folder under tmp_path."""
    from campaign import service
    monkeypatch.setattr(service, "ASSET_ROOT", str(tmp_path / "campaigns"))
    created = []

    def make(name="Test Campaign", **kw):
        c = service.create_campaign(name, **kw)
        created.append(c["id"])
        return c
    yield service, make
    from db.repositories import campaign_repo
    for cid in created:
        campaign_repo.delete_campaign(cid)


class TestCampaignService:
    def test_slugs_are_unique(self, campaign_env):
        service, make = campaign_env
        a = make("Acme Summer Launch!")
        b = make("Acme Summer Launch!")
        assert a["slug"] == "acme-summer-launch" and b["slug"] == "acme-summer-launch-2"
        with pytest.raises(ValueError):
            make("   ")

    @needs_ffmpeg
    def test_assets_copied_probed_and_kinds_inferred(self, campaign_env, mediadir):
        service, make = campaign_env
        c = make("Assets")
        kinds = {}
        for f in ("brand.png", "photo.jpg", "track.wav", "clip.mp4", "Brand.ttf"):
            a = service.add_asset(c["id"], str(mediadir / f))
            kinds[f] = a["kind"]
            assert os.path.isfile(a["path"]) and a["path"].startswith(service.campaign_dir(c))
        assert kinds == {"brand.png": "logo", "photo.jpg": "image", "track.wav": "audio",
                         "clip.mp4": "video", "Brand.ttf": "font"}
        cat = {x["name"]: x for x in service.asset_catalog(c["id"])}
        assert cat["track.wav"]["duration"] == pytest.approx(3.0, abs=0.2)
        assert cat["brand.png"]["size"] == "200x80"

    @needs_ffmpeg
    def test_readding_replaces_and_remove_deletes_file(self, campaign_env, mediadir):
        service, make = campaign_env
        c = make("Replace")
        a1 = service.add_asset(c["id"], str(mediadir / "brand.png"))
        a2 = service.add_asset(c["id"], str(mediadir / "brand.png"))
        assert a1["id"] == a2["id"] and len(service.list_assets(c["id"])) == 1
        service.remove_asset(a2["id"])
        assert not os.path.exists(a2["path"]) and service.list_assets(c["id"]) == []

    def test_unsupported_asset_rejected(self, campaign_env, tmp_path):
        service, make = campaign_env
        c = make("Bad")
        f = tmp_path / "notes.txt"
        f.write_text("hi")
        with pytest.raises(ValueError):
            service.add_asset(c["id"], str(f))

    def test_recipe_versioning(self, campaign_env):
        from db.repositories import campaign_repo
        service, make = campaign_env
        c = make("Recipe", brief="add logo", platforms=["tiktok"])
        assert c["recipe_version"] == 0 and c["platforms"] == ["tiktok"]
        r1 = campaign_repo.save_recipe(c["id"], {"version": 1, "ops": []}, {"summary": "x"})
        r2 = campaign_repo.save_recipe(c["id"], {"version": 1, "ops": [{"op": "trim"}]})
        assert (r1["recipe_version"], r2["recipe_version"]) == (1, 2)
        assert r2["recipe"]["ops"] == [{"op": "trim"}] and r2["requirements"] == {"summary": "x"}
        upd = service.update_campaign(c["id"], brief="new brief", status="ready")
        assert upd["brief"] == "new brief" and upd["status"] == "ready"
        with pytest.raises(ValueError):
            service.update_campaign(c["id"], slug="hack")


class TestJobsRepo:
    def test_job_inputs_deliverables_roundtrip(self, campaign_env):
        from db.repositories import job_repo
        service, make = campaign_env
        c = make("Jobs")
        j = job_repo.insert_job(c["id"], {"version": 1, "ops": []}, 3, mode_policy="edit")
        assert j["status"] == "queued" and j["recipe_version"] == 3
        i0 = job_repo.add_input(j["id"], 0, "/tmp/a.mp4")
        job_repo.add_input(j["id"], 1, "https://youtu.be/x", mode="clip")
        job_repo.update_input(i0["id"], probe={"duration": 12.5}, mode="edit", status="done")
        ins = job_repo.list_inputs(j["id"])
        assert [x["idx"] for x in ins] == [0, 1] and ins[0]["probe"] == {"duration": 12.5}
        d = job_repo.add_deliverable(j["id"], i0["id"], platform="tiktok", recipe_version=3)
        job_repo.update_deliverable(d["id"], status="rendered", qa={"duration_ok": True},
                                    path="/tmp/out.mp4")
        ds = job_repo.list_deliverables(j["id"])
        assert ds[0]["status"] == "rendered" and ds[0]["qa"] == {"duration_ok": True}
        job_repo.update_job(j["id"], status="running", progress=0.5, started_at=True)
        jj = job_repo.get_job(j["id"])
        assert jj["status"] == "running" and jj["started_at"] is not None and jj["finished_at"] is None
        assert job_repo.list_jobs(campaign_id=c["id"])[0]["id"] == j["id"]
