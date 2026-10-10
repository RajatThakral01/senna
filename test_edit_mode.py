# test_edit_mode.py — campaign pipeline C4: supplied videos → deliverables.
# Real ffmpeg renders; Whisper is mocked. The DB test needs the Docker DB.
import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from campaign import edit_mode as E  # noqa: E402
from campaign.recipe import validate_recipe  # noqa: E402
from pipeline import media  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(not FFMPEG, reason="ffmpeg not installed")

WORDS = [{"word": w, "start": 0.2 + i * 0.4, "end": 0.55 + i * 0.4}
         for i, w in enumerate("buy the new summer drink".split())]
SEGS = [{"start": 0.2, "end": WORDS[-1]["end"], "words": WORDS, "text": ""}]


def _ff(*args):
    r = subprocess.run([FFMPEG, "-loglevel", "error", "-y", *args], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]


@pytest.fixture(scope="module")
def m(tmp_path_factory):
    d = tmp_path_factory.mktemp("edit")
    _ff("-f", "lavfi", "-i", "testsrc2=size=360x640:rate=30:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=300:duration=3", "-shortest",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(d / "promo.mp4"))
    (d / "b").mkdir()
    _ff("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=2", "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(d / "b" / "promo.mp4"))
    _ff("-f", "lavfi", "-i", "sine=frequency=660:duration=4", str(d / "track.wav"))
    _ff("-f", "lavfi", "-i", "color=c=red@0.6:size=300x120,format=rgba", "-frames:v", "1",
        str(d / "logo.png"))
    _ff("-f", "lavfi", "-i", "sine=frequency=440:duration=1", str(d / "notvideo.wav"))
    cat = [{"name": "logo.png", "kind": "logo", "path": str(d / "logo.png")},
           {"name": "track.wav", "kind": "audio", "path": str(d / "track.wav")}]
    return d, cat


def _recipe(cat, ops, export=("tiktok",), **extra):
    v = validate_recipe(dict({"mode": "edit", "ops": ops, "export": list(export)}, **extra), cat)
    assert v["ok"], v["errors"]
    return v["recipe"]


class TestNaming:
    def test_unique_labels(self):
        assert E.input_labels(["a/promo.mp4", "b/promo.mp4", "https://x.com/v/clip.mov?t=1",
                               "c/My Video!.mp4"]) == ["promo", "promo_2", "clip", "My_Video"]

    def test_basename_pattern(self):
        assert E.output_basename("{campaign}_{input}_{clip}_{platform}", "acme", "promo") == \
            "acme_promo_01_{platform}"
        assert E.output_basename("{n}-{input}", "acme", "promo", n=3) == "03-promo_{platform}"

    def test_job_dirs(self):
        out, work = E.job_dirs("acme", "1234567890ab", 2)
        assert out.endswith(os.path.join("acme", "12345678"))
        assert work.endswith(os.path.join("acme", "12345678", "002"))


class TestTranscriptNeed:
    def test_rules(self, m):
        _, cat = m
        assert not E._needs_transcript(_recipe(cat, []))
        assert E._needs_transcript(_recipe(cat, [{"op": "captions"}]))
        assert not E._needs_transcript(_recipe(cat, [{"op": "captions", "enabled": False}]))
        # replacement voice: the compositor captions the track instead
        assert not E._needs_transcript(_recipe(cat, [{"op": "captions"},
                                                     {"op": "audio.replace", "track": "track.wav"}]))


class TestRenderInput:
    def test_logo_text_two_platforms(self, m, tmp_path):
        d, cat = m
        rec = _recipe(cat, [{"op": "logo", "asset": "logo.png"},
                            {"op": "text_overlay", "text": "Link in bio", "role": "cta",
                             "from_end": 1}], export=("tiktok", "reels", "youtube"))
        base = E.output_basename(rec["output"]["filename"], "acme", "promo")
        r = E.render_input(str(d / "promo.mp4"), rec, str(tmp_path / "out"),
                           str(tmp_path / "work"), base)
        names = sorted(os.path.basename(x["output_path"]) for x in r["results"])
        assert names == ["acme_promo_01_reels.mp4", "acme_promo_01_tiktok.mp4",
                         "acme_promo_01_youtube.mp4"]
        yt = media.probe(str(tmp_path / "out" / "acme_promo_01_youtube.mp4"))
        assert (yt["width"], yt["height"]) == (1920, 1080)
        assert {"logo", "text_overlay"} <= set(r["results"][0]["applied"])

    def test_variants_render_side_by_side(self, m, tmp_path):
        d, cat = m
        rec = _recipe(cat, [{"op": "text_overlay", "text": "Hook A", "role": "hook"},
                            {"op": "logo", "asset": "logo.png"}],
                      variants=[{"name": "b", "ops": [{"op": "text_overlay", "text": "Hook B",
                                                       "role": "hook"}]},
                                {"name": "nologo", "remove": ["logo"]}])
        base = E.output_basename(rec["output"]["filename"], "acme", "promo")
        r = E.render_input(str(d / "promo.mp4"), rec, str(tmp_path / "o"), str(tmp_path / "w"),
                           base)
        got = {(x["variant"], os.path.basename(x["output_path"])) for x in r["results"]}
        assert got == {("main", "acme_promo_01_tiktok.mp4"), ("b", "acme_promo_01_b_tiktok.mp4"),
                       ("nologo", "acme_promo_01_nologo_tiktok.mp4")}
        by = {x["variant"]: x for x in r["results"]}
        assert "logo" not in by["nologo"]["applied"] and "logo" in by["b"]["applied"]
        assert "Hook B" in (tmp_path / "w" / "b" / "tiktok" / "overlay.ass").read_text()
        assert E.variant_basename("x_{variant}_{platform}", "b") == "x_b_{platform}"

    def test_transcribed_once_for_all_platforms(self, m, tmp_path, monkeypatch):
        d, cat = m
        calls = []
        monkeypatch.setattr(E, "transcribe_input", lambda p, w: calls.append(p) or SEGS)
        rec = _recipe(cat, [{"op": "captions"}], export=("tiktok", "youtube"))
        r = E.render_input(str(d / "promo.mp4"), rec, str(tmp_path / "o"), str(tmp_path / "w"),
                           "acme_promo_{platform}")
        assert calls == [str(d / "promo.mp4")]
        assert all("captions" in x["applied"] for x in r["results"])

    def test_transcript_cached_on_disk(self, m, tmp_path, monkeypatch):
        import json
        (tmp_path / "transcript.json").write_text(json.dumps(SEGS))
        monkeypatch.setattr("pipeline.transcriber.transcribe_audio",
                            lambda *a, **k: pytest.fail("should use the cache"))
        assert E.transcribe_input("unused.mp4", str(tmp_path)) == SEGS

    def test_silent_input_gets_replacement_audio(self, m, tmp_path):
        d, cat = m
        rec = _recipe(cat, [{"op": "audio.replace", "track": "track.wav"}])
        r = E.render_input(str(d / "b" / "promo.mp4"), rec, str(tmp_path), str(tmp_path / "w"),
                           "x_{platform}")
        out = r["results"][0]
        assert media.probe(out["output_path"])["has_audio"] and "audio.replace" in out["applied"]

    def test_audio_file_rejected(self, m, tmp_path):
        d, cat = m
        with pytest.raises(ValueError, match="no video stream"):
            E.render_input(str(d / "notvideo.wav"), _recipe(cat, []), str(tmp_path),
                           str(tmp_path / "w"), "x")


class TestProcessInput:
    """DB bookkeeping: job_inputs status + deliverable rows (Docker DB)."""

    def test_roundtrip_and_rerun(self, m, tmp_path, monkeypatch):
        from campaign import service
        from db.repositories import campaign_repo, job_repo
        d, cat = m
        monkeypatch.setattr(service, "ASSET_ROOT", str(tmp_path / "assets"))
        monkeypatch.setattr(E, "OUTPUT_ROOT", str(tmp_path / "output"))
        monkeypatch.setattr(E, "WORK_ROOT", str(tmp_path / "work"))
        c = service.create_campaign("Edit Mode Test")
        try:
            rec = _recipe(cat, [{"op": "logo", "asset": "logo.png"}], export=("tiktok", "x"))
            job = job_repo.insert_job(c["id"], rec, 1, "edit")
            good = job_repo.add_input(job["id"], 0, str(d / "promo.mp4"))
            bad = job_repo.add_input(job["id"], 1, str(d / "notvideo.wav"))
            rows = E.process_input(job, c, good, "promo")
            assert [r["platform"] for r in rows] == ["tiktok", "x"]
            assert all(r["status"] == "rendered" and os.path.isfile(r["path"]) for r in rows)
            assert rows[0]["qa"]["width"] == 1080 and "logo" in rows[0]["qa"]["applied"]
            assert rows[1]["qa"].get("reused")             # x shares tiktok's 9:16 render
            with pytest.raises(ValueError):
                E.process_input(job, c, bad, "notvideo")
            st = {i["idx"]: i for i in job_repo.list_inputs(job["id"])}
            assert st[0]["status"] == "done" and st[0]["probe"]["has_video"]
            assert st[1]["status"] == "failed" and "no video stream" in st[1]["error"]
            E.process_input(job, c, st[0], "promo")          # re-run replaces the rows
            assert len(job_repo.list_deliverables(job["id"])) == 2
        finally:
            campaign_repo.delete_campaign(c["id"])
