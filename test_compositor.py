# test_compositor.py — campaign pipeline C3: recipe → rendered deliverable.
# Renders small synthetic videos with real ffmpeg and checks the output
# (size, duration, audio content, overlays in the ASS script, timing).
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from campaign.recipe import validate_recipe  # noqa: E402
from pipeline import compositor as C  # noqa: E402
from pipeline import media  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(not FFMPEG, reason="ffmpeg not installed")


def _ff(*args):
    r = subprocess.run([FFMPEG, "-loglevel", "error", "-y", *args], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]


@pytest.fixture(scope="module")
def m(tmp_path_factory):
    d = tmp_path_factory.mktemp("comp")
    _ff("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=6",
        "-f", "lavfi", "-i", "sine=frequency=300:duration=6", "-shortest",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(d / "wide.mp4"))
    _ff("-f", "lavfi", "-i", "testsrc2=size=360x640:rate=30:duration=4", "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(d / "vertical_silent.mp4"))
    _ff("-f", "lavfi", "-i", "sine=frequency=660:duration=2", str(d / "track.wav"))
    _ff("-f", "lavfi", "-i", "sine=frequency=1000:duration=2", str(d / "music.wav"))
    _ff("-f", "lavfi", "-i", "color=c=red@0.6:size=300x120,format=rgba", "-frames:v", "1",
        str(d / "logo.png"))
    cat = [{"name": "logo.png", "kind": "logo", "path": str(d / "logo.png")},
           {"name": "track.wav", "kind": "audio", "path": str(d / "track.wav")},
           {"name": "music.wav", "kind": "audio", "path": str(d / "music.wav")}]
    return d, cat


def _recipe(cat, ops, export=("generic",)):
    v = validate_recipe({"mode": "edit", "ops": ops, "export": list(export)}, cat)
    assert v["ok"], v["errors"]
    return v["recipe"]


def _peak_freqs(path, start=0.5, dur=1.5):
    """Dominant audio frequencies (Hz) of a file section."""
    r = subprocess.run([FFMPEG, "-loglevel", "error", "-ss", str(start), "-t", str(dur),
                        "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                       capture_output=True)
    x = np.frombuffer(r.stdout, dtype=np.float32)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1 / 16000)
    rms = float(np.sqrt((x ** 2).mean())) if len(x) else 0.0
    top = f[np.argsort(spec)[-40:]]
    return {round(v / 10) * 10 for v in top}, rms


WORDS = [{"word": w, "start": 0.3 + i * 0.5, "end": 0.3 + i * 0.5 + 0.45}
         for i, w in enumerate("hello campaign world this is great".split())]
SEGS = [{"start": WORDS[0]["start"], "end": WORDS[-1]["end"], "words": WORDS, "text": ""}]


class TestTiming:
    def test_trim_and_word_snap(self):
        s, d, info = C.plan_timing(10.0, {"start": 1, "max_duration": 4},
                                   [{"start": 1.2, "end": 2.0}, {"start": 3.0, "end": 4.6},
                                    {"start": 4.8, "end": 5.6}])
        assert s == 1.0 and info["snapped_to_word"] and d == pytest.approx(3.85, abs=0.01)

    def test_min_duration_flag(self):
        _, d, info = C.plan_timing(3.0, {"min_duration": 10})
        assert d == 3.0 and info["too_short"]

    def test_window(self):
        assert C._window({"from_end": 2}, 10) == (8, 10)
        assert C._window({"start": 1, "end": 30}, 10) == (1, 10)

    def test_static_crop(self):
        cw, ch, x, y = C._static_crop(1920, 1080, 9 / 16)
        assert (cw, ch, y) == (608, 1080, 0) and x == (1920 - 608) // 2
        cw, ch, x, y = C._static_crop(1920, 1080, 1.0, cx=300, cy=500)
        assert (cw, ch) == (1080, 1080) and x == 0


class TestASS:
    def test_captions_and_overlays_in_one_script(self):
        from pipeline.captions import group_phrases
        cues = group_phrases(WORDS, [{"source_start": 0, "source_end": 10, "output_start": 0,
                                      "output_end": 10}], {})
        ass = C.build_overlay_ass(1080, 1920, 5.0, cues,
                                  {"enabled": True, "style": "karaoke", "uppercase": True},
                                  [{"text": "Link {in} bio", "role": "cta", "position": "bottom",
                                    "from_end": 2, "animation": "fade", "box": True}])
        assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
        assert "\\kf" in ass and "HELLO" in ass
        cta = [ln for ln in ass.splitlines() if ",T0," in ln][0]
        assert "0:00:03.00,0:00:05.00" in cta and "\\fad(250,250)" in cta
        assert "Link (in) bio" in cta          # braces escaped, no ASS injection

    def test_plain_and_boxed_styles(self):
        from pipeline.captions import group_phrases
        cues = group_phrases(WORDS, [{"source_start": 0, "source_end": 10, "output_start": 0,
                                      "output_end": 10}], {})
        plain = C.build_overlay_ass(1080, 1080, 5, cues, {"style": "plain"}, [])
        boxed = C.build_overlay_ass(1080, 1080, 5, cues, {"style": "boxed"}, [])
        assert "\\kf" not in plain
        assert ",3," in [ln for ln in boxed.splitlines() if ln.startswith("Style: Cap")][0]

    def test_bottom_text_clears_captions(self):
        from pipeline.captions import group_phrases
        cues = group_phrases(WORDS, [{"source_start": 0, "source_end": 10, "output_start": 0,
                                      "output_end": 10}], {})
        for W, H in ((1080, 1920), (1920, 1080)):
            ass = C.build_overlay_ass(W, H, 5, cues, {"style": "plain"},
                                      [{"text": "Link in bio", "position": "bottom"}])
            st = {ln.split(",")[0][7:]: ln.split(",") for ln in ass.splitlines()
                  if ln.startswith("Style:")}
            cap_mv, cap_size = int(st["Cap"][-2]), int(st["Cap"][2])
            assert int(st["T0"][-2]) >= cap_mv + 2 * cap_size      # above both caption lines

    def test_nothing_to_draw(self):
        assert C.build_overlay_ass(1080, 1920, 5, None, None, []) is None

    def test_color(self):
        assert C.ass_color("#FF8000") == "&H000080FF"


class TestCompose:
    def test_full_recipe_wide_to_vertical(self, m, tmp_path, monkeypatch):
        d, cat = m
        heard = []
        monkeypatch.setattr(C, "transcribe_segment",
                            lambda path, *a, **k: heard.append(path) or SEGS)
        rec = _recipe(cat, [{"op": "trim", "max_duration": 4},
                            {"op": "reframe", "aspect": "9:16", "method": "center"},
                            {"op": "audio.replace", "track": "track.wav"},
                            {"op": "logo", "asset": "logo.png", "position": "top-left"},
                            {"op": "captions"},
                            {"op": "text_overlay", "text": "Link in bio", "role": "cta",
                             "from_end": 1.5}])
        out = tmp_path / "o.mp4"
        r = C.compose(str(d / "wide.mp4"), rec, str(out), work_dir=str(tmp_path / "w"),
                      transcript_segments=SEGS)
        p = media.probe(str(out))
        assert (p["width"], p["height"]) == (1080, 1920) and p["has_audio"]
        assert p["duration"] == pytest.approx(r["expected_duration"], abs=0.1)
        assert r["expected_duration"] <= 4.0
        assert set(r["applied"]) >= {"trim", "reframe", "audio.replace", "logo", "captions",
                                     "text_overlay"}
        freqs, _ = _peak_freqs(out)
        assert 660 in freqs and 300 not in freqs       # original replaced, track looped in
        ass = (tmp_path / "w" / "overlay.ass").read_text()
        assert "Link in bio" in ass and "Dialogue: 0," in ass
        assert heard == [str(d / "track.wav")]     # captions follow the new voice

    def test_keep_original_under_replacement(self, m, tmp_path):
        d, cat = m
        rec = _recipe(cat, [{"op": "audio.replace", "track": "track.wav", "keep_original": 0.5}])
        out = tmp_path / "o.mp4"
        C.compose(str(d / "wide.mp4"), rec, str(out))
        freqs, _ = _peak_freqs(out)
        assert 660 in freqs and 300 in freqs

    def test_music_ducked_under_original(self, m, tmp_path):
        d, cat = m
        rec = _recipe(cat, [{"op": "audio.add_music", "track": "music.wav", "volume": 0.3}])
        out = tmp_path / "o.mp4"
        C.compose(str(d / "wide.mp4"), rec, str(out), work_dir=str(tmp_path / "w"))
        cmd = " ".join(json.load(open(tmp_path / "w" / "compose_cmd.json")))
        assert "sidechaincompress" in cmd and "loudnorm" in cmd
        freqs, _ = _peak_freqs(out)
        assert 300 in freqs and 1000 in freqs

    def test_mute_gives_silence_track(self, m, tmp_path):
        d, cat = m
        out = tmp_path / "o.mp4"
        r = C.compose(str(d / "wide.mp4"), _recipe(cat, [{"op": "audio.mute_original"}]), str(out))
        _, rms = _peak_freqs(out)
        assert media.probe(str(out))["has_audio"] and rms < 1e-3
        assert any("silent track" in w for w in r["warnings"])

    def test_vertical_silent_input_square_fit_blur(self, m, tmp_path):
        d, cat = m
        rec = _recipe(cat, [{"op": "reframe", "aspect": "1:1", "method": "fit_blur"}])
        out = tmp_path / "o.mp4"
        r = C.compose(str(d / "vertical_silent.mp4"), rec, str(out))
        p = media.probe(str(out))
        assert (p["width"], p["height"]) == (1080, 1080) and p["has_audio"]
        assert r["framing"] == "fit_blur"

    def test_vertical_to_wide_auto_uses_fit_blur(self, m, tmp_path):
        d, cat = m
        r = C.compose(str(d / "vertical_silent.mp4"), _recipe(cat, [], export=("youtube",)),
                      str(tmp_path / "o.mp4"), platform="youtube")
        assert r["framing"] == "fit_blur" and any("blurred background" in w
                                                  for w in r["warnings"])

    def test_already_vertical_is_scaled_not_cropped(self, m, tmp_path):
        d, cat = m
        r = C.compose(str(d / "vertical_silent.mp4"), _recipe(cat, []), str(tmp_path / "o.mp4"))
        assert r["framing"] == "scale" and (r["width"], r["height"]) == (1080, 1920)

    def test_missing_files_warn_not_crash(self, m, tmp_path):
        d, cat = m
        rec = _recipe(cat, [{"op": "logo", "asset": "logo.png"}])
        rec["ops"][0]["asset_path"] = str(tmp_path / "gone.png")
        r = C.compose(str(d / "wide.mp4"), rec, str(tmp_path / "o.mp4"))
        assert any("logo: file missing" in w for w in r["warnings"]) and "logo" not in r["applied"]

    def test_speed_keeps_pitch_and_retimes_words(self, m, tmp_path):
        d, cat = m
        rec = _recipe(cat, [{"op": "trim", "max_duration": 4}, {"op": "speed", "factor": 2},
                            {"op": "captions"}])
        out = tmp_path / "o.mp4"
        r = C.compose(str(d / "wide.mp4"), rec, str(out), work_dir=str(tmp_path / "w"),
                      transcript_segments=SEGS)
        # 4 s cap snapped to the last word (3.5 s) → 1.75 s at 2x
        assert "speed" in r["applied"] and r["duration"] == pytest.approx(1.75, abs=0.1)
        assert r["expected_duration"] == pytest.approx(r["duration"], abs=0.1)
        freqs, _ = _peak_freqs(out, 0.3, 1.2)
        assert 300 in freqs                             # atempo: faster, same pitch
        ass = (tmp_path / "w" / "overlay.ass").read_text()
        first = [ln for ln in ass.splitlines() if ln.startswith("Dialogue: 0,")][0]
        assert first.split(",")[1] == "0:00:00.15"      # 0.3 s word at 2x → 0.15 s

    def test_color_bw(self, m, tmp_path):
        d, cat = m
        out = tmp_path / "o.mp4"
        r = C.compose(str(d / "wide.mp4"), _recipe(cat, [{"op": "color", "preset": "bw"}]),
                      str(out))
        fr = subprocess.run([FFMPEG, "-loglevel", "error", "-ss", "1", "-i", str(out),
                             "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                            capture_output=True).stdout
        px = np.frombuffer(fr, dtype=np.uint8).reshape(-1, 3).astype(int)
        assert "color" in r["applied"] and np.abs(px[:, 0] - px[:, 2]).mean() < 4
        assert C.color_filter({"preset": "none", "contrast": 1.2}).startswith("eq=")
        assert C.color_filter({"preset": "none"}) is None

    def test_watermark_is_translucent_text(self, m, tmp_path):
        d, cat = m
        r = C.compose(str(d / "wide.mp4"), _recipe(cat, [{"op": "watermark", "text": "@acme",
                                                          "opacity": 0.5}]),
                      str(tmp_path / "o.mp4"), work_dir=str(tmp_path / "w"))
        ass = (tmp_path / "w" / "overlay.ass").read_text()
        style = [ln for ln in ass.splitlines() if ln.startswith("Style: T0")][0]
        assert "watermark" in r["applied"] and "&H80FFFFFF" in style and "@acme" in ass

    def test_intro_outro_end_card(self, m, tmp_path, monkeypatch):
        monkeypatch.chdir(os.getcwd())              # restored after the test
        d, cat = m
        _ff("-f", "lavfi", "-i", "color=c=green:size=640x360:rate=30:duration=1.5",
            "-f", "lavfi", "-i", "sine=frequency=880:duration=1.5", "-shortest",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(tmp_path / "outro.mp4"))
        cat2 = cat + [{"name": "outro.mp4", "kind": "video", "path": str(tmp_path / "outro.mp4")},
                      {"name": "card.png", "kind": "image", "path": str(d / "logo.png")}]
        rec = _recipe(cat2, [{"op": "trim", "max_duration": 2},
                             {"op": "intro", "asset": "card.png", "duration": 1},
                             {"op": "outro", "asset": "outro.mp4", "transition": "fade"},
                             {"op": "end_card", "text": "Follow @acme", "duration": 1},
                             {"op": "logo", "asset": "logo.png"}])
        out = tmp_path / "o.mp4"
        os.chdir(tmp_path)                          # relative work dir, as the jobs use
        r = C.compose(str(d / "wide.mp4"), rec, str(out), platform="tiktok", work_dir="w")
        p = media.probe(str(out))
        assert {"intro", "outro", "end_card"} <= set(r["applied"]), r["warnings"]
        assert r["body_offset"] == pytest.approx(1.0, abs=0.05)
        assert p["duration"] == pytest.approx(1 + 2 + 1.5 + 1, abs=0.2)
        assert r["expected_duration"] == pytest.approx(p["duration"], abs=0.2)
        assert (p["width"], p["height"]) == (1080, 1920)
        freqs, _ = _peak_freqs(out, 3.3, 0.8)          # inside the outro
        assert 880 in freqs
        from campaign.qa import check_file
        qa = check_file(str(out), rec, "tiktok", r)
        assert qa["status"] == "pass", qa["checks"]     # logo found inside the body window

    def test_compose_all_reuses_same_aspect(self, m, tmp_path):
        d, cat = m
        rec = _recipe(cat, [{"op": "trim", "max_duration": 2}], export=("tiktok", "reels", "youtube"))
        res = C.compose_all(str(d / "wide.mp4"), rec, str(tmp_path), "acme_v1")
        assert [r["platform"] for r in res] == ["tiktok", "reels", "youtube"]
        assert res[1].get("reused") and not res[2].get("reused")
        assert (res[2]["width"], res[2]["height"]) == (1920, 1080)
        assert all(os.path.isfile(r["output_path"]) for r in res)

    def test_no_video_rejected(self, m, tmp_path):
        d, cat = m
        with pytest.raises(C.ComposeError):
            C.compose(str(d / "track.wav"), _recipe(cat, []), str(tmp_path / "o.mp4"))
