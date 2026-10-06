# test_editplan_polish.py — Phase 5: plans, consolidated command, audio graph.
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from pipeline.editplan import build_edit_plan, plan_fingerprint
from pipeline.polish import (build_video_chain, build_audio_chain,
                             build_polish_cmd, probe_streams)


def _clip():
    return {"clip_number": 7, "id": "x", "start_time": 10.0, "end_time": 40.0,
            "source_ranges": [[10.0, 40.0]],
            "timeline": [{"source_start": 10.0, "source_end": 40.0,
                          "output_start": 0.0, "output_end": 30.0}],
            "layout": "branded_fit", "layout_reason": "test"}


def _cfg():
    return {"framing": {}, "captions": {"font": "Arial"},
            "audio": {"normalize_loudness": True, "duck_music": True},
            "export": {"codec": "libx264", "preset": "fast", "crf": 23},
            "fades": {"audio": False, "duration": 0.5},
            "ai": {"llm_model": "m"}, "embeddings": {"model": "e"},
            "transcription": {"model": "base"}}


class TestPlan:
    def test_structure(self):
        plan = build_edit_plan(_clip(), {}, _cfg())
        assert plan["plan_version"] == 1
        assert plan["output_duration"] == 30.0
        assert plan["export"]["width"] == 1080
        assert "does NOT restore detail" in plan["export"]["detail_note"]
        assert plan["audio"]["audio_fade"] is False
        assert plan["config_fingerprint"]

    def test_fingerprint_sensitive(self):
        a = build_edit_plan(_clip(), {}, _cfg())
        cfg2 = _cfg()
        cfg2["export"]["crf"] = 18
        b = build_edit_plan(_clip(), {}, cfg2)
        assert a["config_fingerprint"] != b["config_fingerprint"]

    def test_caption_data_carried(self):
        cap = {"cues": [{"start": 0, "end": 1, "words": [],
                         "text": "hi"}], "placement": "top"}
        plan = build_edit_plan(_clip(), {}, _cfg(), caption_data=cap)
        assert plan["captions"]["cue_count"] == 1
        assert plan["captions"]["placement"] == "top"

    def test_plan_fingerprint_stable(self):
        assert plan_fingerprint({"a": 1}) == plan_fingerprint({"a": 1})


class TestVideoChain:
    def test_ass_fade_setsar(self):
        chain, _ = build_video_chain(ass_file="x.ass", fade_duration=0.5,
                                     out_duration=30.0, use_logo=False)
        assert "ass=x.ass" in chain and "fade=t=in" in chain
        assert "fade=t=out" in chain and chain.endswith("setsar=1[v]")

    def test_logo_overlay(self):
        chain, _ = build_video_chain(use_logo=True)
        assert "overlay=" in chain and chain.endswith("[v]")

    def test_no_fade_when_zero(self):
        chain, _ = build_video_chain(fade_duration=0, out_duration=30.0,
                                     use_logo=False)
        assert "fade=" not in chain


class TestAudioChain:
    def test_speech_only_normalized(self):
        filt = build_audio_chain(True, False)
        assert "loudnorm" in filt and "alimiter" in filt
        assert "sidechain" not in filt and filt.endswith("[aout]")

    def test_ducking_with_music(self):
        filt = build_audio_chain(True, True, duck=True)
        assert "sidechaincompress" in filt and "amix" in filt

    def test_no_duck_without_flag(self):
        filt = build_audio_chain(True, True, duck=False)
        assert "sidechaincompress" not in filt and "amix" in filt

    def test_silent_no_chain(self):
        assert build_audio_chain(False, False) == ""

    def test_music_only(self):
        filt = build_audio_chain(False, True)
        assert "[1:a]" in filt and filt.endswith("[aout]")

    def test_afade_opt_in_only(self):
        assert "afade" not in build_audio_chain(True, False, audio_fade=False,
                                               out_duration=30.0)
        assert "afade" in build_audio_chain(True, False, audio_fade=True,
                                            fade_duration=0.5, out_duration=30.0)


class TestCmd:
    def test_assembled(self):
        cmd, desc = build_polish_cmd("in.mp4", "out.mp4", ass_rel="c.ass",
                                     logo_path="l.png", music_path="m.mp3",
                                     out_duration=30.0, has_audio=True)
        s = " ".join(cmd)
        assert "ass=c.ass" in s and "overlay=" in s and "sidechain" in s
        assert "-map" in cmd and "ass" in desc and "logo=True" in desc

    def test_silent_an(self):
        cmd, _ = build_polish_cmd("in.mp4", "out.mp4", has_audio=False)
        assert "-an" in cmd


class TestProbe:
    def test_real_file(self):
        info = probe_streams("input/raw_video.mp4")
        assert info["has_video"] and info["has_audio"]
        assert info["width"] == 3840 and info["duration"] > 300

    def test_missing_file(self):
        info = probe_streams("no/such/file.mp4")
        assert info == {"has_video": False, "has_audio": False, "width": 0,
                        "height": 0, "fps": 0.0, "duration": 0.0}


class TestPlanRepo:
    def test_round_trip(self):
        from db.repositories import video_repo, clip_repo, editplan_repo
        vid = video_repo.insert_video("test://phase5", "test://phase5")
        try:
            cid = clip_repo.insert_clip(vid, 1, 0.0, 10.0, duration_seconds=10.0)
            plan = build_edit_plan({"clip_number": 1, "id": cid,
                                    "start_time": 0.0, "end_time": 10.0,
                                    "source_ranges": [[0.0, 10.0]]}, {}, _cfg())
            pid, ver = editplan_repo.save_plan(vid, cid, plan)
            assert pid and ver == 1
            latest = editplan_repo.get_latest_plan(cid)
            assert latest["version"] == 1
            pid2, ver2 = editplan_repo.save_plan(vid, cid, plan)
            assert ver2 == 2
            assert editplan_repo.get_latest_plan(cid)["version"] == 2
        finally:
            from conftest import cleanup_video
            cleanup_video(vid)
