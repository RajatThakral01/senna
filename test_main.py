# test_main.py — tests for the current main.py pipeline API.
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(__file__))
from main import merge_config, to_sec, load_template, select_renderable_clips


class TestHelpers:
    def test_merge_config_override(self):
        template = {"a": 1, "b": 2, "c": 3}
        campaign = {"b": 20, "d": None}
        merged = merge_config(template, campaign)
        assert merged == {"a": 1, "b": 20, "c": 3}

    def test_to_sec(self):
        assert to_sec(90) == 90.0
        assert to_sec("00:01:30") == 90.0
        assert to_sec("01:30") == 90.0

    def test_load_template(self):
        tmpl = load_template("podcast_clip")
        assert isinstance(tmpl, dict)

    def test_select_renderable_clips(self):
        clips = [
            {"clip_number": 1, "refine_status": "refined"},
            {"clip_number": 2, "refine_status": "rejected",
             "refine_reason": "no complete ending"},
            {"clip_number": 3},  # legacy rows without status still render
        ]
        renderable, nrej = select_renderable_clips(clips)
        assert [c["clip_number"] for c in renderable] == [1, 3]
        assert nrej == 1


class TestRunPipeline:
    def _mocks(self):
        p = {}
        p["handle"] = patch("main.handle_input", return_value="input/raw_video.mp4")
        p["insert"] = patch("main.video_repo.insert_video", return_value="vid-1")
        p["parse"] = patch("main.parse_campaign", return_value={"subtitles": True, "fade": False})
        p["transcribe"] = patch("main.transcribe_audio",
                                return_value=("transcripts/transcript.json", {"segments": []}))
        p["chunk"] = patch("main.build_chunks", return_value=[])
        p["embed"] = patch("main.embed_all_chunks", return_value=[])
        p["analyze"] = patch("main.analyze", return_value=[])
        p["sim"] = patch("main.find_related_segments", return_value=[])
        # refine helpers are imported function-locally in run_pipeline
        p["refine"] = patch("pipeline.boundaries.refine_all_clips", return_value=[])
        p["persist"] = patch("pipeline.boundaries.persist_refinements", return_value=(0, 0))
        p["render"] = patch("main.render_clips", return_value=[{"file": "output/clip_1_final.mp4"}])
        p["skip"] = patch("main.should_skip_stage", return_value=False)
        # stages query repos directly (function-local imports in main)
        p["chunks"] = patch("db.repositories.chunk_repo.get_chunks_for_video",
                            return_value=[])
        p["getclips"] = patch("db.repositories.clip_repo.get_clips_for_video",
                              return_value=[])
        p["confirmed"] = patch(
            "db.repositories.clip_repo.get_confirmed_segments_for_video",
            return_value={})
        # outline stage fingerprint checks
        p["fp_match"] = patch("main.run_repo.fingerprint_matches",
                              return_value=True)
        p["fp_set"] = patch("main.run_repo.set_fingerprint")
        p["get_status"] = patch("main.run_repo.get_stage_status",
                                return_value="done")
        p["invalidate"] = patch("main.invalidate_stage")
        # outline build/persist called function-locally
        p["load_words"] = patch("pipeline.boundaries.load_words", return_value=[])
        p["build_sents"] = patch("pipeline.boundaries.build_sentences",
                                 return_value=[])
        p["build_ol"] = patch("pipeline.outline.build_outline",
                              return_value={"method": "t", "sections": []})
        p["persist_ol"] = patch("pipeline.outline.persist_outline")
        # audio_events stage (function-local imports)
        p["ensure_wav"] = patch("pipeline.audio_events.ensure_source_audio",
                                return_value="downloads/audio.wav")
        p["detect_ev"] = patch("pipeline.audio_events.detect_audio_events",
                               return_value=([], "fp"))
        p["classify_ev"] = patch("pipeline.audio_classifier.classify_events",
                                 return_value=([], {"status": "disabled"}))
        p["propose_ev"] = patch("pipeline.audio_events.propose_from_events",
                                return_value=[])
        p["enrich_ev"] = patch("pipeline.audio_events.enrich_event_candidates",
                               return_value=[])
        p["persist_ev"] = patch("pipeline.audio_events.persist_events")
        p["cand_clear"] = patch("db.repositories.candidate_repo.clear_candidates")
        p["cand_insert"] = patch("db.repositories.candidate_repo.insert_candidate")
        p["start"] = patch("main.run_repo.start_stage")
        p["complete"] = patch("main.run_repo.complete_stage")
        p["dur"] = patch("main.get_video_duration", return_value=100.0)
        # resume lookup / workspace / per-run log / keep-awake
        import contextlib
        from pipeline.workspace import Workspace
        p["find_prev"] = patch("main.video_repo.find_latest_by_source", return_value=None)
        p["get_video"] = patch("main.video_repo.get_video",
                               return_value={"raw_path": "input/raw_video.mp4"})
        p["ws"] = patch("main.workspace.activate", return_value=Workspace())
        p["reset"] = patch("main.reset_from_stage")
        p["run_log"] = patch("main.run_log",
                             side_effect=lambda *a, **k: contextlib.nullcontext("test.log"))
        p["awake"] = patch("main._keep_awake", return_value=None)
        return p

    def test_run_pipeline_calls_stages(self):
        from main import run_pipeline
        m = self._mocks()
        s = {k: v.start() for k, v in m.items()}
        try:
            out = run_pipeline("some_video.mp4", "podcast style", None)
            assert out == [{"file": "output/clip_1_final.mp4"}]
            s["handle"].assert_called_once_with("some_video.mp4")
            s["transcribe"].assert_called_once()
            s["chunk"].assert_called_once()
            s["embed"].assert_called_once()
            s["analyze"].assert_called_once()
            s["sim"].assert_called_once()
            s["render"].assert_called_once()
        finally:
            for v in m.values():
                v.stop()

    def test_run_pipeline_layout_forwarded(self):
        from main import run_pipeline
        m = self._mocks()
        s = {k: v.start() for k, v in m.items()}
        try:
            run_pipeline("v.mp4", "", None, layout="branded_fit")
            _, kwargs = s["render"].call_args
            assert kwargs.get("layout") == "branded_fit"
        finally:
            for v in m.values():
                v.stop()


class TestCLI:
    def test_no_input_prints_usage(self):
        from main import main as cli_main
        with patch("sys.argv", ["main.py"]):
            with patch("builtins.print") as mock_print:
                cli_main()
        assert mock_print.call_count > 0

    def test_layout_choices_accepted(self):
        import argparse
        # argparse-level check: --layout must accept the four layouts
        from main import main as cli_main
        with patch("sys.argv", ["main.py", "v.mp4", "--layout", "stacked_split"]):
            with patch("main.run_pipeline", return_value=[]) as rp:
                cli_main()
        rp.assert_called_once()
        _, kwargs = rp.call_args
        assert kwargs.get("layout") == "stacked_split"


class TestResumeAndWorkspace:
    def test_workspace_paths_are_per_video(self):
        from pipeline.workspace import Workspace
        a = Workspace("c9aa5363-b79c-4837-9c15-5aa64519d6b6", "https://youtu.be/plN7JMbadRg")
        b = Workspace("11111111-2222-3333-4444-555555555555", "https://youtu.be/dQw4w9WgXcQ")
        assert a.label == "plN7JMbadRg_c9aa5363"
        assert a.transcript != b.transcript and a.audio != b.audio
        assert a.final(1) == os.path.join("output", "plN7JMbadRg_c9aa5363", "clip_1_final.mp4")
        assert a.clip_file(3, ".srt").endswith(os.path.join("clips", "clip_3.srt"))

    def test_legacy_workspace_without_video(self):
        from pipeline.workspace import Workspace
        w = Workspace()
        assert w.transcript == os.path.join("transcripts", "transcript.json")
        assert w.final(2) == os.path.join("output", "clip_2_final.mp4")

    def test_reset_from_stage_rejects_unknown(self):
        import pytest
        from main import reset_from_stage
        with pytest.raises(ValueError):
            reset_from_stage("vid", "bogus")

    def test_resume_reuses_previous_video(self):
        t = TestRunPipeline()
        m = t._mocks()
        m["find_prev"] = patch("main.video_repo.find_latest_by_source",
                               return_value={"id": "old-vid", "raw_path": "input/raw_video.mp4"})
        s = {k: v.start() for k, v in m.items()}
        try:
            from main import run_pipeline
            run_pipeline("some_video.mp4", "", None)
            s["insert"].assert_not_called()
            s["ws"].assert_called_once_with("old-vid")
        finally:
            for v in m.values():
                v.stop()

    def test_fresh_and_from_stage(self):
        t = TestRunPipeline()
        m = t._mocks()
        m["find_prev"] = patch("main.video_repo.find_latest_by_source",
                               return_value={"id": "old-vid", "raw_path": "x"})
        s = {k: v.start() for k, v in m.items()}
        try:
            from main import run_pipeline
            run_pipeline("some_video.mp4", "", None, resume=False, from_stage="refine")
            s["insert"].assert_called_once()
            s["reset"].assert_any_call("vid-1", "refine")
        finally:
            for v in m.values():
                v.stop()
