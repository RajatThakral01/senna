# test_speakers_visual.py — Phase 7: adapter rules + visual interface.
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from pipeline.speakers import stable_turns, match_voices_to_faces, plan_switches
from pipeline.visual import backend_status, sample_keyframes, analyze_moments

ON = {"speakers": {"enabled": True}}


def _analysis(track_ids=(0, 1)):
    return {"width": 1280, "height": 720, "samples": [
        {"t": 0.0, "tracks": [{"id": i, "cx": 200 + i * 800, "cy": 300}
                              for i in track_ids]},
        {"t": 0.5, "tracks": [{"id": i, "cx": 200 + i * 800, "cy": 300}
                              for i in track_ids]},
    ]}


class TestTurns:
    def test_micro_turns_dropped(self):
        diar = [{"speaker": "A", "start": 0, "end": 0.5},
                {"speaker": "A", "start": 0.6, "end": 3.0}]
        turns = stable_turns(diar, min_turn_seconds=1.5)
        assert len(turns) == 1 and turns[0]["end"] == 3.0

    def test_empty(self):
        assert stable_turns([], 1.5) == []


class TestMapping:
    def test_disabled(self):
        m, st = match_voices_to_faces([{"speaker": "A", "start": 0, "end": 2}],
                                      _analysis(), {}, {"speakers": {"enabled": False}})
        assert m == {} and st["status"] == "disabled"

    def test_no_diarization(self):
        m, st = match_voices_to_faces([], _analysis(), {"A": "left"}, ON)
        assert m == {} and st["status"] == "unavailable_no_diarization"

    def test_single_face_never_maps(self):
        # one visible face during speech must NOT imply mapping
        m, st = match_voices_to_faces([{"speaker": "A", "start": 0, "end": 2}],
                                      _analysis(track_ids=(0,)),
                                      {"A": "left"}, ON)
        assert m == {} and st["status"] == "unavailable_no_faces"

    def test_explicit_map(self):
        diar = [{"speaker": "SPEAKER_00", "start": 0, "end": 2}]
        m, st = match_voices_to_faces(
            diar, _analysis(), {"SPEAKER_00": "left"}, ON)
        assert st["status"] == "mapped_explicit"
        assert m == {"SPEAKER_00": 0}

    def test_auto_unevaluated(self):
        diar = [{"speaker": "A", "start": 0, "end": 2}]
        m, st = match_voices_to_faces(
            diar, _analysis(), None,
            {"speakers": {"enabled": True, "auto_match": True}})
        assert m == {} and st["status"] == "unavailable_unevaluated"


class TestSwitches:
    def _diar(self):
        return [{"speaker": "A", "start": 0, "end": 5},
                {"speaker": "B", "start": 6, "end": 11}]

    def test_switches_with_hold(self):
        sw = plan_switches(self._diar(), {"A": 0, "B": 1},
                           hold_seconds=2.0)
        assert [s["speaker"] for s in sw] == ["A", "B"]
        assert sw[0]["track_id"] == 0 and sw[1]["track_id"] == 1

    def test_no_mapping_no_switches(self):
        assert plan_switches(self._diar(), {}) == []

    def test_overlap_holds_current(self):
        diar = [{"speaker": "A", "start": 0, "end": 6},
                {"speaker": "B", "start": 5, "end": 11}]
        sw = plan_switches(diar, {"A": 0, "B": 1}, hold_seconds=0.5)
        assert [s["speaker"] for s in sw] == ["A"]


class TestVisual:
    def test_disabled_default(self):
        st = backend_status({})
        assert st["status"] == "disabled"

    def test_unknown_backend(self):
        st = backend_status({"visual": {"enabled": True, "backend": "wat"}})
        assert st["status"] == "disabled"

    def test_groq_vision_unavailable(self):
        st = backend_status({"visual": {"enabled": True, "backend": "groq_vision"}})
        assert st["status"] == "unavailable"

    def test_no_fake_labels(self):
        labels, st = analyze_moments("v", [(0, 10)], [], {})
        assert labels == {} and st["status"] == "disabled"

    def test_keyframe_sampling(self):
        if not os.path.exists("input/raw_video.mp4"):
            return
        frames = sample_keyframes("input/raw_video.mp4", [(30.0, 40.0)],
                                  {"scenes": [32.0, 38.0]},
                                  {"visual": {"max_frames": 4}})
        assert 1 <= len(frames) <= 4
        assert all(os.path.exists(f["path"]) for f in frames)
        kinds = {f["kind"] for f in frames}
        assert "mid" in kinds  # midpoint always sampled
        for f in frames:
            os.remove(f["path"])
