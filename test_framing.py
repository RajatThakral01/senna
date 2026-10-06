# test_framing.py — layout decisions, tracking stability, crop bounds.
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from pipeline.framing import (
    SUPPORTED_LAYOUTS, validate_layout, decide_layout, smooth_centres,
    _crop_for_cx, map_speakers_to_faces,
)


def _analysis(face_counts, w=1280, h=720):
    samples = []
    for i, n in enumerate(face_counts):
        tracks = [{"id": j, "cx": 300 + j * 600, "cy": 300,
                   "box": {}} for j in range(n)]
        samples.append({"t": round(i * 0.25, 3), "boxes": [{}] * n,
                        "tracks": tracks, "scene_cut": False, "n": n})
    return {"samples": samples, "scenes": [0.0], "width": w, "height": h,
            "fps": 25.0, "duration": len(samples) * 0.25}


class TestLayouts:
    def test_supported_layouts(self):
        assert set(SUPPORTED_LAYOUTS) == {"auto", "speaker_crop",
                                          "stacked_split", "branded_fit"}

    def test_one_face_speaker_crop(self):
        d = decide_layout(_analysis([1] * 12), {}, "auto")
        assert d["layout"] == "speaker_crop"

    def test_two_faces_stacked(self):
        d = decide_layout(_analysis([2] * 12), {}, "auto")
        assert d["layout"] == "stacked_split"

    def test_no_faces_branded_fit(self):
        d = decide_layout(_analysis([0] * 12), {}, "auto")
        assert d["layout"] == "branded_fit"

    def test_uncertain_faces_branded_fit(self):
        d = decide_layout(_analysis([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
                          {"min_face_fraction": 0.5}, "auto")
        assert d["layout"] == "branded_fit"

    def test_manual_override_respected(self):
        for layout in SUPPORTED_LAYOUTS:
            if layout == "auto":
                continue
            d = decide_layout(_analysis([1] * 12), {}, layout)
            assert d["layout"] == layout

    def test_invalid_layout_falls_back(self):
        assert validate_layout("blur", "auto") == "auto"
        assert validate_layout(None, "branded_fit") == "branded_fit"

    def test_layout_stability_brief_dropout(self):
        # mostly 1 face with a 2-sample dropout -> stays speaker_crop
        counts = [1] * 6 + [0, 0] + [1] * 6
        d = decide_layout(_analysis(counts), {"layout_stability_frames": 6}, "auto")
        assert d["layout"] == "speaker_crop"


class TestCameraCut:
    def test_scene_cut_resets_and_switches(self):
        a = _analysis([1] * 6 + [1] * 6)
        # second half is a different shot (scene cut flag) — decision stays stable
        a["samples"][6]["scene_cut"] = True
        a["scenes"] = [0.0, 1.5]
        d = decide_layout(a, {}, "auto")
        assert d["layout"] == "speaker_crop"
        assert d["stable"] is True


class TestTracking:
    def test_temporary_loss_retains_then_falls_back(self):
        a = _analysis([1] * 4 + [0] * 12)
        sm = smooth_centres(a, {"hold_last_position_seconds": 1.0,
                                "sample_interval": 0.25})
        # first missing samples retain last position
        assert sm[4]["cx"] is not None and sm[4]["lost"] is False
        # after hold window -> lost fallback (branded_fit at render)
        assert sm[-1]["lost"] is True

    def test_no_largest_face_assumption(self):
        # two tracks; lowest stable id is followed, not the largest box
        a = _analysis([2] * 6)
        sm = smooth_centres(a, {})
        assert all(s["cx"] is not None for s in sm)


class TestCropBounds:
    def test_crop_inside_frame_edges(self):
        cfg = {}
        for cx, cy in [(0, 0), (1280, 720), (640, 360), (-50, 900)]:
            x, y, cw, ch = _crop_for_cx(1280, 720, cx, cy, cfg)
            assert 0 <= x <= 1280 - cw
            assert 0 <= y <= 720 - ch
            assert x + cw <= 1280 and y + ch <= 720

    def test_output_ratio_is_9_16(self):
        x, y, cw, ch = _crop_for_cx(3840, 2160, 1920, 1080, {})
        assert abs((cw / ch) - (1080 / 1920)) < 0.02


class TestDiarizationMapping:
    def test_audio_alone_never_maps(self):
        a = _analysis([2] * 6)
        diar = [{"speaker": "SPEAKER_00", "start": 0, "end": 1}]
        assert map_speakers_to_faces(diar, a, speaker_map=None) == {}
        assert map_speakers_to_faces(diar, a, speaker_map={}) == {}

    def test_explicit_speaker_map(self):
        a = _analysis([2] * 6)
        diar = [{"speaker": "SPEAKER_00", "start": 0, "end": 1}]
        m = map_speakers_to_faces(diar, a,
                                  speaker_map={"SPEAKER_00": "left",
                                               "SPEAKER_01": "right"})
        assert m["SPEAKER_00"] != m["SPEAKER_01"]
