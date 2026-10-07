# test_framing.py — layout decisions, tracking stability, crop bounds,
# full-screen vertical requirement (no bars), scene plans, manual ROI.
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from pipeline.framing import (
    SUPPORTED_LAYOUTS, validate_layout, decide_layout, smooth_centres,
    _crop_for_cx, map_speakers_to_faces, plan_scene_crops, resolve_anchor,
    validate_roi, center_crop_rect, estimate_content_bounds, fullscreen,
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
                                          "stacked_split", "center_crop",
                                          "branded_fit"}

    def test_one_face_speaker_crop(self):
        d = decide_layout(_analysis([1] * 12), {}, "auto")
        assert d["layout"] == "speaker_crop"

    def test_two_faces_stacked(self):
        d = decide_layout(_analysis([2] * 12), {}, "auto")
        assert d["layout"] == "stacked_split"

    def test_no_faces_center_crop_needs_review(self):
        d = decide_layout(_analysis([0] * 12), {}, "auto")
        assert d["layout"] == "center_crop"
        assert d["needs_review"] is True

    def test_uncertain_faces_center_crop(self):
        d = decide_layout(_analysis([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
                          {"min_face_fraction": 0.5}, "auto")
        assert d["layout"] == "center_crop"
        assert d["needs_review"] is True

    def test_manual_override_respected(self):
        for layout in SUPPORTED_LAYOUTS:
            if layout in ("auto", "branded_fit"):
                continue
            d = decide_layout(_analysis([1] * 12), {}, layout)
            assert d["layout"] == layout

    def test_branded_fit_mapped_to_auto_under_fullscreen(self):
        d = decide_layout(_analysis([1] * 12),
                          {"full_screen_vertical": True}, "branded_fit")
        assert d["layout"] == "speaker_crop"  # auto wins; never bars
        assert "full_screen_vertical" in d["reason"]

    def test_branded_fit_honoured_when_flag_off(self):
        d = decide_layout(_analysis([1] * 12),
                          {"full_screen_vertical": False}, "branded_fit")
        assert d["layout"] == "branded_fit"

    def test_invalid_layout_falls_back(self):
        assert validate_layout("blur", "auto") == "auto"
        assert validate_layout(None, "branded_fit") == "branded_fit"

    def test_layout_stability_brief_dropout(self):
        # mostly 1 face with a 2-sample dropout -> stays speaker_crop
        counts = [1] * 6 + [0, 0] + [1] * 6
        d = decide_layout(_analysis(counts), {"layout_stability_frames": 6}, "auto")
        assert d["layout"] == "speaker_crop"

    def test_fullscreen_default_on(self):
        assert fullscreen({}) is True
        assert fullscreen({"full_screen_vertical": False}) is False


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
        # after hold window -> lost (render falls back to crop chain, never bars)
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

    def test_center_crop_rect_ratio_and_inside(self):
        x, y, cw, ch = center_crop_rect(3840, 2160)
        assert abs((cw / ch) - (1080 / 1920)) < 0.02
        assert 0 <= x and 0 <= y and x + cw <= 3840 and y + ch <= 2160

    def test_center_crop_rect_respects_bounds(self):
        bounds = {"x": 0, "y": 100, "w": 1280, "h": 520}
        x, y, cw, ch = center_crop_rect(1280, 720, bounds)
        assert y >= 100 and y + ch <= 620
        assert abs((cw / ch) - (1080 / 1920)) < 0.02


class TestScenePlan:
    def test_face_then_empty_scene_retains(self):
        a = _analysis([1] * 8 + [0] * 8)
        a["samples"][8]["scene_cut"] = True
        a["scenes"] = [0.0, 2.0]
        plan = plan_scene_crops(a, {})
        assert len(plan) == 2
        assert plan[0]["kind"] == "face"
        assert plan[0]["needs_review"] is False
        # second scene keeps the established crop, flagged for review
        assert plan[1]["kind"] == "retain"
        assert plan[1]["needs_review"] is True
        assert (plan[1]["cx"], plan[1]["cy"]) == (plan[0]["cx"], plan[0]["cy"])

    def test_all_empty_scene_centre(self):
        plan = plan_scene_crops(_analysis([0] * 8), {})
        assert len(plan) == 1
        assert plan[0]["kind"] == "center"
        assert plan[0]["needs_review"] is True

    def test_workarea_preferred_over_centre(self):
        wa = {"x": 100, "y": 100, "w": 405, "h": 720}  # 9:16
        plan = plan_scene_crops(_analysis([0] * 8),
                                {"work_area": wa})
        assert plan[0]["kind"] == "workarea"


class TestManualROI:
    def test_validate_roi_ok(self):
        roi = validate_roi({"x": 100, "y": 50, "w": 540, "h": 960})
        assert roi["w"] == 540 and roi["h"] == 960

    def test_validate_roi_bad_ratio(self):
        with pytest.raises(ValueError, match="9:16"):
            validate_roi({"x": 0, "y": 0, "w": 640, "h": 480})

    def test_validate_roi_time_range(self):
        with pytest.raises(ValueError, match="t_start < t_end"):
            validate_roi({"x": 0, "y": 0, "w": 540, "h": 960,
                          "t_start": 5.0, "t_end": 2.0})

    def test_manual_roi_authoritative_over_detection(self):
        a = _analysis([1] * 12)
        sm = smooth_centres(a, {})
        plan = plan_scene_crops(a, {})
        roi = {"x": 10, "y": 20, "w": 540, "h": 960}
        anchor = resolve_anchor(plan, sm, 1.0, {}, [roi],
                                {"x": 0, "y": 0, "w": 1280, "h": 720})
        assert anchor["kind"] == "manual"
        assert anchor["needs_review"] is False
        assert (anchor["cx"], anchor["cy"]) == (280.0, 500.0)

    def test_manual_roi_time_window(self):
        a = _analysis([1] * 12)
        sm = smooth_centres(a, {})
        plan = plan_scene_crops(a, {})
        roi = {"x": 10, "y": 20, "w": 540, "h": 960,
               "t_start": 0.0, "t_end": 0.5}
        anchor = resolve_anchor(plan, sm, 2.0, {}, [roi],
                                {"x": 0, "y": 0, "w": 1280, "h": 720})
        assert anchor["kind"] == "face"  # ROI window expired


class TestContentBounds:
    def test_estimate_ignores_embedded_bars(self, tmp_path):
        import cv2
        import numpy as np
        path = str(tmp_path / "bars.mp4")
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
        frame = np.full((240, 320, 3), 128, dtype=np.uint8)
        frame[:20, :] = 0
        frame[-20:, :] = 0
        for _ in range(10):
            vw.write(frame)
        vw.release()
        bounds = estimate_content_bounds(path, threshold=8, max_samples=4)
        assert 15 <= bounds["y"] <= 25
        assert bounds["h"] >= 190
        assert bounds["x"] == 0 and bounds["w"] == 320

    def test_estimate_full_frame_without_bars(self, tmp_path):
        import cv2
        import numpy as np
        path = str(tmp_path / "full.mp4")
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
        for _ in range(10):
            vw.write(np.full((240, 320, 3), 128, dtype=np.uint8))
        vw.release()
        bounds = estimate_content_bounds(path, threshold=8, max_samples=4)
        assert bounds == {"x": 0, "y": 0, "w": 320, "h": 240}


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
