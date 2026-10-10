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


# ── Phase A: detection fix, persistent tracking, virtual camera, panels ──────

class TestDetectionGeometry:
    def test_mediapipe_box_bottom_is_origin_plus_height(self, monkeypatch):
        # regression: y2 used to be bb.height (not origin_y + height), which
        # silently dropped every face whose top edge sat below its height
        from types import SimpleNamespace as NS
        import numpy as np
        from pipeline import framing

        class Det:
            def detect(self, img):
                bb = NS(origin_x=800, origin_y=400, width=150, height=150)
                return NS(detections=[NS(bounding_box=bb, categories=[NS(score=0.9)])])
        fake_mp = NS(Image=lambda **kw: None, ImageFormat=NS(SRGB=1))
        monkeypatch.setitem(sys.modules, "mediapipe", fake_mp)
        boxes = framing._detect_mediapipe(Det(), np.zeros((1080, 1920, 3), np.uint8))
        assert len(boxes) == 1
        assert boxes[0]["y2"] == 550 and boxes[0]["cy"] == 475

    def test_min_face_ratio_filters_tiny_faces(self, monkeypatch):
        import numpy as np
        from pipeline import framing
        monkeypatch.setattr(framing, "get_yunet", lambda cfg=None: object())
        monkeypatch.setattr(framing, "_detect_yunet", lambda d, f, c: [
            framing._box(0, 0, 10, 10), framing._box(100, 100, 300, 300)])
        boxes = framing.detect_faces_bgr(np.zeros((1080, 1920, 3), np.uint8),
                                         {"min_face_ratio": 0.02})
        assert len(boxes) == 1 and boxes[0]["x1"] == 100


class TestPersistentTracker:
    def _b(self, cx, w=100):
        from pipeline.framing import _box
        return _box(cx - w / 2, 300, cx + w / 2, 400)

    def test_identity_survives_dropout(self):
        from pipeline.framing import FaceTracker
        tr = FaceTracker(1920, max_missing=1.5)
        a = tr.update([self._b(500), self._b(1400)], 0.0)
        ids = {round(t["cx"]): t["id"] for t in a}
        tr.update([self._b(1400)], 0.1)            # left face missed
        tr.update([self._b(1400)], 0.5)
        back = tr.update([self._b(510), self._b(1400)], 0.9)
        got = {round(t["cx"]): t["id"] for t in back}
        assert got[510] == ids[500] and got[1400] == ids[1400]

    def test_far_jump_is_new_identity(self):
        from pipeline.framing import FaceTracker
        tr = FaceTracker(1920)
        first = tr.update([self._b(300)], 0.0)[0]["id"]
        assert tr.update([self._b(1600)], 0.1)[0]["id"] != first

    def test_reset_on_scene_cut(self):
        from pipeline.framing import FaceTracker
        tr = FaceTracker(1920)
        first = tr.update([self._b(300)], 0.0)[0]["id"]
        tr.reset()
        assert tr.update([self._b(300)], 0.1)[0]["id"] != first


class TestVirtualCamera:
    def test_deadzone_locks_camera_for_small_motion(self):
        import math
        import numpy as np
        from pipeline.framing import plan_camera_axis
        ts = [i * 0.1 for i in range(50)]
        xs = [960 + 20 * math.sin(t * 6) for t in ts]   # head sway << deadzone
        path = plan_camera_axis(ts, xs, np.arange(125) / 25, {}, span=607)
        assert np.ptp(path) < 1.0                          # locked shot

    def test_walk_is_smooth_and_followed(self):
        import numpy as np
        from pipeline.framing import plan_camera_axis
        ts = [i * 0.1 for i in range(60)]
        xs = [600 if t < 2 else (600 + 350 * (t - 2) if t < 4 else 1300) for t in ts]
        out_ts = np.arange(150) / 25
        path = plan_camera_axis(ts, xs, out_ts, {}, span=607, frame_w=1920)
        acc = np.abs(np.diff(path, 2))
        assert acc.max() < 2.0                             # no velocity steps
        assert abs(path[-1] - 1300) < 1300 * 0.15 * 0.5     # ends on subject

    def test_outlier_detection_ignored(self):
        import numpy as np
        from pipeline.framing import plan_camera_axis
        ts = [i * 0.1 for i in range(30)]
        xs = [900] * 30
        xs[15] = 1500                                      # one bad detection
        path = plan_camera_axis(ts, xs, np.arange(75) / 25, {}, span=607)
        assert np.ptp(path) < 1.0

    def test_speed_limit(self):
        import numpy as np
        from pipeline.framing import plan_camera_axis
        ts = [i * 0.1 for i in range(40)]
        xs = [300] * 20 + [1600] * 20
        path = plan_camera_axis(ts, xs, np.arange(100) / 25,
                                {"max_pan_speed": 0.5}, span=607, frame_w=1920)
        assert np.abs(np.diff(path)).max() <= 0.5 * 1920 / 25 + 1e-6

    def test_never_seen_returns_none(self):
        from pipeline.framing import plan_camera_axis
        assert plan_camera_axis([0, 0.1], [None, None], [0, 0.04], {}, 607) is None


class TestRenderPlan:
    def test_stacked_panels_keep_panel_aspect(self):
        from pipeline.framing import _render_plan, plan_scene_crops, OUT_W
        a = _analysis([2] * 40, w=1920, h=1080)
        for s in a["samples"]:
            for tr in s["tracks"]:
                tr["box"] = {"x1": tr["cx"] - 100, "x2": tr["cx"] + 100}
        a["content_bounds"] = {"x": 0, "y": 0, "w": 1920, "h": 1080}
        cfg = {"split_divider_px": 6}
        plans = _render_plan(a, plan_scene_crops(a, cfg), cfg, 25.0, 250,
                             "stacked_split", a["content_bounds"])
        p = plans[0]
        assert p["mode"] == "stacked"
        top_h = p["heights"][0]
        assert abs(p["pcw"] / p["pch"] - OUT_W / top_h) < 1e-6   # no squash

    def test_single_person_scene_in_stacked_clip_uses_single_crop(self):
        from pipeline.framing import _render_plan, plan_scene_crops
        a = _analysis([2] * 20 + [1] * 20, w=1920, h=1080)
        a["samples"][20]["scene_cut"] = True
        a["scenes"] = [0.0, 5.0]
        a["content_bounds"] = {"x": 0, "y": 0, "w": 1920, "h": 1080}
        plans = _render_plan(a, plan_scene_crops(a, {}), {}, 25.0, 250,
                             "stacked_split", a["content_bounds"])
        assert [p["mode"] for p in plans] == ["stacked", "single"]
        assert plans[1]["kind"] == "face"


# ── Phase C: speaker zoom, emphasis punch-in, debug video ───────────────────

class TestShotStyling:
    def test_auto_zoom_frames_by_face(self):
        from pipeline.framing import _shot_crop_size, OUT_W, OUT_H
        cw, ch = _shot_crop_size(3840, 2160, {"target_face_ratio": 0.33}, 400)
        assert abs(cw - 400 / 0.33) < 1 and abs(cw / ch - OUT_W / OUT_H) < 1e-6

    def test_auto_zoom_bounded(self):
        from pipeline.framing import _shot_crop_size
        # tiny face: limited by max_upscale, never a postage stamp
        cw, _ = _shot_crop_size(1920, 1080, {"max_upscale": 2.4}, 40)
        assert abs(cw - 1080 / 2.4) < 1
        # huge face: never wider than the full-height crop
        cw, ch = _shot_crop_size(1920, 1080, {}, 900)
        assert ch == 1080
        # off / unknown face: full-height crop
        assert _shot_crop_size(1920, 1080, {"auto_zoom": False}, 200)[1] == 1080
        assert _shot_crop_size(1920, 1080, {}, 0)[1] == 1080

    def _seg(self, dbs, voiced=True):
        return [{"t": round(k * 0.1, 3), "db": d, "voiced": voiced, "tracks": []}
                for k, d in enumerate(dbs)]

    def test_emphasis_punch_in_on_peak(self):
        import numpy as np
        from pipeline.framing import emphasis_zoom
        dbs = [-30.0] * 100
        dbs[50] = -18.0                                   # one emphatic peak at 5.0s
        out_ts = np.arange(250) / 25
        z = emphasis_zoom(self._seg(dbs), out_ts, {"emphasis_zoom": 0.06})
        assert abs(z.max() - 1.06) < 1e-6
        assert 4.6 < out_ts[np.argmax(z)] < 6.5
        assert z[0] == 1.0 and z[-1] == 1.0
        assert np.abs(np.diff(z)).max() < 0.03           # eased, not a jump

    def test_emphasis_off_or_flat(self):
        import numpy as np
        from pipeline.framing import emphasis_zoom
        out_ts = np.arange(250) / 25
        assert (emphasis_zoom(self._seg([-30.0] * 100), out_ts, {}) == 1).all()
        dbs = [-30.0] * 100; dbs[50] = -18.0
        assert (emphasis_zoom(self._seg(dbs), out_ts, {"emphasis_zoom": 0}) == 1).all()
        assert (emphasis_zoom(self._seg(dbs, voiced=False), out_ts, {}) == 1).all()

    def test_no_punch_near_shot_edges(self):
        import numpy as np
        from pipeline.framing import emphasis_zoom
        dbs = [-30.0] * 100; dbs[2] = -18.0; dbs[97] = -18.0
        z = emphasis_zoom(self._seg(dbs), np.arange(250) / 25, {})
        assert (z == 1).all()


def _has_ffmpeg():
    import shutil
    return shutil.which("ffmpeg") is not None


@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not installed")
class TestRenderSmoke:
    def test_render_with_debug_video(self, tmp_path):
        import cv2
        import numpy as np
        from pipeline.framing import render_vertical
        src = str(tmp_path / "src.mp4")
        vw = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 360))
        for i in range(25):
            vw.write(np.full((360, 640, 3), 90 + i, dtype=np.uint8))
        vw.release()
        out = str(tmp_path / "out.mp4")
        r = render_vertical(src, out, cfg={"face_backend": "haar"},
                            debug_path=str(tmp_path / "dbg.mp4"))
        assert r["frames"] == 25 and os.path.getsize(out) > 0
        assert r["debug_path"] and os.path.getsize(r["debug_path"]) > 0
        cap = cv2.VideoCapture(out)
        assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (1080, 1920)
        cap.release()


class TestCutDetector:
    """Hard cuts vs. fast motion (whip pans) on adjacent frames."""

    def _frames(self, kind):
        import numpy as np
        rng = np.random.default_rng(0)
        a = rng.integers(0, 255, (90, 160, 3)).astype(np.uint8)
        a[:, :, 2] = 40                                    # scene A: dark reds
        b = rng.integers(0, 255, (90, 160, 3)).astype(np.uint8)
        b[:, :, 0] = 230; b[:, :, 2] = 20                  # scene B: different colours
        if kind == "cut":
            return [a] * 5 + [b] * 5
        # whip pan: same scene sliding a little more each frame
        return [np.roll(a, 8 * k * k, axis=1) for k in range(10)]

    def test_hard_cut_detected_once(self):
        from pipeline.framing import _CutDetector
        d = _CutDetector({})
        hits = [i for i, f in enumerate(self._frames("cut")) if d.update(f, i / 30)]
        assert hits == [5]

    def test_whip_pan_is_not_a_cut(self):
        from pipeline.framing import _CutDetector
        d = _CutDetector({})
        assert not any(d.update(f, i / 30) for i, f in enumerate(self._frames("pan")))

    def test_min_gap_merges_flash_cuts(self):
        import numpy as np
        from pipeline.framing import _CutDetector
        d = _CutDetector({"min_scene_seconds": 0.3})
        f = self._frames("cut")
        seq = f[:5] + f[5:6] + f[:1] + f[5:]               # cut, cut back, cut again
        hits = [i for i, x in enumerate(seq) if d.update(x, i / 30)]
        assert hits == [5]


# ── Phase D: eligibility, prominence, crowds, sliced faces ───────────────────

def _tr(i, cx, w, cy=400, mouth=None):
    return {"id": i, "cx": cx, "cy": cy, "mouth": mouth,
            "box": {"x1": cx - w / 2, "x2": cx + w / 2, "y1": cy - w / 2, "y2": cy + w / 2,
                    "cx": cx, "cy": cy}}


class TestPhaseDEligibility:
    def test_small_faces_not_eligible_for_speaker(self):
        from pipeline.framing import _eligible_people
        seg = [{"t": k * 0.1, "tracks": [_tr(0, 900, 200), _tr(1, 300, 40)]} for k in range(10)]
        assert _eligible_people(seg, 0.2) == [0, 1] or set(_eligible_people(seg, 0.2)) == {0, 1}
        assert _eligible_people(seg, 0.2, 1920, 0.045) == [0]

    def test_edge_clipped_faces_not_eligible(self):
        from pipeline.framing import _eligible_people
        seg = [{"t": k * 0.1, "tracks": [_tr(0, 960, 200), _tr(1, 60, 120)]} for k in range(10)]
        seg_tracks = seg[0]["tracks"][1]["box"]
        seg_tracks["x1"] = 0.0                      # face cut by the frame edge
        for s in seg:
            s["tracks"][1]["box"]["x1"] = 0.0
        assert _eligible_people(seg, 0.2, 1920, 0.045) == [0]

    def test_prominence_prefers_big_central_face(self):
        from pipeline.framing import _scene_subject
        seg = [{"t": k * 0.1, "tracks": [_tr(0, 150, 120), _tr(1, 960, 110)]} for k in range(10)]
        # similar size, but id1 is centred: the operator framed id1
        assert _scene_subject(seg, 1920) == 1


class TestPhaseDDirector:
    def _seg(self, n, acts, voiced=True):
        return [{"t": round(k * 0.1, 3), "voiced": voiced,
                 "tracks": [{"id": i, "cx": 400 + 600 * i, "cy": 300, "mouth": a}
                            for i, a in acts.items()]} for k in range(n)]

    def test_no_clear_speaker_returns_none(self):
        from pipeline.active_speaker import direct_shots
        # everyone moves their mouth a bit, nobody clearly talks (narration/chewing)
        seg = self._seg(30, {0: 0.12, 1: 0.11, 2: 0.12})
        assert direct_shots(seg, [0, 1, 2], {}) is None

    def test_two_shot_only_with_exactly_two_people(self):
        from pipeline.active_speaker import direct_shots, GROUP
        seg = self._seg(40, {0: 0.3, 1: 0.29, 2: 0.01})
        for s in seg[:10]:
            s["tracks"][1]["mouth"] = 0.01          # a clear solo first
        states = [s["state"] for s in direct_shots(seg, [0, 1, 2], {})]
        assert GROUP not in states

    def test_prior_breaks_silence_toward_prominent(self):
        from pipeline.active_speaker import direct_shots
        seg = self._seg(10, {0: 0.01, 1: 0.01}, voiced=True)
        seg += self._seg(10, {0: 0.01, 1: 0.4})     # id1 clearly talks briefly
        for k, s in enumerate(seg):
            s["t"] = round(k * 0.1, 3)
        shots = direct_shots(seg, [0, 1], {"min_shot_seconds": 3.0}, prior={0: 1.0, 1: 0.1})
        assert shots is not None and len(shots) == 1


class TestPhaseDGeometry:
    def test_group_target_keeps_faces_whole(self):
        from pipeline.framing import _group_target, _overlap_frac
        boxes = [_tr(i, cx, 80)["box"] for i, cx in enumerate([500, 640, 780, 1500])]
        cw = 607
        c = _group_target(boxes, cw, cw / 2, 1920 - cw / 2)
        left, right = c - cw / 2, c + cw / 2
        fr = [_overlap_frac(b, left, right) for b in boxes]
        assert sum(1 for f in fr if f >= 0.95) == 3     # the cluster of three
        assert not any(0.05 < f < 0.95 for f in fr)     # nobody sliced

    def test_deslice_moves_crop_off_neighbour(self):
        from pipeline.framing import _deslice_target, _sliced_weight
        cw = 600
        subj = _tr(0, 900, 150)["box"]
        other = _tr(1, 1200, 120)["box"]                # straddles the right edge at cx=900
        assert _sliced_weight([other], 900 - cw / 2, 900 + cw / 2) > 0
        c = _deslice_target(900, cw, subj, [other], cw / 2, 1920 - cw / 2, 0.25 * cw)
        assert _sliced_weight([other], c - cw / 2, c + cw / 2) == 0
        assert subj["x1"] >= c - cw / 2 and subj["x2"] <= c + cw / 2

    def test_crowd_scene_with_weak_speaker_uses_group_shot(self, monkeypatch):
        import pipeline.active_speaker as A
        from pipeline.framing import _render_plan, plan_scene_crops
        seg = []
        for k in range(30):
            seg.append({"t": round(k * 0.1, 3), "scene_cut": False, "voiced": True,
                        "tracks": [_tr(i, 500 + 250 * i, 100, mouth=0.1) for i in range(4)]})
            seg[-1]["n"] = 4
        a = {"samples": seg, "scenes": [0.0], "width": 1920, "height": 1080, "fps": 25.0,
             "duration": 3.0, "content_bounds": {"x": 0, "y": 0, "w": 1920, "h": 1080}}
        monkeypatch.setattr(A, "direct_shots", lambda *a, **k: [
            {"t0": 0.0, "t1": None, "state": 2, "score": -0.4}])
        plans = _render_plan(a, plan_scene_crops(a, {}), {}, 25.0, 75, "stacked_split",
                             a["content_bounds"], requested="auto")
        assert [p["kind"] for p in plans] == ["group"]


class TestSaliencyFallback:
    def test_salient_cx_finds_the_object(self):
        import numpy as np
        from pipeline.framing import salient_cx
        img = np.full((360, 640, 3), 120, np.uint8)
        img[140:220, 480:560] = (0, 0, 255)                # a red product on the right
        img[150:210, 490:550] = 255
        cx = salient_cx(img)
        assert cx is not None and 430 < cx < 610

    def test_faceless_scene_follows_saliency(self):
        from pipeline.framing import _render_plan, plan_scene_crops
        seg = [{"t": round(k * 0.1, 3), "scene_cut": False, "n": 0, "tracks": [],
                "boxes": [], "sal_cx": 1500.0} for k in range(30)]
        a = {"samples": seg, "scenes": [0.0], "width": 1920, "height": 1080, "fps": 25.0,
             "duration": 3.0, "content_bounds": {"x": 0, "y": 0, "w": 1920, "h": 1080}}
        plans = _render_plan(a, plan_scene_crops(a, {}), {}, 25.0, 75, "center_crop",
                             a["content_bounds"], requested="auto")
        assert plans[0]["kind"] == "salient"
        assert abs(plans[0]["cx"][-1] - 1500) < 5

    def test_long_face_gap_uses_saliency(self):
        import numpy as np
        from pipeline.framing import track_path
        seg = []
        for k in range(40):
            tracks = [_tr(0, 600, 150)] if k < 10 else []
            seg.append({"t": round(k * 0.1, 3), "tracks": tracks, "sal_cx": 1400.0})
        b = {"x": 0, "y": 0, "w": 1920, "h": 1080}
        px, _ = track_path(seg, 0, np.arange(100) / 25, {}, b, 607, 1080, 1920)
        assert px[0] < 700 and px[-1] > 1300


class TestTrackPathNeverSeen:
    def test_subject_absent_in_shot_returns_none(self):
        # regression: saliency filled x for a subject never seen in the shot,
        # y stayed None and _frame_rects crashed ('NoneType' not subscriptable)
        import numpy as np
        from pipeline.framing import track_path
        seg = [{"t": round(k * 0.1, 3), "tracks": [_tr(1, 900, 150)], "sal_cx": 1200.0}
               for k in range(30)]
        b = {"x": 0, "y": 0, "w": 1920, "h": 1080}
        assert track_path(seg, 0, np.arange(75) / 25, {}, b, 607, 1080, 1920) is None

    def test_render_plan_shot_with_absent_subject_falls_back(self, monkeypatch):
        import pipeline.active_speaker as A
        from pipeline.framing import _render_plan, plan_scene_crops, _frame_rects
        seg = []
        for k in range(30):
            tr = [_tr(0, 700, 150, mouth=0.3), _tr(1, 1300, 150, mouth=0.02)]
            if k >= 15:
                tr = tr[:1]                       # id1 leaves the frame
            seg.append({"t": round(k * 0.1, 3), "scene_cut": False, "voiced": True,
                        "tracks": tr, "n": len(tr), "sal_cx": 1000.0})
        a = {"samples": seg, "scenes": [0.0], "width": 1920, "height": 1080, "fps": 25.0,
             "duration": 3.0, "content_bounds": {"x": 0, "y": 0, "w": 1920, "h": 1080}}
        monkeypatch.setattr(A, "direct_shots", lambda *a, **k: [
            {"t0": 0.0, "t1": 1.5, "state": 0, "score": 0.5},
            {"t0": 1.5, "t1": None, "state": 1, "score": 0.5}])   # id1 absent in shot 2
        plans = _render_plan(a, plan_scene_crops(a, {}), {}, 25.0, 75, "speaker_crop",
                             a["content_bounds"], requested="auto")
        for p in plans:
            for k in range(p["f1"] - p["f0"]):
                _frame_rects(p, k, a["content_bounds"], None, 1920, 1080)  # must not raise


class TestStackedSafety:
    def _a(self, seg):
        return {"samples": seg, "scenes": [0.0], "width": 1920, "height": 1080, "fps": 25.0,
                "duration": len(seg) / 10.0,
                "content_bounds": {"x": 0, "y": 0, "w": 1920, "h": 1080}}

    def test_no_split_when_second_face_only_appears_late(self):
        # regression: close-up of one person; a second face shows up only at
        # the end -> split-screen showed the same face in both panels
        from pipeline.framing import _render_plan, plan_scene_crops
        seg = []
        for k in range(14):
            tr = [_tr(3, 1150, 300)] + ([_tr(4, 1200, 140)] if k >= 8 else [])
            seg.append({"t": round(k * 0.1, 3), "scene_cut": False, "tracks": tr,
                        "n": len(tr), "sal_cx": None})
        a = self._a(seg)
        plans = _render_plan(a, plan_scene_crops(a, {}), {"active_speaker": False}, 25.0, 35,
                             "stacked_split", a["content_bounds"], requested="auto")
        assert all(p["mode"] != "stacked" for p in plans)

    def test_split_kept_for_two_people_side_by_side(self):
        from pipeline.framing import _render_plan, plan_scene_crops
        seg = [{"t": round(k * 0.1, 3), "scene_cut": False, "n": 2, "sal_cx": None,
                "tracks": [_tr(0, 500, 160), _tr(1, 1400, 160)]} for k in range(30)]
        a = self._a(seg)
        plans = _render_plan(a, plan_scene_crops(a, {}), {"active_speaker": False}, 25.0, 75,
                             "stacked_split", a["content_bounds"], requested="auto")
        assert [p["mode"] for p in plans] == ["stacked"]


class TestSaliencyV2:
    def test_two_clusters_pick_one_not_the_gap(self):
        import numpy as np
        from pipeline.framing import salient_cx
        img = np.full((360, 640, 3), 90, np.uint8)
        img[150:230, 40:120] = (255, 40, 40)              # cluster left
        img[150:230, 520:600] = (40, 40, 255)             # cluster right
        com = salient_cx(img)                              # centre of mass: the gap
        win = salient_cx(img, win_frac=0.316)
        assert 220 < com < 420
        assert win < 200 or win > 440                      # on a cluster, not the gap

    def test_motion_beats_static_bright_object(self):
        import numpy as np
        import cv2
        from pipeline.framing import salient_cx
        prev = np.full((360, 640, 3), 90, np.uint8)
        prev[60:100, 480:600] = 255                        # bright static "lights"
        cur = prev.copy()
        prev[200:280, 100:160] = (30, 30, 30)              # player moves
        cur[200:280, 130:190] = (30, 30, 30)
        g0 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        assert salient_cx(cur, g0, 0.316) < 320
