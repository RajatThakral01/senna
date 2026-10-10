# test_active_speaker.py — lip-motion scoring, voicing gate, shot director.
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from pipeline.active_speaker import (GROUP, direct_shots, mouth_motion,
                                     voiced_at, _enforce_min_shot, _runs)


def _seg(script, interval=0.1, voiced=None):
    """script: list of (seconds, {track_id: mouth_activity})."""
    seg, t = [], 0.0
    for dur, act in script:
        for _ in range(int(round(dur / interval))):
            seg.append({"t": round(t, 3), "voiced": True if voiced is None else voiced,
                        "tracks": [{"id": i, "cx": 400 + 800 * i, "cy": 300,
                                    "mouth": a} for i, a in act.items()]})
            t += interval
    return seg


class TestDirector:
    def test_cuts_follow_turns(self):
        seg = _seg([(3, {0: 0.3, 1: 0.02}), (3, {0: 0.02, 1: 0.3}),
                    (3, {0: 0.3, 1: 0.02})])
        shots = direct_shots(seg, [0, 1], {})
        assert [s["state"] for s in shots] == [0, 1, 0]
        assert abs(shots[1]["t0"] - 3.0) <= 0.3 and abs(shots[2]["t0"] - 6.0) <= 0.3

    def test_brief_interjection_no_cut(self):
        seg = _seg([(4, {0: 0.3, 1: 0.02}), (0.5, {0: 0.02, 1: 0.3}),
                    (4, {0: 0.3, 1: 0.02})])
        shots = direct_shots(seg, [0, 1], {"min_shot_seconds": 1.6})
        assert [s["state"] for s in shots] == [0]

    def test_overlap_gives_two_shot(self):
        seg = _seg([(3, {0: 0.3, 1: 0.02}), (3, {0: 0.3, 1: 0.28}),
                    (3, {0: 0.02, 1: 0.3})])
        states = [s["state"] for s in direct_shots(seg, [0, 1], {})]
        assert GROUP in states and states[0] == 0 and states[-1] == 1

    def test_nodding_listener_not_group(self):
        seg = _seg([(6, {0: 0.3, 1: 0.08})])
        assert [s["state"] for s in direct_shots(seg, [0, 1], {})] == [0]

    def test_pause_holds_current_shot(self):
        seg = (_seg([(3, {0: 0.3, 1: 0.02})])
               + _seg([(2, {0: 0.0, 1: 0.4})], voiced=False))   # B fidgets in silence
        for k, s in enumerate(seg):
            s["t"] = round(k * 0.1, 3)
        assert [s["state"] for s in direct_shots(seg, [0, 1], {})] == [0]

    def test_weak_evidence_returns_none(self):
        seg = _seg([(5, {0: 0.01, 1: 0.01})])
        assert direct_shots(seg, [0, 1], {}) is None

    def test_needs_two_people(self):
        assert direct_shots(_seg([(5, {0: 0.3})]), [0], {}) is None

    def test_min_shot_merge(self):
        cost = np.zeros((10, 2))
        runs = _enforce_min_shot(_runs(np.array([0] * 4 + [1] + [0] * 5)), cost, 3)
        assert runs == [[0, 10, 0]]


class TestSignals:
    def _face(self):
        return {"x1": 100, "y1": 100, "x2": 200, "y2": 220}

    def test_mouth_motion_detects_lips_not_still_face(self):
        rng = np.random.default_rng(0)
        a = rng.integers(60, 200, (400, 400)).astype(np.uint8)
        b = a.copy()
        assert mouth_motion(a, b, self._face()) == 0.0
        b[180:215, 130:170] = 20          # mouth opens
        assert mouth_motion(a, b, self._face()) > 0.2

    def test_head_motion_compensated(self):
        rng = np.random.default_rng(1)
        a = rng.integers(60, 200, (400, 400)).astype(np.uint8)
        b = np.roll(a, 3, axis=0)        # whole head shifts (nod)
        lips = a.copy(); lips[180:215, 130:170] = 20
        assert mouth_motion(a, b, self._face()) < mouth_motion(a, lips, self._face())

    def test_no_audio_means_voiced(self):
        assert voiced_at(None, [0, 0.1]).all()

    def test_voicing_threshold(self):
        t = np.arange(100) * 0.1
        db = np.where((t > 2) & (t < 5), -20.0, -60.0)
        v = voiced_at((t, db), [1.0, 3.0, 7.0])
        assert list(v) == [False, True, False]


class TestRenderPlanDirector:
    def _analysis(self):
        seg = _seg([(3, {0: 0.3, 1: 0.02}), (3, {0: 0.02, 1: 0.3})])
        for s in seg:
            for tr in s["tracks"]:
                tr["box"] = {"x1": tr["cx"] - 80, "x2": tr["cx"] + 80}
            s["n"] = 2; s["scene_cut"] = False
        return {"samples": seg, "scenes": [0.0], "width": 1920, "height": 1080,
                "fps": 25.0, "duration": 6.0,
                "content_bounds": {"x": 0, "y": 0, "w": 1920, "h": 1080}}

    def test_auto_cuts_between_speakers(self):
        from pipeline.framing import _render_plan, plan_scene_crops
        a = self._analysis()
        plans = _render_plan(a, plan_scene_crops(a, {}), {}, 25.0, 150,
                             "stacked_split", a["content_bounds"], requested="auto")
        assert [(p["mode"], p["track_id"]) for p in plans] == [("single", 0), ("single", 1)]
        assert plans[0]["f1"] == plans[1]["f0"] and plans[-1]["f1"] == 150
        assert plans[0]["cx"][0] < 960 < plans[1]["cx"][0]

    def test_explicit_stacked_respected(self):
        from pipeline.framing import _render_plan, plan_scene_crops
        a = self._analysis()
        plans = _render_plan(a, plan_scene_crops(a, {}), {}, 25.0, 150,
                             "stacked_split", a["content_bounds"],
                             requested="stacked_split")
        assert [p["mode"] for p in plans] == ["stacked"]

    def test_director_off(self):
        from pipeline.framing import _render_plan, plan_scene_crops
        a = self._analysis()
        plans = _render_plan(a, plan_scene_crops(a, {}), {"active_speaker": False},
                             25.0, 150, "stacked_split", a["content_bounds"],
                             requested="auto")
        assert [p["mode"] for p in plans] == ["stacked"]


class TestCutSnap:
    def test_cut_moves_to_quietest_point(self):
        from pipeline.active_speaker import _snap_cuts
        seg = [{"t": k * 0.1, "db": -20.0} for k in range(40)]
        seg[23]["db"] = -50.0                              # breath 0.3s after the cut
        runs = _snap_cuts([[0, 20, 0], [20, 40, 1]], seg, 4)
        assert runs == [[0, 23, 0], [23, 40, 1]]

    def test_cut_never_crosses_neighbour_middle(self):
        from pipeline.active_speaker import _snap_cuts
        seg = [{"t": k * 0.1, "db": -20.0} for k in range(30)]
        seg[11]["db"] = -50.0                              # too far into the short first shot
        runs = _snap_cuts([[0, 14, 0], [14, 30, 1]], seg, 4)
        assert runs[0][1] > 7 and runs[0][1] == runs[1][0]

    def test_no_audio_keeps_cuts(self):
        from pipeline.active_speaker import _snap_cuts
        seg = [{"t": k * 0.1} for k in range(30)]
        assert _snap_cuts([[0, 14, 0], [14, 30, 1]], seg, 4) == [[0, 14, 0], [14, 30, 1]]


class TestShortScenes:
    def test_scene_shorter_than_smoothing_window(self):
        # fast-cut footage: 3-sample scene vs 5-sample window crashed with
        # "boolean index did not match" (np.convolve "same" grows the series)
        from pipeline.active_speaker import _nan_smooth, speaker_scores
        import numpy as np
        out = _nan_smooth(np.array([0.1, np.nan, 0.3]), 5)
        assert out.shape == (3,)
        seg = _seg([(0.3, {0: 0.3, 1: 0.02})])
        a, voiced = speaker_scores(seg, [0, 1], {})
        assert a.shape == (3, 2)
        direct_shots(seg, [0, 1], {})          # must not raise

    def test_every_tiny_length(self):
        from pipeline.active_speaker import direct_shots
        for n in range(1, 8):
            seg = _seg([(n * 0.1, {0: 0.3, 1: 0.02})])
            direct_shots(seg, [0, 1], {})
