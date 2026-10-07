# test_review.py — Phase 6: validation, overrides, caption rules.
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from pipeline.review import (validate_boundary, apply_boundary_edit,
                             apply_layout_override, apply_caption_text,
                             set_candidate_status, apply_crop_override,
                             get_manual_rois)


@pytest.fixture()
def clip_row():
    from db.repositories import video_repo, clip_repo
    vid = video_repo.insert_video("test://phase6", "test://phase6")
    cid = clip_repo.insert_clip(vid, 1, 10.0, 40.0, duration_seconds=30.0,
                                hook="h")
    yield vid, cid
    from conftest import cleanup_video
    cleanup_video(vid)


class TestValidate:
    def test_ok(self):
        assert validate_boundary("10", "40", 100) == (10.0, 40.0)

    def test_order(self):
        with pytest.raises(ValueError):
            validate_boundary(40, 10)

    def test_min_duration(self):
        with pytest.raises(ValueError):
            validate_boundary(10, 10.5)

    def test_beyond_video(self):
        with pytest.raises(ValueError):
            validate_boundary(10, 200, 100)

    def test_non_numeric(self):
        with pytest.raises(ValueError):
            validate_boundary("a", "b")


class TestOverrides:
    def test_boundary_override(self, clip_row):
        vid, cid = clip_row
        rng = apply_boundary_edit(cid, 12.0, 35.0, video_duration=100)
        assert rng == [12.0, 35.0]
        from db.repositories import clip_repo
        c = [x for x in clip_repo.get_clips_for_video(vid)][0]
        assert c["refine_status"] == "manual"
        assert c["start_time"] == 12.0

    def test_boundary_multi_range_refused(self, clip_row):
        from db.connection import get_conn, release_conn
        import json as _json
        vid, cid = clip_row
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("UPDATE clips SET source_ranges=%s WHERE id=%s",
                            (_json.dumps([[1, 2], [5, 6]]), cid))
                conn.commit()
        finally:
            release_conn(conn)
        with pytest.raises(ValueError, match="multi-range"):
            apply_boundary_edit(cid, 1.0, 6.0, 100)

    def test_layout_override(self, clip_row):
        vid, cid = clip_row
        assert apply_layout_override(cid, "branded_fit") == "branded_fit"
        with pytest.raises(Exception):
            apply_layout_override(cid, "nope-never")

    def test_candidate_status(self, clip_row):
        from db.repositories import candidate_repo
        vid, cid = clip_row
        cand = candidate_repo.insert_candidate(vid, "outline", [0],
                                               [[0.0, 1.0]], hook="h")
        assert set_candidate_status(cand, "rejected", "boring") is True
        rows = candidate_repo.get_candidates(vid, status="rejected")
        assert len(rows) == 1
        with pytest.raises(ValueError):
            set_candidate_status(cand, "bogus")

    def test_caption_word_count_rule(self, clip_row):
        from db.repositories import editplan_repo
        vid, cid = clip_row
        editplan_repo.save_plan(vid, cid, {
            "clip_number": 1,
            "captions": {"placement": "bottom", "cues": [
                {"start": 0.0, "end": 1.0,
                 "words": [{"word": "hello", "display": "hello",
                            "start": 0.0, "end": 0.5},
                           {"word": "world", "display": "world",
                            "start": 0.5, "end": 1.0}],
                 "text": "hello world"}]}})
        assert apply_caption_text(cid, ["hi there"]) == 1
        with pytest.raises(ValueError, match="word count"):
            apply_caption_text(cid, ["too many words here"])
        # plan version bumped twice (save + apply)
        assert editplan_repo.get_latest_plan(cid)["version"] == 2

    def test_caption_no_plan(self, clip_row):
        vid, cid = clip_row
        with pytest.raises(ValueError, match="no edit plan"):
            apply_caption_text(cid, ["x"])


class TestSelectiveRender:
    def test_render_clips_accepts_filter(self):
        import inspect
        from main import render_clips
        assert "clip_numbers" in inspect.signature(render_clips).parameters


class TestCropOverride:
    def _plan(self, vid, cid):
        from db.repositories import editplan_repo
        editplan_repo.save_plan(vid, cid, {
            "clip_number": 1, "framing": {},
            "captions": {"placement": "bottom", "cues": []}})

    def test_crop_override_ok(self, clip_row):
        vid, cid = clip_row
        self._plan(vid, cid)
        roi = apply_crop_override(cid, 100, 50, 540, 960)
        assert roi["x"] == 100 and roi["w"] == 540
        assert get_manual_rois(cid) == [roi]

    def test_crop_override_bad_ratio(self, clip_row):
        vid, cid = clip_row
        self._plan(vid, cid)
        with pytest.raises(ValueError, match="9:16"):
            apply_crop_override(cid, 0, 0, 640, 480)

    def test_crop_override_time_range(self, clip_row):
        vid, cid = clip_row
        self._plan(vid, cid)
        roi = apply_crop_override(cid, 0, 0, 540, 960,
                                  t_start=1.0, t_end=5.0)
        assert roi["t_start"] == 1.0 and roi["t_end"] == 5.0
        # same window overwritten, not duplicated
        apply_crop_override(cid, 10, 10, 540, 960,
                            t_start=1.0, t_end=5.0)
        assert len(get_manual_rois(cid)) == 1

    def test_crop_no_plan(self, clip_row):
        vid, cid = clip_row
        with pytest.raises(ValueError, match="no edit plan"):
            apply_crop_override(cid, 0, 0, 540, 960)
