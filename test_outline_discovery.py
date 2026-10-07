# test_outline_discovery.py — Phase 1: outline, discovery validation, fingerprints.
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from pipeline.boundaries import build_sentences
from pipeline.outline import _window_sentences, build_outline
from pipeline.discovery import validate_candidate, generate_candidates
from pipeline.fingerprints import (compute_fingerprint, outline_fingerprint,
                                   discovery_fingerprint)


def _words(texts, start=0.0, step=0.4, dur=0.35):
    words, t = [], start
    for w in texts:
        words.append({"word": w, "start": round(t, 3), "end": round(t + dur, 3)})
        t += step
    return words


def _sents(n=10):
    texts = []
    for i in range(n):
        texts += [f"w{i}a", f"w{i}b", f"w{i}c."]
    return build_sentences(_words(texts))


class TestWindows:
    def test_cover_all_sentences(self):
        sents = _sents(10)
        wins = _window_sentences(sents, window_size=4, overlap=1)
        covered = {s["id"] for w in wins for s in w}
        assert covered == {s["id"] for s in sents}
        # windows ordered and overlapping
        assert wins[0][0]["id"] == 0

    def test_empty(self):
        assert _window_sentences([], 40) == []


class TestDeterministicOutline:
    def test_no_key_gives_deterministic(self, monkeypatch):
        import pipeline.outline as O
        monkeypatch.setattr(O, "_llm_section_outline", lambda *a, **k: None)
        sents = _sents(10)
        ol = build_outline([], sents, cfg={"outline": {}})
        assert ol["method"] == "deterministic"
        assert len(ol["sections"]) >= 1
        # sections cover the transcript in order
        assert ol["sections"][0]["start_id"] == 0
        assert ol["sections"][-1]["end_id"] == sents[-1]["id"]

    def test_empty_sentences(self):
        ol = build_outline([], [], cfg={})
        assert ol["method"] == "empty"
        assert ol["sections"] == []


class TestValidateCandidate:
    def _by_id(self):
        sents = _sents(6)
        return {s["id"]: s for s in sents}, sents

    def test_valid_span_derives_times(self):
        by_id, sents = self._by_id()
        cand, reason = validate_candidate(
            {"start_id": 1, "end_id": 2, "hook": "h"}, by_id,
            min_span_seconds=0)
        assert reason == ""
        assert cand["source_ranges"][0][0] == sents[1]["start_time"]
        assert cand["source_ranges"][0][1] == sents[2]["end_time"]

    def test_reversed_ids_ordered(self):
        by_id, _ = self._by_id()
        cand, _ = validate_candidate({"start_id": 3, "end_id": 1}, by_id,
            min_span_seconds=0)
        assert cand["sentence_ids"] == [1, 2, 3]

    def test_unknown_ids_dropped(self):
        by_id, _ = self._by_id()
        cand, reason = validate_candidate({"start_id": 999, "end_id": 1005}, by_id)
        assert cand is None and "no known sentence" in reason

    def test_missing_ids_rejected(self):
        by_id, _ = self._by_id()
        cand, reason = validate_candidate({"hook": "x"}, by_id)
        assert cand is None

    def test_joins_ignored_when_disabled(self):
        by_id, _ = self._by_id()
        cand, _ = validate_candidate(
            {"start_id": 0, "end_id": 1, "join_ids": [4, 5]}, by_id,
            allow_noncontiguous=False, min_span_seconds=0)
        assert cand["joins"] == []
        assert len(cand["source_ranges"]) == 1

    def test_joins_validated_when_enabled(self):
        by_id, _ = self._by_id()
        cand, _ = validate_candidate(
            {"start_id": 0, "end_id": 1, "join_ids": [4, 5]}, by_id,
            allow_noncontiguous=True, min_span_seconds=0)
        assert cand["joins"] == [4, 5]
        assert len(cand["source_ranges"]) == 2

    def test_span_bounds_and_distant_join(self):
        by_id, _ = self._by_id()
        cand, reason = validate_candidate(
            {"start_id": 0, "end_id": 1}, by_id,
            min_span_seconds=100.0)
        assert cand is None and "too short" in reason
        cand, reason = validate_candidate(
            {"start_id": 0, "end_id": 1}, by_id,
            min_span_seconds=0, max_span_seconds=0.01)
        assert cand is None and "too long" in reason
        # force a >60s gap by patching times
        by_id, _ = self._by_id()
        for i in (4, 5):
            by_id[i] = dict(by_id[i], start_time=500.0, end_time=501.0)
        cand, reason = validate_candidate(
            {"start_id": 0, "end_id": 1, "join_ids": [4, 5]}, by_id,
            allow_noncontiguous=True, min_span_seconds=0, max_span_seconds=3600)
        assert cand is None and "distant" in reason


class TestDiscovery:
    def test_llm_failure_returns_empty(self, monkeypatch):
        import pipeline.discovery as D
        monkeypatch.setattr(D, "_llm_discover", lambda *a, **k: None)
        sents = _sents(6)
        out = generate_candidates({"method": "deterministic", "sections": []},
                                  [], sents, cfg={"discovery": {}})
        assert out == []

    def test_canned_response_validated(self, monkeypatch):
        import pipeline.discovery as D
        sents = _sents(6)
        monkeypatch.setattr(
            D, "_llm_discover",
            lambda *a, **k: {"candidates": [
                {"start_id": 0, "end_time": 0, "end_id": 1, "hook": "w0a",
                 "main_idea": "idea", "payoff": "pay", "uncertainty": 0.2},
                {"start_id": 999, "end_id": 1000},  # invalid -> dropped
            ]})
        out = generate_candidates({"method": "x", "sections": []}, [], sents,
                                  cfg={"discovery": {"target_clips": 4, "min_span_seconds": 0}})
        assert len(out) == 1
        assert out[0]["sentence_ids"] == [0, 1]
        assert out[0]["uncertainty"] == 0.2


class TestFingerprints:
    def test_stable_and_sensitive(self):
        cfg = {"outline": {"window_sentences": 40}}
        a = outline_fingerprint("t1", "m", cfg)
        assert a == outline_fingerprint("t1", "m", cfg)
        assert a != outline_fingerprint("t2", "m", cfg)
        assert a != outline_fingerprint("t1", "m2", cfg)

    def test_discovery_covers_target(self):
        c1 = {"discovery": {"target_clips": 8}}
        c2 = {"discovery": {"target_clips": 4}}
        assert discovery_fingerprint("t", "o", "m", c1) != \
            discovery_fingerprint("t", "o", "m", c2)


class TestRepoRoundTrip:
    def test_outline_candidate_persist(self):
        # DB-backed round trip with a throwaway video row, cleaned up after.
        from db.repositories import video_repo, outline_repo, candidate_repo
        vid = video_repo.insert_video("test://phase1", "test://phase1")
        try:
            oid = outline_repo.insert_outline(
                vid, "section", 0, start_time=0.0, end_time=10.0,
                title="t", content={"start_id": 0, "end_id": 3})
            assert oid
            secs = outline_repo.get_outlines(vid, "section")
            assert len(secs) == 1 and secs[0]["title"] == "t"
            cid = candidate_repo.insert_candidate(
                vid, "outline", [0, 1], [[0.0, 1.0]], hook="h",
                rationale="r", uncertainty=0.3)
            cands = candidate_repo.get_candidates(vid)
            assert len(cands) == 1 and cands[0]["hook"] == "h"
            candidate_repo.update_candidate_status(cid, "selected", "ok")
            sel = candidate_repo.get_candidates(vid, status="selected")
            assert len(sel) == 1
        finally:
            from conftest import cleanup_video
            cleanup_video(vid)
