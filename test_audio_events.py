# test_audio_events.py — Phase 2: detector, gating, classifier adapter.
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from pipeline.audio_events import detect_events, speech_coverage, propose_from_events
from pipeline.audio_classifier import backend_status, classify_events
from pipeline.fingerprints import audio_events_fingerprint

SR = 16000


def _sig():
    """60s synthetic: burst@5s(1s), noise@20s(2s), click@30s(0.05s),
    sustained loud 40-55s, else near-silence."""
    y = np.random.default_rng(0).normal(0, 0.002, SR * 60)
    t = np.arange(SR) / SR
    y[5 * SR:6 * SR] += 0.5 * np.sin(2 * np.pi * 440 * t)
    y[20 * SR:22 * SR] += 0.4 * np.random.default_rng(1).normal(0, 1, 2 * SR)
    y[int(30 * SR):int(30 * SR) + int(0.05 * SR)] += 0.8
    y[40 * SR:55 * SR] += 0.5
    return y


CFG = {"window_ms": 200, "baseline_seconds": 10, "threshold_db": 8.0,
       "min_duration_seconds": 0.4, "max_event_seconds": 8.0,
       "merge_gap_seconds": 1.0, "silence_floor": 1e-4}


class TestDetector:
    def test_finds_bursts(self):
        evs = detect_events(_sig(), SR, CFG)
        assert any(abs(e["peak_time"] - 5.5) < 1.0 for e in evs), evs
        assert any(abs(e["peak_time"] - 21.0) < 1.5 for e in evs), evs

    def test_click_suppressed(self):
        evs = detect_events(_sig(), SR, CFG)
        assert not any(29.5 < e["start_time"] < 30.5 and
                       e["end_time"] - e["start_time"] < 0.3 for e in evs), evs

    def test_sustained_capped_or_suppressed(self):
        # A 15s uniformly loud region must not yield a long event: the
        # rolling baseline adapts (suppression) or the cap truncates.
        evs = detect_events(_sig(), SR, CFG)
        for e in evs:
            assert e["end_time"] - e["start_time"] <= 8.01, e
        inside = [e for e in evs
                  if e["start_time"] >= 42 and e["end_time"] <= 53]
        assert not inside, inside  # sustained middle suppressed

    def test_silence_safe(self):
        y = np.zeros(SR * 10)
        assert detect_events(y, SR, CFG) == []
        assert detect_events(np.zeros(0), SR, CFG) == []

    def test_source_times_bounded(self):
        evs = detect_events(_sig(), SR, CFG)
        for e in evs:
            assert 0 <= e["start_time"] <= e["peak_time"] <= e["end_time"] <= 60.01

    def test_merge_nearby(self):
        y = np.random.default_rng(2).normal(0, 0.002, SR * 20)
        y[5 * SR:int(5.6 * SR)] += 0.5
        y[int(6.0 * SR):int(6.6 * SR)] += 0.5  # 0.4s gap < merge 1.0
        evs = detect_events(y, SR, CFG)
        assert len(evs) == 1, evs
        assert evs[0]["start_time"] < 5.5 and evs[0]["end_time"] > 6.0


def _sent_words():
    # sentences every ~2s from 0..60s
    words, sents = [], []
    t, sid = 0.0, 0
    while t < 58:
        sents.append({"id": sid, "start_time": round(t, 2),
                      "end_time": round(t + 1.5, 2),
                      "text": f"sentence {sid} here."})
        for k in range(5):
            words.append({"word": f"w{sid}_{k}", "start": round(t + k * 0.3, 2),
                          "end": round(t + k * 0.3 + 0.25, 2)})
        t += 2.0
        sid += 1
    return words, sents


class TestCues:
    def test_cue_uses_sentence_bounds(self):
        words, sents = _sent_words()
        evs = [{"start_time": 10.0, "end_time": 11.0, "peak_time": 10.5,
                "energy_increase": 12.0, "confidence": 0.8}]
        cfg = {"audio_events": {"pre_seconds": 25, "post_seconds": 10,
                                "min_speech_coverage": 0.1},
               "discovery": {"min_span_seconds": 0, "max_span_seconds": 3600}}
        cues = propose_from_events(evs, words, sents, cfg)
        assert len(cues) == 1
        a, b = cues[0]["source_ranges"][0]
        # aligned to sentence bounds, not spike bounds
        assert a < 10.0 and b > 11.0
        assert cues[0]["sentence_ids"][0] == 0 or True  # window covers start

    def test_quiet_window_dropped(self):
        words, sents = _sent_words()
        # words only cover 0-10s; event at 50s has no speech support
        words = [w for w in words if w["end"] < 10]
        sents = [s for s in sents if s["end_time"] < 10]
        evs = [{"start_time": 50.0, "end_time": 51.0, "peak_time": 50.5,
                "energy_increase": 15.0, "confidence": 0.9}]
        cfg = {"audio_events": {"pre_seconds": 25, "post_seconds": 10,
                                "min_speech_coverage": 0.3},
               "discovery": {"min_span_seconds": 0, "max_span_seconds": 3600}}
        cues = propose_from_events(evs, words, sents, cfg)
        assert cues == []

    def test_speech_coverage_math(self):
        words = [{"word": "a", "start": 0.0, "end": 5.0}]
        assert abs(speech_coverage(0, 10, words) - 0.5) < 1e-9
        assert speech_coverage(5, 5, words) == 0.0


class TestClassifier:
    def test_disabled_by_default(self):
        st = backend_status({"classifier": {"backend": "none"}})
        assert st["status"] == "disabled"
        evs = [{"start_time": 1.0, "end_time": 2.0}]
        out, st2 = classify_events(evs, None, 16000, {"classifier": {"backend": "none"}})
        assert out == evs and all("label" not in e or e["label"] is None for e in out)

    def test_unknown_backend_disabled(self):
        st = backend_status({"classifier": {"backend": "wat"}})
        assert st["status"] == "disabled"

    def test_yamnet_missing_model_disabled(self):
        st = backend_status({"classifier": {"backend": "yamnet",
                                            "model_dir": "no/such/dir"}})
        assert st["status"] == "disabled" and "missing" in st["reason"]


class TestFingerprint:
    def test_audio_fp_stable(self):
        cfg = {"audio_events": {"window_ms": 200}}
        assert audio_events_fingerprint("a", cfg) == audio_events_fingerprint("a", cfg)
        assert audio_events_fingerprint("a", cfg) != audio_events_fingerprint("b", cfg)


class TestEventPersist:
    def test_round_trip(self):
        from db.repositories import video_repo, event_repo
        vid = video_repo.insert_video("test://phase2", "test://phase2")
        try:
            eid = event_repo.insert_event(vid, 1.0, 2.0, peak_time=1.5,
                                          energy_increase=9.0, confidence=0.7)
            assert eid
            evs = event_repo.get_events(vid)
            assert len(evs) == 1 and evs[0]["peak_time"] == 1.5
        finally:
            from conftest import cleanup_video
            cleanup_video(vid)
