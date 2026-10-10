# test_boundaries.py — boundary refinement & stitched subtitle sync.
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from pipeline.boundaries import (
    build_sentences, snap_range, refine_clip, refine_all_clips,
)
from pipeline.regen_srt import generate_srt_for_ranges


def _words(texts, start=0.0, step=0.4, dur=0.35):
    words = []
    t = start
    for w in texts:
        words.append({"word": w, "start": round(t, 3), "end": round(t + dur, 3)})
        t += step
    return words


CFG = {"context_seconds": 20, "max_extension_seconds": 15, "min_duration": 5,
       "max_duration": 90, "start_padding": 0.15, "end_padding": 0.3,
       "llm_validation": False}  # deterministic in tests


class TestSnapToSentence:
    def test_end_inside_sentence_extends_to_sentence_end(self):
        # "This is a complete thought. This is another one."
        words = _words(["This", "is", "a", "complete", "thought.",
                        "This", "is", "another", "one."])
        sents = build_sentences(words)
        # candidate ends mid second sentence ("another")
        cand_end = words[7]["end"] - 0.05
        ns, ne, info = snap_range(words[0]["start"], cand_end, words, sents, CFG)
        assert ne >= words[8]["end"] - 0.01  # extended to sentence end
        assert info["end_sentence_id"] == 1

    def test_start_inside_sentence_moves_to_beginning(self):
        words = _words(["First", "sentence.", "Second", "sentence", "here."])
        sents = build_sentences(words)
        ns, ne, _ = snap_range(words[3]["start"], words[4]["end"], words, sents, CFG)
        assert ns <= words[2]["start"]  # snapped to "Second"

    def test_missing_punctuation_pause_creates_utterance(self):
        words = _words(["well", "um", "so", "hello", "there", "friend", "today"])
        # long pause after 3rd word, no punctuation: shift tail by +1.5s
        for w in words[3:]:
            w["start"] = round(w["start"] + 1.5, 3)
            w["end"] = round(w["end"] + 1.5, 3)
        sents = build_sentences(words, pause_threshold=1.0)
        assert len(sents) >= 2  # pause boundary without punctuation handled
        ns, ne, _ = snap_range(words[0]["start"], words[-1]["end"], words, sents, CFG)
        assert ne >= words[-1]["end"] - 0.01

    def test_missing_word_timestamps_skipped(self):
        words = _words(["hello", "world", "again."])
        words.insert(1, {"word": "oops"})  # no timestamps
        from pipeline.boundaries import load_words  # noqa
        sents = build_sentences([w for w in words if "start" in w])
        assert len(sents) == 1

    def test_candidate_near_end_bounded_by_video(self):
        words = _words(["Final", "words", "of", "the", "video."])
        sents = build_sentences(words)
        clip = {"start_time": words[0]["start"], "end_time": words[-1]["end"] + 5,
                "hook": "x"}
        refined, action = refine_clip(clip, words, sents, video_duration=words[-1]["end"] + 0.1,
                                      cfg=CFG)
        assert refined["refined_end_time"] <= words[-1]["end"] + 0.31 + 1e-6
        assert action in ("refined", "passthrough")

    def test_duration_conflict_rejects_or_trims_start(self):
        words = _words([f"w{i}" + ("." if i % 5 == 4 else "") for i in range(60)],
                       step=0.5)
        sents = build_sentences(words)
        clip = {"start_time": words[0]["start"], "end_time": words[-1]["end"], "hook": "x"}
        cfg = dict(CFG, max_duration=3.0)
        refined, action = refine_clip(clip, words, sents, cfg=cfg)
        assert action in ("refined", "rejected")
        if action == "refined":
            assert refined["refined_end_time"] - refined["refined_start_time"] <= 3.0 + 0.6
            # end kept at a sentence end (never truncated mid-sentence)
            assert any(abs(refined["refined_end_time"] - (s["end_time"] + 0.3)) < 0.6
                       for s in sents)


class TestLLMCompleteness:
    def test_thought_unfinished_falls_back_or_rejects(self, monkeypatch):
        # Complete sentence but unfinished thought: "The reason it failed is"
        words = _words(["The", "reason", "it", "failed", "is.",
                        "The", "capacitor", "burned", "out."])
        sents = build_sentences(words)
        clip = {"start_time": words[0]["start"], "end_time": words[4]["end"],
                "hook": "reason"}
        import pipeline.boundaries as B
        monkeypatch.setattr(B, "_llm_validate",
                            lambda *a, **k: {"start_id": 0, "end_id": 0,
                                             "complete": False,
                                             "reason": "setup without payoff"})
        refined, action = refine_clip(clip, words, sents, cfg=dict(CFG, llm_validation=True))
        # either extended to include the payoff or rejected with a reason
        assert action in ("refined", "rejected")
        assert refined.get("refine_reason")

    def test_llm_selects_ids_timestamps_come_from_transcript(self, monkeypatch):
        words = _words(["Hello", "world.", "Goodbye", "world."])
        sents = build_sentences(words)
        clip = {"start_time": 0.0, "end_time": 0.5, "hook": "hi"}
        import pipeline.boundaries as B
        monkeypatch.setattr(B, "_llm_validate",
                            lambda *a, **k: {"start_id": 1, "end_id": 1,
                                             "complete": True, "reason": "ok"})
        refined, _ = refine_clip(clip, words, sents, cfg=dict(CFG, llm_validation=True))
        # timestamps derived from sentence 1, not invented; start is bounded
        # by neighbouring speech (prev word end + 0.01 wins over raw padding)
        assert abs(refined["refined_start_time"] - (words[1]["end"] + 0.01)) < 1e-6
        assert abs(refined["refined_end_time"] - (sents[1]["end_time"] + 0.3)) < 0.05


class TestStitchedSubtitles:
    def _segs(self):
        # two ranges: words 0-3 @0-2s, words 4-7 @10-12s
        words = []
        t = 0.0
        for w in ["a", "b", "c.", "d."]:
            words.append({"word": w, "start": t, "end": t + 0.3})
            t += 0.5
        t = 10.0
        for w in ["e", "f", "g.", "h."]:
            words.append({"word": w, "start": t, "end": t + 0.3})
            t += 0.5
        return [{"start": 0, "end": 12, "words": words}]

    def test_subtitle_sync_after_stitching(self):
        segs = self._segs()
        srt, mapping = generate_srt_for_ranges(segs, [(0.0, 2.0), (10.0, 12.0)], 1)
        assert len(mapping) == 2
        assert mapping[0]["output_start"] == 0.0
        assert mapping[1]["output_start"] == 2.0  # gap removed
        # word "e" at source 10.0 appears at output ~0.0 of second range
        assert "00:00:02" in srt  # second range starts at output 2s
        lines = [l for l in srt.splitlines() if l.strip() and "-->" in l]
        assert len(lines) == 8  # all 8 words kept, none dropped/duplicated


class TestPayoffExtension:
    """The selected ending is checked, and extended forward to the payoff."""

    def _story(self):
        # setup (0-4), unfinished line, then the punchline ~20s later
        words = _words(["So", "we", "opened", "the", "box.",
                        "And", "inside", "was."])
        t = words[-1]["end"] + 20.0
        for w in ["A", "live", "goat!"]:
            words.append({"word": w, "start": t, "end": t + 0.3})
            t += 0.4
        return words, build_sentences(words)

    def test_incomplete_selection_extends_to_payoff(self, monkeypatch):
        import pipeline.boundaries as B
        words, sents = self._story()
        clip = {"start_time": 0.0, "end_time": sents[1]["end_time"], "hook": "box"}
        monkeypatch.setattr(B, "_llm_validate", lambda *a, **k: {
            "start_id": 0, "end_id": 1, "complete": False, "candidate_complete": False,
            "missing": "what was inside", "reason": "no reveal"})
        seen = {}

        def ext(clip, s, e, forward, missing, cfg):
            seen["forward"] = [x["id"] for x in forward]
            seen["missing"] = missing
            return {"start_id": 0, "end_id": 2, "complete": True, "reason": "reveal lands"}
        monkeypatch.setattr(B, "_llm_extend", ext)
        refined, action = refine_clip(clip, words, sents, cfg=dict(CFG, llm_validation=True))
        assert action == "refined"
        assert 2 in seen["forward"] and seen["missing"] == "what was inside"
        assert abs(refined["refined_end_time"] - (sents[2]["end_time"] + 0.3)) < 0.05
        assert refined["refine_reason"].startswith("llm_extended_to_payoff")

    def test_no_payoff_found_rejects(self, monkeypatch):
        import pipeline.boundaries as B
        words, sents = self._story()
        clip = {"start_time": 0.0, "end_time": sents[1]["end_time"], "hook": "box"}
        monkeypatch.setattr(B, "_llm_validate", lambda *a, **k: {
            "start_id": 0, "end_id": 1, "complete": False, "missing": "x", "reason": "no reveal"})
        monkeypatch.setattr(B, "_llm_extend", lambda *a, **k: {
            "start_id": 0, "end_id": 2, "complete": False, "reason": "never resolves"})
        refined, action = refine_clip(clip, words, sents, cfg=dict(CFG, llm_validation=True))
        assert action == "rejected"
        assert "thought unfinished" in refined["refine_reason"]

    def test_reject_incomplete_can_be_disabled(self, monkeypatch):
        import pipeline.boundaries as B
        words, sents = self._story()
        clip = {"start_time": 0.0, "end_time": sents[1]["end_time"], "hook": "box"}
        monkeypatch.setattr(B, "_llm_validate", lambda *a, **k: {
            "start_id": 0, "end_id": 1, "complete": False, "missing": "x", "reason": "r"})
        monkeypatch.setattr(B, "_llm_extend", lambda *a, **k: None)
        refined, action = refine_clip(clip, words, sents,
                                      cfg=dict(CFG, llm_validation=True, reject_incomplete=False))
        assert action == "refined"
        assert refined["refine_reason"].startswith("llm_incomplete_kept")

    def test_confirmed_ending_not_cut_back_by_short_cap(self, monkeypatch):
        # the payoff lands ~20s after the candidate end: beyond the 15s
        # max_extension, inside max_payoff_extension -> kept, not truncated
        import pipeline.boundaries as B
        words, sents = self._story()
        clip = {"start_time": 0.0, "end_time": sents[1]["end_time"], "hook": "box"}
        monkeypatch.setattr(B, "_llm_validate", lambda *a, **k: {
            "start_id": 0, "end_id": 1, "complete": False, "missing": "x", "reason": "r"})
        monkeypatch.setattr(B, "_llm_extend", lambda *a, **k: {
            "start_id": 0, "end_id": 2, "complete": True, "reason": "ok"})
        refined, action = refine_clip(clip, words, sents,
                                      cfg=dict(CFG, llm_validation=True,
                                               max_extension_seconds=15,
                                               max_payoff_extension_seconds=30))
        assert action == "refined"
        assert refined["refined_end_time"] > sents[2]["start_time"]


class TestStitchedTotalCap:
    def test_continuation_dropped_when_total_exceeds_cap(self, monkeypatch):
        import pipeline.boundaries as B
        from pipeline.boundaries import refine_all_clips

        def fake_refine(clip, words, sents, vd, cfg):
            s, e = float(clip["start_time"]), float(clip["end_time"])
            return ({**clip, "refined_start_time": s, "refined_end_time": e,
                     "refine_status": "refined", "source_ranges": [[s, e]]}, "refined")
        monkeypatch.setattr(B, "refine_clip", fake_refine)
        monkeypatch.setattr(B, "load_words", lambda p=None: [])
        clip = {"clip_number": 1, "start_time": 0.0, "end_time": 70.0,
                "related_segments": [{"start_time": 75.0, "end_time": 105.0}]}
        out = refine_all_clips([clip], transcript_path="x",
                               cfg={"max_total_seconds": 90, "max_stitch_gap_seconds": 45})
        assert out[0]["source_ranges"] == [[0.0, 70.0]]       # 70 + 30 > 90 -> dropped
        assert out[0]["output_duration"] <= 90

    def test_merged_total(self):
        from pipeline.boundaries import _merged_total
        assert _merged_total([(0, 10), (5, 20), (30, 40)]) == 30
