# test_captions.py — Phase 4: phrase grouping, ASS build, placement, timing.
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from pipeline.captions import (group_phrases, balance_lines, ass_escape,
                               build_ass, decide_placement, build_captions,
                               ass_filter_available, render_captions_ass)


def _words(n=12, start=0.0, step=0.4, dur=0.35, punct_every=0):
    words = []
    t = start
    for i in range(n):
        w = f"w{i}"
        if punct_every and (i + 1) % punct_every == 0:
            w += "."
        words.append({"word": w, "start": round(t, 3), "end": round(t + dur, 3)})
        t += step
    return words


def _mapping(words, offset=0.0):
    a = words[0]["start"] if words else 0.0
    b = words[-1]["end"] if words else 0.0
    return [{"source_start": a, "source_end": b, "output_start": offset,
             "output_end": offset + (b - a)}]


class TestGrouping:
    def test_word_limit(self):
        words = _words(12)
        cues = group_phrases(words, _mapping(words), {"max_words": 5,
                                                      "max_chars": 1000,
                                                      "max_duration": 100})
        assert all(len(c["words"]) <= 5 for c in cues)
        assert sum(len(c["words"]) for c in cues) == 12

    def test_sentence_aware_break(self):
        words = _words(10, punct_every=5)
        cues = group_phrases(words, _mapping(words), {"max_words": 8,
                                                      "max_chars": 1000,
                                                      "max_duration": 100})
        assert cues[0]["words"][-1]["word"].endswith(".")

    def test_limits_win_over_punctuation(self):
        words = _words(10)  # no punctuation at all
        cues = group_phrases(words, _mapping(words), {"max_words": 4,
                                                      "max_chars": 1000,
                                                      "max_duration": 100})
        assert len(cues) == 3  # 4+4+2, no hanging content

    def test_unmapped_words_skipped(self):
        words = _words(6)
        mapping = [{"source_start": 999, "source_end": 1000,
                    "output_start": 0, "output_end": 1}]
        assert group_phrases(words, mapping, {}) == []


class TestLines:
    def test_max_two_lines(self):
        words = _words(10)
        lines = balance_lines(words, max_lines=2)
        assert 1 <= len(lines) <= 2
        assert " ".join(" ".join(
            w["word"] if isinstance(w, dict) else w for w in g) for g in lines
        ).split() == [w["word"] for w in words]

    def test_single_word(self):
        assert balance_lines([{"word": "hi"}], 2) == [[{"word": "hi"}]]


class TestASS:
    def test_escape(self):
        assert ass_escape("a{b}\\c") == "a(b)￣c"
        assert ass_escape("héllo ✨") == "héllo ✨"  # unicode passes through

    def test_karaoke_tags_match_words(self):
        words = _words(4)
        cues = group_phrases(words, _mapping(words), {"max_words": 10,
                                                      "max_chars": 1000,
                                                      "max_duration": 100})
        ass = build_ass(cues, {})
        assert ass.count("{\\kf") == 4
        assert "Dialogue:" in ass and "[V4+ Styles]" in ass

    def test_two_line_break_present(self):
        words = _words(10)
        cues = group_phrases(words, _mapping(words), {"max_words": 10,
                                                      "max_chars": 20,
                                                      "max_duration": 100})
        ass = build_ass(cues, {})
        assert "\\N" in ass


class TestPlacement:
    def test_default_bottom(self):
        assert decide_placement(None, {}) == "bottom"
        assert decide_placement({}, {"placement": "bottom"}) == "bottom"

    def test_forced_top(self):
        assert decide_placement(None, {"placement": "top"}) == "top"

    def test_faces_low_switches_top(self):
        analysis = {"height": 720, "samples": [
            {"tracks": [{"cx": 100, "cy": 600}]},
            {"tracks": [{"cx": 110, "cy": 620}]},
            {"tracks": [{"cx": 105, "cy": 610}]},
        ]}
        assert decide_placement(analysis, {}) == "top"

    def test_faces_high_stays_bottom(self):
        analysis = {"height": 720, "samples": [
            {"tracks": [{"cx": 100, "cy": 200}]},
            {"tracks": [{"cx": 110, "cy": 210}]},
            {"tracks": [{"cx": 105, "cy": 190}]},
        ]}
        assert decide_placement(analysis, {}) == "bottom"


class TestBuild:
    def _segs(self):
        words = []
        t = 0.0
        for w in ["hello", "world.", "foo", "bar."]:
            words.append({"word": w, "start": t, "end": t + 0.3})
            t += 0.5
        t = 10.0
        for w in ["baz", "qux."]:
            words.append({"word": w, "start": t, "end": t + 0.3})
            t += 0.5
        return [{"start": 0, "end": 12, "words": words}]

    def test_stitched_no_drift(self):
        cap = build_captions(self._segs(), [(0.0, 2.0), (10.0, 11.0)], {})
        assert cap["mapping"][1]["output_start"] == 2.0
        total_words = sum(len(c["words"]) for c in cap["cues"])
        assert total_words == 6  # all words, none duplicated
        last_end = max(c["end"] for c in cap["cues"])
        assert last_end <= cap["mapping"][-1]["output_end"] + 0.01
        assert "Dialogue:" in cap["ass"] and len(cap["srt"]) > 0

    def test_filter_probe_bool(self):
        assert isinstance(ass_filter_available(), bool)

    def test_render_failure_returns_false(self):
        assert render_captions_ass("no/such/file.mp4",
                                   "no/such/out.mp4", "x") is False
