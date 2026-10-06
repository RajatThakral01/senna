# test_glossary.py — display corrections traceable to source words.
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from pipeline.glossary import (normalize_glossary, apply_glossary,
                               display_text)


def _w(texts):
    return [{"word": w, "start": float(i), "end": float(i) + 0.3}
            for i, w in enumerate(texts)]


class TestGlossary:
    def test_single_word(self):
        words = _w(["I", "love", "airpower", "much"])
        out, corr = apply_glossary(
            words, [{"match": "airpower", "replacement": "AirPower"}])
        assert out[2]["display"] == "AirPower"
        assert out[2]["word"] == "airpower"  # source kept
        assert out[2]["start"] == 2.0  # timings kept
        assert corr[0]["source"] == "airpower"

    def test_multi_word(self):
        words = _w(["the", "wireless", "charger", "works"])
        out, corr = apply_glossary(
            words, [{"match": "wireless charger", "replacement": "MagSafe"}])
        assert out[1]["display"] == "MagSafe"
        assert out[2]["display"] == ""  # hidden continuation
        assert len(corr) == 1

    def test_case_insensitive(self):
        words = _w(["Hello"])
        out, corr = apply_glossary(words, [{"match": "hello", "replacement": "HELLO"}])
        assert out[0]["display"] == "HELLO"

    def test_no_invention(self):
        words = _w(["a", "b"])
        out, corr = apply_glossary(words, [{"match": "zzz", "replacement": "Q"}])
        assert corr == [] and all("display" not in w for w in out)

    def test_empty(self):
        words = _w(["a"])
        out, corr = apply_glossary(words, [])
        assert corr == [] and out == words

    def test_normalize(self):
        assert normalize_glossary([{"match": "a", "replacement": "a"}]) == []
        assert normalize_glossary([{"match": "", "replacement": "x"}]) == []
        # case-only fixes are legitimate corrections
        assert len(normalize_glossary([{"match": "airpower",
                                        "replacement": "AirPower"}])) == 1

    def test_display_fallback(self):
        assert display_text({"word": "x"}) == "x"
        assert display_text({"word": "x", "display": "Y"}) == "Y"
        assert display_text({"word": "x", "display": ""}) == ""

    def test_srt_uses_display(self):
        from pipeline.regen_srt import generate_srt_for_clip
        segs = [{"start": 0, "end": 2, "words": [
            {"word": "airpower", "display": "AirPower", "start": 0.0, "end": 0.4}]}]
        srt = generate_srt_for_clip(segs, 0.0, 2.0, 1)
        assert "AirPower" in srt and "airpower" not in srt.replace("AirPower", "")

    def test_captions_use_display(self):
        from pipeline.captions import build_captions
        segs = [{"start": 0, "end": 2, "words": [
            {"word": "airpower", "display": "AirPower", "start": 0.0, "end": 0.4}]}]
        cap = build_captions(segs, [(0.0, 2.0)], {})
        assert "AirPower" in cap["ass"]
