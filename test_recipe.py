# test_recipe.py — campaign pipeline C2: edit recipe validation, platform
# presets, brief parsing (keyword fallback + mocked LLM).
import json
import os
import sys
from unittest.mock import MagicMock, patch


sys.path.insert(0, os.path.dirname(__file__))

from campaign.recipe import validate_recipe, describe_recipe, find_asset, empty_recipe  # noqa: E402
from campaign import presets  # noqa: E402
from campaign.brief_parser import keyword_parse, parse_brief  # noqa: E402

CAT = [{"name": "acme_logo.png", "kind": "logo", "path": "a/logo/acme_logo.png"},
       {"name": "summer_track.mp3", "kind": "audio", "path": "a/audio/summer_track.mp3"},
       {"name": "Brand.ttf", "kind": "font", "path": "a/font/Brand.ttf"}]


def ops(res):
    return [o["op"] for o in res["recipe"]["ops"]]


class TestValidate:
    def test_empty_recipe_defaults(self):
        r = validate_recipe({})
        assert r["ok"] and r["recipe"] == empty_recipe()

    def test_defaults_filled_and_assets_resolved(self):
        r = validate_recipe({"ops": [{"op": "logo", "asset": "acme_logo"},
                                     {"op": "captions"}]}, CAT)
        logo, caps = r["recipe"]["ops"]
        assert logo["asset"] == "acme_logo.png" and logo["asset_path"] == "a/logo/acme_logo.png"
        assert logo["position"] == "top-right" and logo["size_pct"] == 12.0
        assert caps["style"] == "karaoke" and caps["enabled"] is True and r["ok"]

    def test_clamping_and_parsing(self):
        r = validate_recipe({"ops": [{"op": "logo", "asset": "acme_logo.png", "size_pct": "90%",
                                      "position": "Bottom Left", "opacity": 2},
                                     {"op": "trim", "max_duration": "60 seconds"}]}, CAT)
        logo, trim = r["recipe"]["ops"]
        assert logo["size_pct"] == 60.0 and logo["opacity"] == 1.0
        assert logo["position"] == "bottom-left" and trim["max_duration"] == 60.0
        assert any("clamped" in w for w in r["warnings"])

    def test_invented_asset_is_an_error_and_a_question(self):
        r = validate_recipe({"ops": [{"op": "audio.replace", "track": "voiceover.wav"}]}, CAT)
        assert not r["ok"] and "not in the campaign's assets" in r["errors"][0]
        assert r["questions"][0]["options"] == ["summer_track.mp3"]

    def test_missing_asset_never_silently_substituted(self):
        r = validate_recipe({"ops": [{"op": "audio.replace"}]}, CAT)
        assert not r["ok"] and r["recipe"]["ops"][0]["track"] is None
        assert r["questions"][0]["default"] == "summer_track.mp3"
        auto = validate_recipe({"ops": [{"op": "audio.replace"}]}, CAT, autopick=True)
        assert auto["ok"] and auto["recipe"]["ops"][0]["track"] == "summer_track.mp3"

    def test_missing_text_asks(self):
        r = validate_recipe({"ops": [{"op": "text_overlay", "role": "cta"}]})
        assert not r["ok"] and "What text" in r["questions"][0]["question"]

    def test_unknown_ops_and_settings_dropped(self):
        r = validate_recipe({"ops": [{"op": "explode"}, {"op": "captions", "glitter": True}]})
        assert ops(r) == ["captions"] and len(r["warnings"]) == 2

    def test_aliases(self):
        r = validate_recipe({"ops": [{"op": "subtitles"}, {"op": "crop", "aspect": "square"},
                                     {"op": "music", "track": "summer_track.mp3"}]}, CAT)
        assert ops(r) == ["captions", "reframe", "audio.add_music"]
        assert r["recipe"]["ops"][1]["aspect"] == "1:1"

    def test_single_ops_deduplicated_last_wins(self):
        r = validate_recipe({"ops": [{"op": "reframe", "aspect": "1:1"},
                                     {"op": "reframe", "aspect": "4:5"}]})
        assert ops(r) == ["reframe"] and r["recipe"]["ops"][0]["aspect"] == "4:5"

    def test_conflicts(self):
        r = validate_recipe({"ops": [{"op": "audio.replace", "track": "summer_track.mp3"},
                                     {"op": "audio.mute_original"},
                                     {"op": "audio.add_music", "track": "summer_track.mp3"}]}, CAT)
        assert "audio.mute_original" not in ops(r)
        assert any(q["id"] == "audio.conflict" for q in r["questions"])

    def test_bad_times_and_clip_bounds(self):
        r = validate_recipe({"clips": {"min_duration": 90, "max_duration": 30},
                             "ops": [{"op": "text_overlay", "text": "x", "start": 5, "end": 2}]})
        assert len(r["errors"]) == 2

    def test_text_role_positions_and_fonts(self):
        r = validate_recipe({"ops": [{"op": "text_overlay", "text": "@acme", "role": "handle"},
                                     {"op": "text_overlay", "text": "Wait", "role": "hook",
                                      "font": "Brand.ttf", "color": "yellow"}]}, CAT)
        a, b = r["recipe"]["ops"]
        assert a["position"] == "bottom-right" and b["position"] == "top"
        assert b["font_path"] == "a/font/Brand.ttf" and b["color"] == "#FFE000"

    def test_c8_ops_validated(self):
        r = validate_recipe({"ops": [{"op": "speed", "factor": 3},
                                     {"op": "color", "preset": "warm", "saturation": "1.2"},
                                     {"op": "watermark", "text": "@acme"},
                                     {"op": "end_card", "text": "Follow for more"}]})
        assert ops(r) == ["speed", "color", "watermark", "end_card"] and r["ok"]
        assert r["recipe"]["ops"][0]["factor"] == 2.0                       # clamped
        assert not any("not rendered" in w for w in r["warnings"])
        lines = describe_recipe(r["recipe"])
        assert "Speed x2" in lines and "Colour: warm, saturation 1.2" in lines

    def test_two_add_only_variants_first_becomes_main(self):
        r = validate_recipe({"ops": [{"op": "speed", "factor": 1.1}], "variants": [
            {"name": "a", "ops": [{"op": "text_overlay", "text": "You need this", "role": "hook"}]},
            {"name": "b", "ops": [{"op": "text_overlay", "text": "Stop scrolling", "role": "hook"}]}]})
        assert ops(r) == ["speed", "text_overlay"] and r["recipe"]["ops"][1]["text"] == "You need this"
        assert [v["name"] for v in r["recipe"]["variants"]] == ["b"]

    def test_variants(self):
        from campaign.recipe import variants_of
        r = validate_recipe({"ops": [{"op": "text_overlay", "text": "Wait for it", "role": "hook"},
                                     {"op": "text_overlay", "text": "Link in bio", "role": "cta"},
                                     {"op": "audio.add_music", "track": "summer_track.mp3"}],
                             "variants": [{"name": "Hook B", "ops": [
                                 {"op": "text_overlay", "text": "You won't believe this",
                                  "role": "hook"}]},
                                 {"name": "no music", "remove": ["audio.add_music"]},
                                 {"name": "empty"}]}, CAT)
        assert [v["name"] for v in r["recipe"]["variants"]] == ["hook-b", "no-music"]
        vs = dict(variants_of(r["recipe"]))
        assert list(vs) == ["main", "hook-b", "no-music"]
        texts = [o["text"] for o in vs["hook-b"]["ops"] if o["op"] == "text_overlay"]
        assert texts == ["You won't believe this", "Link in bio"]
        assert "audio.add_music" not in [o["op"] for o in vs["no-music"]["ops"]]
        assert "variants" not in vs["main"]
        assert any(x.startswith("Variant hook-b: Text (hook)") for x in describe_recipe(r["recipe"]))

    def test_export_platforms_and_limits(self):
        r = validate_recipe({"export": ["TikTok", "instagram", "myspace", "tiktok"],
                             "ops": [{"op": "trim", "max_duration": 120}]})
        assert r["recipe"]["export"] == ["tiktok", "reels"]
        assert any("reels" in w and "90s" in w for w in r["warnings"])
        assert any("myspace" in w for w in r["warnings"])

    def test_filename_pattern_guarded(self):
        assert validate_recipe({"output": {"filename": "{campaign}/../{evil}"}})["recipe"][
            "output"]["filename"] == "{campaign}_{input}_{clip}_{platform}"

    def test_trim_start_warned_in_clip_mode(self):
        r = validate_recipe({"mode": "clip", "ops": [{"op": "trim", "start": 5}]})
        assert any("clip mode" in w for w in r["warnings"])

    def test_describe(self):
        r = validate_recipe({"mode": "edit", "export": ["tiktok"],
                             "ops": [{"op": "audio.replace", "track": "summer_track.mp3"},
                                     {"op": "text_overlay", "text": "Link in bio", "role": "cta",
                                      "from_end": 3}]}, CAT)
        lines = describe_recipe(r["recipe"])
        assert lines[0].startswith("Mode: edit")
        assert "Replace audio with summer_track.mp3, original removed" in lines
        assert 'Text (cta) "Link in bio" at bottom for the last 3s' in lines
        assert lines[-1] == "Export: TikTok"

    def test_find_asset_kinds(self):
        assert find_asset("summer_track", CAT, ("audio",))["name"] == "summer_track.mp3"
        assert find_asset("summer_track", CAT, ("logo",)) is None


class TestPresets:
    def test_aliases_and_sizes(self):
        assert presets.normalize_platform("Instagram") == "reels"
        assert presets.normalize_platform("nope") is None
        assert presets.output_size("4:5") == (1080, 1350)
        p = presets.preset("yt")
        assert p["platform"] == "shorts" and p["video_codec"] == "libx264"


EXAMPLE = ('Replace the audio with our summer track, add the logo top right at 15% width, add '
           'captions, put "Link in bio" for the last 3 seconds, vertical 9:16 for TikTok and '
           'Reels, under 60 seconds.')


class TestKeywordParser:
    def test_plan_example(self):
        r = parse_brief(EXAMPLE, CAT, use_llm=False)
        assert r["method"] == "keyword" and r["ok"]
        assert ops(r) == ["trim", "reframe", "audio.replace", "logo", "captions", "text_overlay"]
        rec = r["recipe"]
        assert rec["mode"] == "edit" and rec["export"] == ["tiktok", "reels"]
        logo = next(o for o in rec["ops"] if o["op"] == "logo")
        assert (logo["position"], logo["size_pct"]) == ("top-right", 15.0)
        cta = next(o for o in rec["ops"] if o["op"] == "text_overlay")
        assert (cta["role"], cta["from_end"]) == ("cta", 3.0)
        assert r["questions"][0]["id"] == "audio.keep_voice"

    def test_clip_brief(self):
        r = keyword_parse("From this long podcast cut 8 clips under 60 seconds, no captions, "
                          "post on YouTube Shorts")
        assert r["mode"] == "clip" and r["clips"] == {"count": 8, "max_duration": 60.0}
        assert {"op": "captions", "enabled": False} in r["ops"]
        assert r["export"] == ["shorts"]

    def test_only_what_is_asked(self):
        assert keyword_parse("make it look nice")["ops"] == []

    def test_hook_first_seconds(self):
        r = keyword_parse('Add the hook "Wait for it" for the first 3 seconds')
        o = r["ops"][0]
        assert (o["role"], o["start"], o["end"]) == ("hook", 0.0, 3.0)


def _llm_reply(payload):
    m = MagicMock()
    m.json.return_value = {"choices": [{"message": {"content": json.dumps(payload)}}]}
    return m


class TestLLMParser:
    def test_llm_recipe_validated(self):
        reply = {"mode": "edit", "ops": [{"op": "logo", "asset": "acme_logo.png",
                                          "position": "top right", "size_pct": 15}],
                 "export": ["tiktok"], "summary": ["logo top right"]}
        with patch("pipeline.llm_client.pool_usable", return_value=True), \
                patch("pipeline.llm_client.post_chat", return_value=_llm_reply(reply)) as pc:
            r = parse_brief("add the logo top right at 15%", CAT)
        assert r["method"] == "llm" and r["ok"] and r["brief_summary"] == ["logo top right"]
        kw = pc.call_args.kwargs
        assert kw["json_mode"] is True and pc.call_args.args[0]["temperature"] == 0
        assert "acme_logo.png" in pc.call_args.args[0]["messages"][1]["content"]

    def test_llm_invented_asset_caught(self):
        reply = {"ops": [{"op": "audio.replace", "track": "voiceover_final.wav"}]}
        with patch("pipeline.llm_client.pool_usable", return_value=True), \
                patch("pipeline.llm_client.post_chat", return_value=_llm_reply(reply)):
            r = parse_brief("use the voiceover", CAT)
        assert not r["ok"] and r["questions"]

    def test_llm_failure_falls_back_to_keywords(self):
        with patch("pipeline.llm_client.pool_usable", return_value=True), \
                patch("pipeline.llm_client.post_chat", side_effect=RuntimeError("boom")):
            r = parse_brief(EXAMPLE, CAT)
        assert r["method"] == "keyword" and "logo" in ops(r)

    def test_no_keys_uses_keywords(self):
        with patch("pipeline.llm_client.pool_usable", return_value=False):
            assert parse_brief(EXAMPLE, CAT)["method"] == "keyword"

    def test_llm_gaps_filled_from_the_brief(self):
        reply = {"mode": "edit", "ops": [{"op": "logo", "asset": "acme_logo.png"}], "export": []}
        with patch("pipeline.llm_client.pool_usable", return_value=True), \
                patch("pipeline.llm_client.post_chat", return_value=_llm_reply(reply)):
            r = parse_brief("add the logo, TikTok only, under 20 seconds", CAT)
        assert ops(r) == ["trim", "logo"] and r["recipe"]["ops"][0]["max_duration"] == 20.0
        assert r["recipe"]["export"] == ["tiktok"]
        assert any("LLM missed" in w for w in r["warnings"])

    def test_llm_filename_ignored_unless_asked(self):
        reply = {"mode": "edit", "ops": [], "output": {"filename": "{campaign}_{input}_tiktok"}}
        with patch("pipeline.llm_client.pool_usable", return_value=True), \
                patch("pipeline.llm_client.post_chat", return_value=_llm_reply(reply)):
            r = parse_brief("TikTok only", CAT)
            assert r["recipe"]["output"]["filename"] == "{campaign}_{input}_{clip}_{platform}"
            r = parse_brief("TikTok only, file names like {campaign}_{input}_tiktok", CAT)
            assert r["recipe"]["output"]["filename"] == "{campaign}_{input}_tiktok"

    def test_llm_invented_text_dropped_and_roles_fixed(self):
        reply = {"mode": "edit", "ops": [
            {"op": "text_overlay", "text": "HOOK_TEXT", "role": "hook"},
            {"op": "text_overlay", "text": "You need this", "role": "custom"}],
            "variants": [{"name": "b", "ops": [{"op": "text_overlay", "text": "Stop scrolling",
                                                "role": "custom"}]}]}
        with patch("pipeline.llm_client.pool_usable", return_value=True), \
                patch("pipeline.llm_client.post_chat", return_value=_llm_reply(reply)):
            r = parse_brief('Two versions: the hook "You need this" and the hook "Stop scrolling"',
                            CAT)
        texts = [(o["text"], o["role"]) for o in r["recipe"]["ops"]]
        assert texts == [("You need this", "hook")]
        assert r["recipe"]["variants"][0]["ops"][0]["role"] == "hook"
        assert any("HOOK_TEXT" in w for w in r["warnings"])

    def test_empty_brief(self):
        r = parse_brief("   ", CAT)
        assert r["method"] == "empty" and r["recipe"]["ops"] == []
