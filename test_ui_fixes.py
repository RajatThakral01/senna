# test_ui_fixes.py — unit tests (no DB, no ffmpeg) for the 15 Oct review fixes:
# review navigation / approve-reject auto-advance, AI-written on-screen text,
# overlay styling, edit-mode warning on long videos, duplicate-job guard,
# campaign summary in New job.
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(__file__))

import campaign_ui as U  # noqa: E402
from campaign import onscreen_text as T  # noqa: E402


def _rows(n):
    return [[i, f"f{i}.mp4", "tiktok / main", "10.0s", "✅ pass", "rendered"]
            for i in range(1, n + 1)]


# ── review navigation ────────────────────────────────────────────────────────

class TestReviewNavigation:
    def test_step_row_clamps(self):
        with patch.object(U, "deliverables_table", return_value=_rows(3)):
            assert U.step_row("j", 1, 1) == 2
            assert U.step_row("j", 3, 1) == 3          # last stays
            assert U.step_row("j", 1, -1) == 1         # first stays
            assert U.step_row("j", None, 1) == 2       # empty box → treated as 1
        assert U.step_row(None, 2, 1) == 1             # no job

    def test_row_from_table_click(self):
        assert U.row_from_select(SimpleNamespace(index=[2, 4])) == 3
        assert U.row_from_select(SimpleNamespace(index=0)) == 1

    def test_decide_advances_to_next(self):
        d = {"id": "d1", "path": "out/a.mp4"}
        with patch.object(U, "_deliverable", return_value=d), \
                patch.object(U, "deliverables_table", return_value=_rows(3)), \
                patch("campaign.delivery.set_status") as st:
            table, msg, nxt = U.decide("j", 1, "approved")
        st.assert_called_once_with("d1", "approved")
        assert nxt == 2 and "approved" in msg and "a.mp4" in msg and "#2" in msg

    def test_decide_last_row_stays(self):
        with patch.object(U, "_deliverable", return_value={"id": "d3", "path": "c.mp4"}), \
                patch.object(U, "deliverables_table", return_value=_rows(3)), \
                patch("campaign.delivery.set_status"):
            _, msg, nxt = U.decide("j", 3, "rejected")
        assert nxt == 3 and "last file" in msg and "rejected" in msg

    def test_decide_without_row(self):
        with patch.object(U, "_deliverable", return_value=None), \
                patch.object(U, "deliverables_table", return_value=[]):
            _, msg, nxt = U.decide("j", 7, "approved")
        assert "Pick a row" in msg and nxt == 7


# ── campaign summary (New job) ───────────────────────────────────────────────

class TestCampaignSummary:
    def test_ready_campaign_lists_steps(self):
        rec = {"mode": "edit", "ops": [{"op": "intro", "asset": "intro_card.png"}],
               "export": ["tiktok"]}
        camp = {"name": "C8 Trial", "recipe": rec, "recipe_version": 4}
        cat = [{"name": "intro_card.png", "kind": "image", "path": "x.png"}]
        with patch("campaign.service.get_campaign", return_value=camp), \
                patch("campaign.service.asset_catalog", return_value=cat):
            md = U.campaign_summary("cid")
        assert "C8 Trial" in md and "Ready to run" in md and "Intro" in md

    def test_draft_campaign_shows_questions(self):
        rec = {"mode": "clip", "ops": [{"op": "text_overlay", "text": None}]}
        with patch("campaign.service.get_campaign",
                   return_value={"name": "u2", "recipe": rec, "recipe_version": 3}), \
                patch("campaign.service.asset_catalog", return_value=[]):
            md = U.campaign_summary("cid")
        assert "Open questions" in md and "text_overlay" in md

    def test_no_campaign(self):
        assert "Pick the campaign" in U.campaign_summary(None)


# ── AI-written on-screen text ────────────────────────────────────────────────

AUTO_REC = {"ops": [{"op": "logo", "asset": "l.png"},
                    {"op": "text_overlay", "text": "{auto}", "role": "hook"}]}


class TestOnscreenText:
    def test_has_auto(self):
        assert T.has_auto(AUTO_REC)
        assert not T.has_auto({"ops": [{"op": "text_overlay", "text": "Hi"}]})
        assert T.has_auto({"ops": [], "variants": [{"ops": [{"op": "text_overlay",
                                                             "text": "{auto}"}]}]})

    def test_resolve_fills_and_avoids_repeats(self):
        seen = []

        def fake(brief, transcript, context, avoid, focus=None):
            seen.append(list(avoid))
            return {"text": "Name one other band that's lasted 50 years.",
                    "caption": "Fifty years. @band #band"}
        caps = []
        with patch.object(T, "write_post", side_effect=fake):
            rec, used, warn = T.resolve_auto_text(AUTO_REC, "brief", "words", avoid=["old"],
                                                  captions=caps)
        assert rec["ops"][1]["text"].startswith("Name one") and used and not warn
        assert seen == [["old"]] and caps == ["Fifty years. @band #band"]
        assert AUTO_REC["ops"][1]["text"] == "{auto}"          # input not mutated

    def test_llm_failure_drops_overlay_with_warning(self):
        with patch.object(T, "write_post", return_value=None):
            rec, used, warn = T.resolve_auto_text(AUTO_REC, "b", "t")
        assert [o["op"] for o in rec["ops"]] == ["logo"] and not used
        assert "could not be written" in warn[0]

    def test_stored_text_reused_without_llm(self):
        with patch.object(T, "write_post", side_effect=AssertionError("LLM called")):
            rec, used, _ = T.resolve_auto_text(AUTO_REC, "b", "t", stored=["Kept line"])
        assert rec["ops"][1]["text"] == "Kept line" and used == ["Kept line"]

    def test_display_mode_never_calls_llm(self):
        with patch.object(T, "write_post", side_effect=AssertionError("LLM called")):
            rec, used, warn = T.resolve_auto_text(AUTO_REC, None, None, generate=False)
        assert rec["ops"][1]["text"] == "{auto}" and not used and not warn

    def test_write_post_returns_line_and_caption(self):
        r = MagicMock()
        r.json.return_value = {"choices": [{"message": {"content": json.dumps(
            {"text": "Came for *Kacey*, stayed for *Estevie*",
             "caption": "Who is she?? @estevie #estevie #kaceymusgraves"})}}]}
        with patch("pipeline.llm_client.pool_usable", return_value=True), \
                patch("pipeline.llm_client.post_chat", return_value=r) as pc:
            post = T.write_post("Tag @estevie in every caption", "la la")
        assert post == {"text": "Came for *Kacey*, stayed for *Estevie*",
                        "caption": "Who is she?? @estevie #estevie #kaceymusgraves"}
        prompt = pc.call_args.args[0]["messages"][0]["content"]
        assert "POST CAPTION" in prompt and "*asterisks*" in prompt

    def test_quotes_kept_unless_they_wrap_the_line(self):
        assert T._norm('"This Man" isn\'t even out yet') == '"This Man" isn\'t even out yet'
        assert T._norm('"Estevie owned that stage"') == "Estevie owned that stage"

    def test_clean_trims_quotes_and_length(self):
        assert T._clean('  "FIFTY. YEARS."  ') == "FIFTY. YEARS."
        out = T._clean("word " * 40)
        assert len(out) <= T.MAX_CHARS and out.endswith("…")
        # keeps whole sentences instead of a dangling half-phrase ("... Put")
        s = ("Chris Thile on mandolin and Marcus singing Here like it's just the two of them. "
             "Put this on repeat")
        assert T._clean(s) == ("Chris Thile on mandolin and Marcus singing Here like it's "
                               "just the two of them.")

    def test_too_long_line_is_rewritten_not_chopped(self):
        long_r, short_r = MagicMock(), MagicMock()
        long_r.json.return_value = {"choices": [{"message": {"content": json.dumps(
            {"text": "x " * 60})}}]}
        short_r.json.return_value = {"choices": [{"message": {"content":
                                                              '{"text": "Short and sharp."}'}}]}
        with patch("pipeline.llm_client.pool_usable", return_value=True), \
                patch("pipeline.llm_client.post_chat", side_effect=[long_r, short_r]) as pc:
            assert T.write_text("b", "t") == "Short and sharp."
        retry = pc.call_args_list[1].args[0]["messages"]
        assert retry[-1]["role"] == "user" and "characters" in retry[-1]["content"]

    def test_write_text_parses_llm_json(self):
        r = MagicMock()
        r.json.return_value = {"choices": [{"message": {"content":
                                                        '{"text": "50 years. Same four lads."}'}}]}
        with patch("pipeline.llm_client.pool_usable", return_value=True), \
                patch("pipeline.llm_client.post_chat", return_value=r) as pc:
            assert T.write_text("Stay positive", "we met at school",
                                focus="Mumford & Sons") == "50 years. Same four lads."
        kw = pc.call_args.kwargs
        assert kw["purpose"] == "onscreen_text" and kw["json_mode"] is True
        prompt = pc.call_args.args[0]["messages"][0]["content"]
        assert "Stay positive" in prompt and "we met at school" in prompt
        assert "Campaign focus" in prompt and "Mumford & Sons" in prompt

    def test_write_text_no_llm(self):
        with patch("pipeline.llm_client.pool_usable", return_value=False):
            assert T.write_text("b", "t") is None

    def test_recipe_describes_auto_text(self):
        from campaign.recipe import describe_op
        assert "written by AI" in describe_op({"op": "text_overlay", "text": "{auto}",
                                               "role": "hook", "position": "top"})

    def test_auto_text_validates_as_ready(self):
        from campaign.recipe import validate_recipe
        v = validate_recipe({"mode": "clip", "ops": [{"op": "text_overlay", "text": "{auto}",
                                                      "role": "hook"}], "export": ["tiktok"]})
        assert v["ok"], v

    def test_parser_prompt_has_auto_rule(self):
        from campaign.brief_parser import SYSTEM_PROMPT
        assert '"{auto}"' in SYSTEM_PROMPT


# ── overlay styling ──────────────────────────────────────────────────────────

class TestOverlayStyle:
    def _style(self, ass, name):
        line = next(ln for ln in ass.splitlines() if ln.startswith(f"Style: {name},"))
        return line.split(":", 1)[1].split(",")

    def _ass(self, **o):
        from pipeline.compositor import build_overlay_ass
        return build_overlay_ass(1080, 1920, 10, overlays=[
            dict({"text": "Came for *Kacey*, stayed for *Estevie*. Who is she??",
                  "role": "hook", "position": "top"}, **o)])

    def test_default_is_tiktok_box_style_below_phone_ui(self):
        ass = self._ass()
        f = self._style(ass, "T0")
        size, primary, outline_c, border, outline, align, mv = (
            int(f[2]), f[3], f[5], f[15], int(f[16]), f[18], int(f[21]))
        assert f[1] == "Montserrat ExtraBold" and size >= 70
        assert border == "3" and outline_c == "&H00FFFFFF"        # white box per line
        assert primary != "&H00FFFFFF"                            # dark text on it
        assert outline >= 14 and align == "8" and mv >= int(1920 * 0.13)

    def test_balanced_lines_and_highlight(self):
        ass = self._ass()
        line = [ln for ln in ass.splitlines() if ln.startswith("Dialogue: 1")][0]
        body = line.split(",,", 1)[1]
        assert body.count(r"\N") == 2                             # 3 even lines
        assert "*" not in body and body.count("{\\c&H") == 2      # Kacey + Estevie coloured
        assert "Who is she??" in body.split(r"\N")[-1]            # sentence on its own line

    def test_clean_style_outline_and_shadow(self):
        f = self._style(self._ass(style="clean"), "T0")
        assert f[15] == "1" and int(f[16]) >= 6 and int(f[17]) >= 3

    def test_dark_box_style_translucent(self):
        f = self._style(self._ass(style="dark_box"), "T0")
        assert f[15] == "3" and f[5] != "&H00000000"

    def test_balance_text(self):
        from pipeline.compositor import balance_text
        assert balance_text("Why is nobody talking about this band", 27) == \
            ["Why is nobody talking", "about this band"]
        assert balance_text("50 years", 27) == ["50 years"]
        lines = balance_text("Estevie playing her unreleased single \"This Man\"", 27)
        assert all(len(x) <= 27 for x in lines) and len(lines) <= 3


# ── routing + duplicate jobs ─────────────────────────────────────────────────

class TestJobsGuards:
    def test_long_video_in_edit_mode_warns(self):
        from campaign.jobs import route
        mode, why = route({"duration": 600}, {"mode": "edit"})
        assert mode == "edit" and "only its opening is used" in why
        mode, why = route({"duration": 60}, {"mode": "edit"})
        assert "⚠️" not in why
        assert route({"duration": 600}, {})[0] == "clip"

    def test_duplicate_job_refused(self):
        from campaign import jobs
        with patch.object(jobs.service, "ready_recipe", return_value=({"ops": []}, 1)), \
                patch.object(jobs, "expand_sources", return_value=["/v/a.mp4"]), \
                patch.object(jobs.job_repo, "list_jobs",
                             return_value=[{"id": "abcdef123", "status": "running"}]), \
                patch.object(jobs.job_repo, "list_inputs",
                             return_value=[{"source": "/v/a.mp4"}]), \
                patch.object(jobs.job_repo, "insert_job") as ins:
            with pytest.raises(ValueError, match="already running"):
                jobs.create_job("cid", ["/v/a.mp4"])
        ins.assert_not_called()

    def test_finished_job_does_not_block_rerun(self):
        from campaign import jobs
        with patch.object(jobs.service, "ready_recipe", return_value=({"ops": []}, 1)), \
                patch.object(jobs, "expand_sources", return_value=["/v/a.mp4"]), \
                patch.object(jobs.job_repo, "list_jobs",
                             return_value=[{"id": "abc", "status": "done"}]), \
                patch.object(jobs.job_repo, "insert_job", return_value={"id": "new"}), \
                patch.object(jobs.job_repo, "add_input"):
            assert jobs.create_job("cid", ["/v/a.mp4"])["id"] == "new"


def test_ui_apps_build():
    import gradio as gr
    with gr.Blocks():
        U.build_campaign_tabs()
    import ui  # noqa: F401  (builds the whole app, incl. the Clip review tab)


def test_no_dropdown_preselects_a_campaign_job_or_video():
    # Gradio 6 preselects the first choice unless value=None: a job then ran on
    # whichever campaign was listed first (the 15 Oct "everything starts with ACME" run)
    import gradio as gr
    import ui
    dds = [b for b in ui.app.blocks.values() if isinstance(b, gr.Dropdown)
           and b.label != "Kind"]                      # "Kind" (asset type) defaults to auto
    labels = " | ".join(b.label for b in dds)
    assert "Campaign" in labels and "Job" in labels and "recent video" in labels
    assert all(b.value is None for b in dds), [(b.label, b.value) for b in dds]


def test_parser_keeps_auto_text_but_still_drops_invented_text():
    # 16 Oct: the grounding filter removed "{auto}" from every "text on screen required"
    # brief ("text '{auto}' dropped — it is not in the brief")
    from campaign import brief_parser as B
    raw = {"mode": "clip", "export": ["tiktok"],
           "ops": [{"op": "text_overlay", "text": "{auto}", "role": "hook"},
                   {"op": "text_overlay", "text": "HOOK_TEXT", "role": "hook"}]}
    with patch.object(B, "_llm_parse", return_value=raw):
        r = B.parse_brief("Text on screen is required for every post. Write your own.", [],
                          use_llm=True)
    texts = [o.get("text") for o in r["recipe"]["ops"] if o["op"] == "text_overlay"]
    assert texts == ["{auto}"] and r["ok"], r


def test_caption_written_even_without_auto_text():
    from campaign import onscreen_text as T
    rec = {"ops": [{"op": "logo", "asset": "l.png"}]}
    caps = []
    with patch.object(T, "write_post",
                      return_value={"text": "x", "caption": "@daniaustin @jordanjramirez"}):
        out, used, _ = T.resolve_auto_text(rec, "Tag @daniaustin on every post", "t",
                                           captions=caps)
    assert caps == ["@daniaustin @jordanjramirez"] and not used and out["ops"] == rec["ops"]
    with patch.object(T, "write_post", side_effect=AssertionError("LLM called")):
        T.resolve_auto_text(rec, "brief", "t")                  # no captions list: no call
        T.resolve_auto_text(rec, "brief", "t", captions=[], generate=False)


def test_highlighted_phrase_never_split_across_lines():
    from pipeline.compositor import balance_text
    for t in ("Cold-called *Chris Stapleton* just because he loves him",
              "House on fire? *Eddie Vedder's lighter* comes first",
              "A decade of heart for *Children in Conflict*"):
        lines = balance_text(t, 20)
        assert all(ln.count("*") % 2 == 0 for ln in lines), lines
        assert " ".join(lines) == t
