# test_fusion.py — Phase 3: merge, scoring, selection, post-refine dedup.
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from pipeline.fusion import (temporal_iou, merge_overlaps, score_candidate,
                             rank_and_select, deduplicate_refined,
                             temporal_overlap_seconds, limit_stitch_ranges)


def _c(a, b, hook="Hook here!", **kw):
    d = {"source_ranges": [[a, b]], "sentence_ids": [0],
         "hook": hook, "main_idea": "idea", "payoff": "payoff done.",
         "required_context": "", "rationale": "r", "uncertainty": 0.2,
         "source": "outline"}
    d.update(kw)
    return d


class TestIoU:
    def test_identical(self):
        assert temporal_iou(_c(0, 10), _c(0, 10)) == 1.0

    def test_disjoint(self):
        assert temporal_iou(_c(0, 10), _c(20, 30)) == 0.0

    def test_partial(self):
        assert abs(temporal_iou(_c(0, 10), _c(5, 15)) - 1 / 3) < 1e-9

    def test_overlap_seconds(self):
        assert temporal_overlap_seconds(_c(0, 10), _c(5, 15)) == 5


class TestMerge:
    def test_merge_high_overlap(self):
        cands = [_c(0, 10, uncertainty=0.5), _c(1, 11, uncertainty=0.1)]
        kept, dropped = merge_overlaps(cands, iou_threshold=0.7)
        assert len(kept) == 1 and len(dropped) == 1
        assert dropped[0][0]["fusion_status"] == "rejected"

    def test_no_merge_distinct(self):
        kept, dropped = merge_overlaps([_c(0, 10), _c(50, 60)], 0.7)
        assert len(kept) == 2 and not dropped


class TestScoring:
    def test_filler_opener_penalized(self):
        good = score_candidate(_c(0, 30, hook="Look at this result!"))
        bad = score_candidate(_c(0, 30, hook="So um well here we go"))
        assert good["opening"] > bad["opening"]

    def test_pronoun_opener_hurts_standalone(self):
        a = score_candidate(_c(0, 30, hook="It finally works now!"))
        b = score_candidate(_c(0, 30, hook="The motor finally works now!"))
        assert b["standalone"] > a["standalone"]

    def test_spike_cannot_rescue_incoherent(self):
        loud_bad = _c(0, 30, hook="It was um so yeah", payoff="",
                      required_context="needs part 1", uncertainty=0.9,
                      source="audio_event")
        loud_bad["provenance"] = {"energy_increase": 15.0}
        quiet_good = _c(40, 70, hook="Here is exactly how the coil works.",
                        payoff="And that triples the range.")
        assert (score_candidate(quiet_good)["total"] >
                score_candidate(loud_bad)["total"])

    def test_scores_are_heuristics(self):
        s = score_candidate(_c(0, 30))
        assert "note" in s and "total" in s
        assert abs(sum(s["weights"].values()) - 1.0) < 1e-9


class TestSelect:
    def test_target_is_maximum(self):
        cands = [_c(i * 100, i * 100 + 30, hook=f"Hook number {i} here!")
                 for i in range(6)]
        sel, rep = rank_and_select(cands, cfg={"discovery": {"target_clips": 2},
                                               "fusion": {"min_score": 0.0}})
        assert len(sel) == 2

    def test_quality_floor_returns_fewer(self):
        cands = [_c(0, 30, hook="So", payoff="", required_context="needs part 1",
                    uncertainty=0.95),
                 _c(100, 130, hook="Um", payoff="", required_context="needs part 2",
                    uncertainty=0.95)]
        sel, rep = rank_and_select(cands, cfg={"discovery": {"target_clips": 8},
                                               "fusion": {"min_score": 0.45}})
        assert len(sel) < 2
        assert any("below quality floor" in r for _, r in rep["skipped"])

    def test_same_topic_distinct_kept(self):
        # same topic words, non-overlapping ranges -> both kept
        cands = [_c(0, 30, hook="The wireless coil overheats quickly.",
                        main_idea="coil heat problem"),
                 _c(200, 230, hook="The wireless coil charges slowly.",
                        main_idea="coil speed problem")]
        sel, _ = rank_and_select(cands, cfg={"discovery": {"target_clips": 8},
                                              "fusion": {"min_score": 0.0}})
        assert len(sel) == 2

    def test_near_duplicate_dropped(self):
        # IoU = 20/40 = 0.5 hits the diversity threshold (below merge 0.7)
        cands = [_c(0, 30, hook="The coil works great today."),
                 _c(10, 40, hook="The coil works great today!")]
        sel, rep = rank_and_select(cands, cfg={"discovery": {"target_clips": 8},
                                               "fusion": {"min_score": 0.0}})
        assert len(sel) == 1
        assert any("near-duplicate" in r for _, r in rep["skipped"])


class TestStitchLimits:
    def _cfg(self, gap=45, total=90):
        return {"similarity": {"max_stitch_gap_seconds": gap,
                               "max_total_seconds": total}}

    def test_distant_dropped(self):
        segs = [{"start_time": 200.0, "end_time": 250.0}]
        kept, dropped = limit_stitch_ranges((0.0, 30.0), segs, self._cfg())
        assert kept == [] and len(dropped) == 1
        assert "related, not continuation" in dropped[0][1]

    def test_adjacent_kept(self):
        segs = [{"start_time": 32.0, "end_time": 50.0}]
        kept, dropped = limit_stitch_ranges((0.0, 30.0), segs, self._cfg())
        assert len(kept) == 1 and not dropped

    def test_total_capped(self):
        segs = [{"start_time": 30.0, "end_time": 70.0},
                {"start_time": 70.0, "end_time": 120.0}]
        kept, dropped = limit_stitch_ranges((0.0, 30.0), segs, self._cfg())
        assert len(kept) == 1 and len(dropped) == 1
        assert "max_total" in dropped[0][1]

    def test_bad_ranges_skipped(self):
        segs = [{"start_time": 5.0}, {"start_time": 40.0, "end_time": 30.0}]
        kept, dropped = limit_stitch_ranges((0.0, 30.0), segs, self._cfg())
        assert kept == [] and len(dropped) == 2

    def test_defaults_without_config(self):
        kept, dropped = limit_stitch_ranges((0.0, 30.0),
                                            [{"start_time": 500.0,
                                              "end_time": 510.0}], {})
        assert kept == [] and dropped


class TestPostRefineDedup:
    def test_expanded_overlap_deduped(self):
        # refined boundaries expanded into near-identical coverage
        clips = [
            {"clip_number": 1, "start_time": 0, "end_time": 60,
             "hook": "the coil works", "reason": "r",
             "source_ranges": [[0, 60]],
             "provenance": {"fusion_score": 0.8}},
            {"clip_number": 2, "start_time": 10, "end_time": 70,
             "hook": "the coil works well", "reason": "r",
             "source_ranges": [[10, 70]],
             "provenance": {"fusion_score": 0.5}},
        ]
        kept, dropped = deduplicate_refined(clips)
        assert [c["clip_number"] for c in kept] == [1]
        assert len(dropped) == 1 and "duplicate of clip 1" in dropped[0][1]

    def test_distinct_survive(self):
        clips = [
            {"clip_number": 1, "start_time": 0, "end_time": 30,
             "hook": "a", "reason": "r", "source_ranges": [[0, 30]],
             "provenance": {"fusion_score": 0.8}},
            {"clip_number": 2, "start_time": 200, "end_time": 230,
             "hook": "b", "reason": "r", "source_ranges": [[200, 230]],
             "provenance": {"fusion_score": 0.5}},
        ]
        kept, dropped = deduplicate_refined(clips)
        assert len(kept) == 2 and not dropped


class TestNoneHookReport:
    def test_none_hooks_do_not_crash_report(self):
        # Regression: audio cues with hook=None crashed the merged/skipped
        # report with TypeError: 'NoneType' object is not subscriptable.
        cands = [_c(0, 30, hook=None), _c(1, 31, hook=None),
                 _c(100, 130, hook=None)]
        selected, report = rank_and_select(
            cands, "", cfg={"discovery": {"target_clips": 8},
                            "fusion": {"min_score": 0.0}})
        assert isinstance(report["merged"], list)
        assert isinstance(report["skipped"], list)
