# Implementation Plan — Completion Log (all phases done)

> Synced 06 Oct 2026. The original MiniMax-era phase list (input → campaign → dynamic pipeline → config → UI → testing) is long finished. What follows is the actual build history: **Phase 0 baseline repairs + Phases 1–7**, ending at the current 11-stage pipeline (Groq LLM + local embeddings, 183/183 tests, verified end-to-end on a 324s sample). Keep this file as history; `AGENT_CONTEXT.md` is the live spec.

## Phase 0 — Baseline: repairs + first verified e2e ✅
- Verified the real 11-stage execution path; applied migration v4 (never applied).
- Repaired 5 stale test files (wrong import paths, teardown deleting real assets, obsolete APIs).
- Fixed campaign-parser crash on Groq 429 (empty-description → safe defaults).
- Full e2e via `rerun_from_embed.py`: 4 clips refined/rendered; rejected-skip, stitched-SRT, resume verified. **61/61 tests green** at that point.

## Phase 1 — Contextual discovery ✅
- New `pipeline/outline.py` (windowed section outlines: topics, Q&A, story arcs, speaker turns; LLM + deterministic fallback; video-level combine), `pipeline/discovery.py` (two-pass sentence-ID candidates, 15–90s span validation, non-contiguous off by default), `pipeline/fingerprints.py` + `stage_fingerprints` table, migration v5 (outlines, candidates, audio_events, fingerprints, `clips.provenance`).
- `analyze()` is now two-pass with legacy per-chunk fallback.
- Live catch: unbounded spans produced 283s candidates → span validation; re-verified (8 candidates, 19–40s).

## Phase 2 — Audio events ✅
- New `pipeline/audio_events.py` (200ms RMS, rolling-median baseline, silence-safe, click/sustain suppression, source timestamps; ±25/10s transcript windows, speech-coverage gating, sentence-aligned bounds; one batched LLM enrich) + `pipeline/audio_classifier.py` (YAMNet-optional, default off, never fakes).
- Checkpointed `audio_events` stage in both entry points. Verified live: 2 spikes (+9.4/+10.7dB) → 2 cues in 4.7s.

## Phase 3 — Fusion / ranking ✅
- New `pipeline/fusion.py` (IoU≥0.7 merge, 7-component editorial scores with audio as 5% bonus, diversity select preserving same-topic-distinct clips, 0.45 quality floor with reasons, post-refine dedup with coverage metric). Wired into analyze + refine; stitch caps (`max_stitch_gap_seconds: 45`, `max_total_seconds: 90`) enforced in clipper + refine after a live 120s Frankenstein stitch.
- Verified live: 6 candidates fused, near-dup rejected with reason.

## Phase 4 — Captions / transcription ✅
- New `pipeline/captions.py` (phrase grouping, balanced 2-line ASS karaoke, unicode-safe, face-aware placement reusing framing analysis, SRT preserved, drawtext fallback; fixed config-passthrough + Windows ASS-path bugs) + `pipeline/glossary.py` (traceable display corrections into transcriber/SRT/ASS).
- `bench_transcribe.py`: base CPU 10.7s vs CUDA 1.5s, identical words → `device: auto` default. turbo/large-v3 uncached → skipped, never silently downloaded. ASS verified live (21 cues, one pass).

## Phase 5 — Edit plans / render / audio ✅
- `pipeline/editplan.py` + migration v6 (`edit_plans`, versioned) + repo; `pipeline/polish.py` consolidates subs/logo/music/fades/normalization into one ffmpeg pass (−16 LUFS loudnorm, sidechain ducking, limiter, no speech fade, `-an` on silence, music loop).
- Verified live: 14.8s single pass, 1080×1920, duration + audio preserved.

## Phase 6 — Review UI ✅
- `pipeline/review.py` (boundary/layout/caption overrides with timeline validation, word-count-preserving caption edits, candidate accept/reject, per-edit plan versions) + Gradio **Review & Correct** tab + `render_clips(clip_numbers=[...])` selective re-render. Unit-verified; data-load verified live.

## Phase 7 — Adapters (optional, off) ✅
- `pipeline/speakers.py` (explicit-map only; auto-match reports unevaluated; turn stability + overlap/hold rules) + `pipeline/visual.py` (keyframe sampling verified live; backends report disabled/unavailable, never fake). Both default off, 15 tests green.

## Final verification state ✅
- **183/183 pytest pass.** Three full e2e runs on the 324s sample (final: 3 clips, render 406s). Migrations v4–v6 applied with graceful fallbacks.
- Benchmark (same source): baseline 4 clips/25–65s, word-captions, blur default → final 3 clips/35–50s, fusion 0.53–0.81, 1 near-dup rejected, 2 audio cues fused, 18–31 phrase cues, plans v1, single-pass polish. Target 8 is a maximum — floor + dedup explain fewer, by design.
- Evidence kept: `output/` finals, `output/phase0|phase1|phase4_monster` snapshots, `output/previews/`, `output/bench_transcription.json`, `preview_framing.py`, `demo_faces.py`.
- Also fixed live along the way: MediaPipe `mp.Image` move (detector silently finding nothing), campaign 429 crash, 283s spans, 120s stitch, ASS path/config bugs.
- Honest gaps (no claims beyond these): no visual/listening inspection here (machine checks only); single-speaker sample only (stacked/switching/diarization/YAMNet untested on real footage); large-v3/turbo unbenchmarked; selective re-render + enrich-under-429 not exercised live; Groq 429s needed fallbacks twice (deterministic paths held).

## Earlier history (original plan, all done)
- Phase 1 (input handling), Phase 2 (campaign parser + templates), Phase 3 (dynamic pipeline logic), Phase 4 (config + assets), Phase 5 (Gradio UI), Phase 6 (integration testing) — all implemented; superseded and extended by Phases 0–7 above. Provider migrated MiniMax → NVIDIA NIM → **Groq + local embeddings** (current).
