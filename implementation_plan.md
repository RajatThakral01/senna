# Implementation Plan — Completion Log (all phases done)

> Synced 10 Oct 2026. The original MiniMax-era phase list (input → campaign → dynamic pipeline → config → UI → testing) is long finished. What follows is the actual build history: **Phase 0 baseline repairs + Phases 1–8** (11-stage pipeline, Groq LLM + local embeddings), then **Phases 9–14** (OpusClip-style framing A–D, per-video workspaces/resume, and two real end-to-end runs on MrBeast videos), then **Phase 15** — the campaign pipeline C1–C8 (403 tests). Keep this file as history; `AGENT_CONTEXT.md` is the live spec.

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

## Phase 8 — Scale & hardening ✅
- `pipeline/llm_client.py`: 4-key rotation (slots by measured usage: key1 outline/discovery/enrich, key2 similarity, key3 campaign/analyzer/boundaries, key4 fallback; values never logged; placeholders excluded). All 7 LLM call sites rewired; 4xx fast-fail everywhere; pool-aware guards. Verified live: 429s absorbed mid-run via rotation.
- Deterministic offline path: `_deterministic_section_clips` top-up + `topup_clips.py` resume; fusion `None`-hook hardening. Verified live on a 58-min video with a dead (403) key: 15 top-up + 3 audio clips.
- `input/input_handler.py`: HD enforcement (`--js-runtimes node/deno`, `height>=720` ladder, ffprobe gate, loud SD refusal after 360p incident). `validate_fullscreen.py` render checks + `render_crop_debug`. Framing truncation guard. Suite 220/220.

## Phase 9 — Framing A: detection, tracking, smooth camera ✅ (09 Oct 2026)
- Root causes of "odd / laggy" framing found in `framing.py`: MediaPipe box bottom bug (`y2 = bb.height`) dropping most faces; selfie-range model; "lowest track id" subject choice; track ids churning on any missed sample; split panels squashed ~1.8x; per-sample EMA + linear interpolation (velocity steps every 0.25s); inaccurate seeking.
- Fixes: YuNet detector, sequential decode at 0.1s, persistent `FaceTracker`, offline per-scene virtual camera (`plan_camera_axis`: median → deadzone → zero-phase Gaussian → pan cap), sub-pixel `warpAffine`, panels at panel aspect. Synthetic benchmark: jerk 7.0 → 0.7 px/frame², 9 → 2 track ids under dropouts.

## Phase 10 — Framing B: active speaker ✅
- `pipeline/active_speaker.py`: lip motion (mouth vs eye band) gated by audio voicing; Viterbi director with switch cost + min shot; hard cuts per turn; two-shot on overlapping speech. Synthetic 2-person test: 94% frame accuracy.

## Phase 11 — Framing C: shot styling + debug ✅
- Face-sized auto-zoom close-ups, eased emphasis punch-ins, cuts snapped to pauses (97% accuracy on the synthetic test), side-by-side `--debug-framing` video sharing `_frame_rects` with the renderer.

## Phase 12 — Environment + first real run (MrBeast 58 min → 14 clips) ✅
- Setup on macOS arm64 / 8 GB: `.venv`, Docker pgvector on port 5433, embedding model download, Whisper models pre-fetched.
- Issues hit and fixed: YouTube audio 403 after a long video download (HQ retry); `.env` DB port; LLM JSON truncated by hidden reasoning tokens (headroom + JSON mode); free-tier 429s on all keys (wait-and-retry); director crash on sub-0.5s scenes; 341 fake scene cuts in 86s (new adjacent-frame `_CutDetector`, 6/6 real cuts, 0 false on test footage); Mac sleep pausing runs.
- Discovered on review: shared global paths (second video would reuse the first video's audio and old finals) → per-video workspaces; no resume → resume/`--fresh`/`--from-stage`; clips ending before the payoff (ambiguous verdict) → selection-complete verdict + 30s payoff extension + reject; Whisper `base` errors → `small`/int8; dedup comparing against rejected clips.

## Phase 13 — Framing D: wrong person / random split-screens ✅
- Speaker evidence only from faces ≥4.5% width (not edge-clipped), absolute lip floor, minimum clear-speech time, two-shot only for exactly two eligible people, prominence prior (size × centrality), crowd mode (`group` shots), deslice (no half faces), saliency fallback for faceless shots. Measured on the 14 clips: random split-screens 86% → 26% of split frames, head cuts 3.0% → 0.3%; wrong-person proxy unchanged (proxy shown to be noisy; visual checks improved).
- `tools/framing_audit.py`, `OPEN_ISSUES.md`, CI pytest job, `db/init_db.sh`, hermetic tests.

## Phase 14 — Second real run (MrBeast 27 min → 13 clips, 64 min) ✅
- H.264 ≤1080p download preference (4K AV1 offered); stitched clips re-checked against the 90s total; crash when a shot's subject never appeared (→ clip fell back to static crop); split-screen showing the same face twice (both people must be on screen simultaneously, panels must not overlap); saliency motion-weighted + densest window (dunker instead of ceiling lights, strongman instead of empty field); crowd shots follow saliency through face-less gaps.
- Audit: holding a stale crop 4.9% (was ~24%), head cut 0.5%, wrong-person proxy 24%. User review: "the output looks good now".

## Phase 15 — Campaign pipeline C1–C8 ✅ (10 Oct 2026)
- Campaigns + asset library + jobs/deliverables (migration v7), versioned edit recipe + LLM brief parser with questions and deterministic guards, recipe compositor (`pipeline/compositor.py`), edit mode, clip mode on the existing pipeline, router + background worker + CLI, QA on every file, manifest / posting text / zip, 4 campaign UI tabs, C8 ops (speed, colour, watermark, intro, outro, end card) + variants.
- Real runs: edit-mode samples from real footage (incl. all C8 ops × 2 hook versions) and clip mode on the MrBeast video — all QA checks pass. Bugs they found (vertical→16:9 strip crop, CTA over captions, stale standalone render reused, relative bookend paths, LLM filename pattern / invented text) are fixed. Full log: `CAMPAIGN_PIPELINE_PLAN.md` §11.

## Verification state (10 Oct 2026) ✅
- **403 pytest pass** (300 at the end of Phase 14) (`.venv/bin/python -m pytest -q`; DB round-trip tests use the Docker DB).
- Real end-to-end runs: two MrBeast videos (above); resume (~2s for a finished video), `--from-stage refine`, selective re-render all exercised live. Media of both runs deleted after review; logs kept in `logs/`.
- Not yet exercised: Gradio UI since the workspace refactor, CI test job, diarization/YAMNet, sit-down podcast footage. Full list: `OPEN_ISSUES.md`.

## Earlier verification (≤ 06 Oct 2026)
- 220/220 tests; three e2e runs on a 324s DIY sample (3 clips, render 406s) and one 58-min run (25 clips, 23 finals). Evidence folders referenced then (`output/phase0|…`, `output/previews/`) are not part of this checkout.
- Also fixed then: MediaPipe `mp.Image` move, campaign 429 crash, 283s spans, 120s stitch, ASS path/config bugs.

## Earlier history (original plan, all done)
- Phase 1 (input handling), Phase 2 (campaign parser + templates), Phase 3 (dynamic pipeline logic), Phase 4 (config + assets), Phase 5 (Gradio UI), Phase 6 (integration testing) — all implemented; superseded and extended by Phases 0–7 above. Provider migrated MiniMax → NVIDIA NIM → **Groq + local embeddings** (current).
