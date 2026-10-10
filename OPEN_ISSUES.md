# Open Issues

Status as of 10 Oct 2026, after two real end-to-end runs — MrBeast "Eat Everything In A Grocery
Store" (58 min → 14 clips) and MrBeast "100 Kids Vs World's Strongest Man!" (27 min → 13 clips,
64 min end to end on this Mac) — and the fixes/Phase D framing work that followed.
Everything fixed during that work is listed in `AGENT_CONTEXT.md` §9 (v4.2–v4.6) and `implementation_plan.md` (Phases 9–14). This file
tracks only what is **still open**, most important first within each section.

Measure framing changes with `python tools/framing_audit.py <video_id>` and judge the flagged
moments in the `--debug-framing` videos (`work/<video>/clips/clip_N_vertical_debug.mp4`).

---

## 0. Fix roadmap for the HIGH / MEDIUM issues (agreed plan, not started)

Order matters: step 1 gives a trustworthy measurement that steps 3–6 are tuned and proven against.

| Step | Fixes | Work | Effort | Needs from the user |
|---|---|---|---|---|
| 1 | 1.1 | **Ground-truth framing measurement**: a small labelling page (play a clip, key "who is speaking" per second: left/right/both/nobody/off-screen → labels JSON); `tools/framing_audit.py --truth labels.json` reports real speaker accuracy instead of the lip-motion proxy. Label set: 3–4 clips each from a podcast, an interview, a MrBeast-style video. | ½ day | ~30–45 min labelling; a podcast + an interview link |
| 2a | 2.1 | **LLM consistency**: temperature 0 for boundary calls; reject a clip only when two independent verdicts agree; cache verdicts per candidate so resumes are stable. | 2 h | — |
| 2b | 2.2 | **Top-up after rejections**: if fewer clips survive refinement than `target_clips`, promote the next-best fusion candidates (the "skipped" list) and refine them. | 2 h | — |
| 2c | 4.1 | **Render speed**: hardware encoder `h264_videotoolbox` for the intermediate vertical file; 5 fps detection in single-face scenes; debug video opt-in (already). Measure before/after (target ~2x). | ½ day | — |
| 2d | 5.1 | **UI end-to-end pass** against workspaces (Generate + Review tabs), fix breakages, add a UI smoke test. | ½ day | — |
| 2e | 5.2 | **CI first run**: needs a pushed GitHub repo; then fix install/test issues on the runner. | 2 h | `git init` + push |
| 3 | 1.3, 1.5 | **Person/body tracking**: YOLO11n (~6 MB ONNX, OpenCV DNN) a few times/s; link bodies to face tracks; follow the body when the face is lost; in action scenes the most-moving person is the subject (not still spectators). Saliency kept only for person-less shots; OpenCV text detector to ignore big on-screen graphics. Cost ~+20–30% render time. | 2 days | — |
| 4 | 1.2 | **Trained speaker model (LR-ASD)**: ~1M params, CPU, existing torch; scores each face track from mouth crops + audio; feeds `direct_shots` (director/cut logic unchanged); lip-motion heuristic stays as fallback. Keep only if step-1 labels prove it better. | 2–3 days | — |
| 5 | 1.4 | **Shot-level crop optimisation**: pick each shot's crop path to balance "speaker centred" vs "no sliced faces" over the whole shot (instead of per-sample nudges), with ±10% per-shot zoom to include/exclude a neighbour. Target < 4% sliced, < 6% off-centre. | 1 day | — |
| 6 | 1.9 | **Multi-genre tuning**: run a podcast + an interview, tune director/crowd thresholds against step-1 labels; add a `genre` preset (auto-detected from shot length / people count, or set per campaign) if genres need different settings. | 1 day | the two links |

Not on the roadmap: 2.3 (provider switch planned — small change in `llm_client.py`), LOW items.

---

## 1. Framing (the main open area)

### 1.1 Wrong-person framing is not proven fixed — HIGH
- **Evidence:** audit proxy "big face clearly talking but not in the crop": video 1 29% before
  Phase D, 30% after; video 2 (different footage, Phase D) 24%. Visual before/after checks on clips 2/7/9 did
  improve (presenter back in frame, no extras in split-screens, crowd shots instead of random
  faces), so the proxy and the eye disagree.
- **Why it's hard to tell:** the proxy itself is noisy — camera moves and faces appearing
  create fake lip-motion spikes (e.g. 6.73 on a face entering frame), and it penalises the
  director for holding a shot through fast back-and-forth (correct editing).
- **Next step:** hand-label "who is speaking" per second for 3–4 clips of a real podcast /
  interview (ground truth), then measure against that instead of the proxy.

### 1.2 Lip-motion speaker detection is a heuristic — HIGH
- `pipeline/active_speaker.py` (`mouth_motion`): mouth-region change minus eye-region change,
  gated by audio loudness. Fooled by camera motion, chewing (eating videos), off-screen
  narration (voiced audio, nobody on screen talking) and anything under ~4.5% face width
  (those faces are now ignored, `speaker_min_face_ratio`).
- **Fix:** plug a trained audio-visual speaker model (LR-ASD or TalkNet-ASD) into
  `direct_shots` as an alternative scorer. Needs torch (already installed) + model weights.

### 1.3 No body / person tracking: the action is often not a face — HIGH
- Sports/action footage: the subject doing the thing (a player mid-dunk, face turned up and
  blurred) has no detectable face, while spectators' faces at the bottom of the frame do — so
  face-driven framing follows the spectators (video 2, clip 3 @0.1s; clip 8 @3.7s).
- Mitigated: after 1s without faces the crop follows motion-weighted saliency, and holding a
  stale position dropped from ~24% of samples (video 1) to 4.9% (video 2). But a scene that
  HAS faces keeps following them even when the action is elsewhere.
- **Fix:** a small person detector (e.g. YOLOv8n ONNX via OpenCV DNN, or MediaPipe pose) to
  keep tracking the body; link body tracks to face tracks.

### 1.4 Faces sliced by the crop edge ~8–9%, followed face off-centre ~11% — MEDIUM
- Video 1: 8.4% → 8.0% after `_deslice_target`; video 2: 9.2% sliced, 10.7% of single-face
  samples have the face > 18% off-centre (the deslice nudge trades centring for not slicing
  a neighbour; the deadzone also lets the face drift before the camera follows). The nudge is capped at 25% of the crop width and must
  keep the subject inside, so tight groups can't be resolved; crowd shots slice whoever is at
  the crop edge.
- **Ideas:** allow a slightly tighter/looser crop per shot (zoom) to include/exclude a
  neighbour; score crop positions over the whole shot instead of per sample.

### 1.5 Saliency fallback is still heuristic — MEDIUM
- `salient_cx` = spectral-residual saliency × motion (adjacent frame), densest crop-width
  window, top/bottom title bands ignored. Fixed during video 2: ceiling lights beating a
  dunking player (motion weighting) and crops landing on the empty gap between two clusters
  (densest window instead of centre of mass). Still fooled by on-screen graphics in the middle
  band (a big "FINISH" graphic) and by whole-frame camera motion. Not used to pick positions
  inside crowd shots that have faces (only for their face-less gaps).
- **Ideas:** object detection for products/people; penalise text regions (OCR/EAST).

### 1.6 Crowd shots: subject sits off-centre — LOW
- `_group_target` ties go to the weighted centroid, so a lone central subject (the referee in
  MrBeast clip 9) can end up in the left/right third of the crop rather than centred.

### 1.7 Fast-cut footage gets ~1 shot per scene — LOW (by design)
- MrBeast scenes average ~2s; with `min_shot_seconds: 1.6` + switch cost the director
  rarely cuts within a scene, so quick exchanges stay on one person. Correct for most
  editing, but worth re-checking on long podcast takes (where it should cut per turn).

### 1.8 Source-limited framing — WON'T FIX (documented)
- A face at the very top of the source frame can't get headroom with a full-height crop;
  zooming out would need bars, which `full_screen_vertical` forbids.

### 1.9 Thresholds tuned on one genre — MEDIUM
- All Phase D thresholds (`speaker_min_face_ratio`, `crowd_*`, `speaker_min_clear_seconds`,
  `deslice_*`, saliency bands/motion weight, `_CutDetector` 27/0.85/2.5) were tuned on two
  MrBeast videos — fast-cut, crowded, handheld. Validate on a sit-down podcast and a 1-on-1
  interview (where the director should cut per speaking turn) before trusting them.

---

### 1.10 Campaign compositor: non-9:16 reframes use one static crop — MEDIUM
- `pipeline/compositor.py`: wide → 9:16 uses the face-tracking virtual camera, but 1:1 / 4:5 /
  16:9 targets get a single crop centred on the main face for the WHOLE video (no per-scene
  tracking), because `framing.render_vertical` only renders 1080×1920. Fix: parameterise the
  framing output size (OUT_W/OUT_H) so the same virtual camera serves every aspect.

---

## 2. Clip content / LLM

### 2.1 LLM verdicts are non-deterministic — MEDIUM
- Same clip, same transcript: run 1 rejected it ("story never resolves"), run 2 extended it
  to the payoff (+20s). Boundary calls use temperature 0.1.
- **Fix options:** temperature 0, or ask twice and require agreement before rejecting.

### 2.2 `reject_incomplete` can thin out clip counts — MEDIUM
- On a 150s smoke test, 8 candidates → 2 clips (1 unfinished + 5 genuine duplicates once a
  clip was extended). On full videos this should be fine, but watch the clip count; there is
  no top-up after refinement rejects clips (`topup_clips.py` is manual).

### 2.3 Free-tier Groq limits — KNOWN (moving providers later)
- 8,000 tokens/min per key ≈ 1 big call per key per minute. Handled by waiting
  (`LLM_RATE_LIMIT_MAX_WAIT`, default 300s) but LLM stages are slow. When switching
  providers: `pipeline/llm_client.py` is OpenAI-compatible; review `_is_reasoning_model`
  (reasoning headroom) and JSON mode support for the new model.

### 2.4 Outline windows widen on long videos — LOW
- `outline.max_windows: 12` → a 58-min video got ~120 sentences per window (designed for
  40). Fine after the headroom fix, but on a paid tier more, smaller windows would be better.

### 2.5 Stitched continuations rarely used — LOW
- The similarity stage links many continuations, but only 1 of 14 MrBeast clips actually
  rendered multi-part. Evaluate whether stitching is worth its LLM calls.

---

## 3. Transcription

### 3.1 `small` still makes mistakes — LOW
- Fixed "galleries→calories" and "spill→spell" vs `base`; numbers/names can still be off
  (e.g. "201,700"). Use `transcription.glossary` for recurring names; `medium` was 2x slower
  and not better on our sample.
- No diarization (`pipeline/diarization.py` needs HF_TOKEN + pyannote; not wired in).

---

## 4. Performance

### 4.1 Rendering ~2–3.5x realtime on CPU — MEDIUM
- 58-min source → ~45 min to render 14 clips (with debug video). Main costs: per-frame cubic
  warp, libx264 encode at 1080x1920, debug video encode, face detection at 10 fps.
- **Options:** `h264_videotoolbox` hardware encoder on Mac, `interpolation: linear`,
  `debug_video` off (now the default), detection at 5 fps outside speaker scenes.

### 4.2 8 GB RAM — LOW
- ~1.9 GB swap used during runs. `small` Whisper + embeddings + framing run sequentially and
  fit; `medium`/`large` Whisper would be tight.

### 1.11 Campaign jobs share the global workspace with the legacy tabs — MEDIUM
- Clip-mode jobs run in the background worker and use `pipeline/workspace.current()` (the
  long-video pipeline's global "active video"). Running a Quick-clips / Clip-review action in
  the UI *while* a clip-mode job is rendering could switch the active workspace under it.
  Today: run one thing at a time. Fix: pass the Workspace explicitly through
  `render_clips` / `run_pipeline` instead of the module global.

### 1.12 Clip-mode reuse of cached analysis — LOW
- A clip-mode job reuses the source's earlier analysis when the clip spec (count, min/max,
  focus) matches the last campaign run (`work/campaigns/_analysis/<video8>.json`) or, with no
  focus, when enough existing clips fit. A different spec re-runs discovery + refinement
  (LLM calls — slow on free-tier keys). Clip-spec limits are applied as config overrides, so
  `max_duration` is respected by refinement; the compositor never re-cuts a clip.

### 1.13 Brief parsing still depends on the LLM's reading — MEDIUM
- The free-tier `gpt-oss-120b` occasionally drops a requirement, invents placeholder text, or
  sets settings the brief never stated (e.g. hook timing 0–3 s). Deterministic guards now catch
  the worst cases (dropped durations / platforms copied back from the brief, text not in the
  brief removed, roles corrected, made-up filename patterns ignored) and every recipe is shown
  in plain English before it is saved — but always read the recipe summary before running a job.

---

## 5. Untested since the refactor

### 5.1 Gradio UI (`ui.py`) not exercised end-to-end in a browser — HIGH before using the UI
- The campaign tabs (C7) are tested through their handlers (`test_campaign_ui.py`: create →
  assets → parse → save → check inputs → run → review → re-render → zip) and the app was
  launched and served all 6 tabs, but nobody has clicked through it in a browser yet. The
  legacy "Quick clips" and "Clip review" tabs (selective re-render via
  `render_clips(clip_numbers=…)`) still need a manual pass after the workspace refactor.

### 5.2 CI test job never run — MEDIUM
- `.github/workflows/ci.yml` now has a `tests` job (pgvector service, CPU torch, ffmpeg).
  It has not run yet (this checkout is not a pushed git repo); expect to fix install
  details on first push.

### 5.3 Legacy dev entry points use old global paths — LOW
- `pipeline/regen_srt.py:main()`, `pipeline/fast_burn.py:main()` (hard-coded old clip
  times), `bench_transcribe.py` (default `downloads/audio.wav`), `demo_faces.py`. Not used by
  the pipeline; update or delete.

---

## 6. Robustness / housekeeping

- **Transcript fingerprint uses mtime+size** (`pipeline/fingerprints.py:transcript_signature`):
  copying a transcript invalidates the outline cache; an edit of the same size isn't noticed.
  Switch to a content hash (will invalidate existing caches once).
- **YouTube downloads:** audio-403 after long video downloads is handled by one retry; repeated
  403s still fail (then try again later). Live streams record only 300s.
- **Harmless startup noise:** torchcodec "could not load libtorchcodec" and the objc
  duplicate-class warning (OpenCV vs PyAV both bundle libavdevice). Audio loads via librosa;
  webcam capture (the only affected feature) isn't used.
- **Pre-existing lint warnings** (unused imports/variables in `main.py`, `framing.py`,
  `boundaries.py`); CI's strict check (syntax/undefined names) is clean.
- **`.env` must use `DB_PORT=5433`** for the Docker database (another Postgres uses 5432 on
  this Mac) — documented in `.env.example` and README.

---

## 7. Found and fixed during the video 2 run (for the record)

- Stitched clips could exceed the 90s cap (95–97s: clips 1, 11, 14) — refinement extended each
  part but never re-checked the total. `refine_all_clips` now drops continuations that would
  push the total over `similarity.max_total_seconds`. (Applies to new runs; video 2's three
  long clips were rendered before the fix.)
- Crash → static centre crop for a whole clip (clip 11): saliency filled the x-path for a
  director shot whose subject never appeared, y stayed None. `track_path` now returns None for
  a never-seen subject (caller falls back) and defaults y.
- Split-screen showing the SAME face twice (clip 3 @3.2s): a second face appeared only for the
  last 0.5s of a close-up and still qualified. Split-screen now needs both people on screen
  simultaneously ≥ 60% of the shot (`stack_min_concurrent`) and non-overlapping panels.
- Saliency: motion-weighted + densest window (see 1.5).
- Downloads now prefer H.264 ≤ 1080p (`YT_MAX_HEIGHT`, default 1080): the video was offered in
  4K AV1, which would have made every decode step ~3–4x slower on this CPU.

