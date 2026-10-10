# 🤖 Agent Implementation Context: Viral Clips Automator

**Version:** 5.0
**Date:** 10 Oct 2026
**Status:** 11-stage pipeline verified end-to-end on two real long-form videos (58 min → 14 clips,
27 min → 13 clips); per-video workspaces + resume; OpusClip-style framing (Phases A–D);
**300/300 tests green**.
**Purpose:** single source of truth for the agent maintaining this project — what it does, how
every part works, where things live, what is verified and what is not. Open problems live in
**`OPEN_ISSUES.md`**; history in §9 and `implementation_plan.md`.

> **Built — campaign pipeline (C1–C8 ✅, 10 Oct 2026):** data model (`db/migration_v7.sql`, `db/repositories/campaign_repo.py` + `job_repo.py`), `campaign/service.py` (campaigns, assets, recipes), `campaign/recipe.py` + `presets.py` + `brief_parser.py`, compositor `pipeline/compositor.py`, `campaign/edit_mode.py`, `campaign/clip_mode.py`, `campaign/jobs.py` (router, runner, worker), `campaign/qa.py`, `campaign/delivery.py`, CLI `campaign/cli.py` (`python main.py campaign …` / `job …`), UI `campaign_ui.py` (tabs 1–4 of `ui.py`). Tests `test_campaigns.py`, `test_recipe.py`, `test_compositor.py`, `test_edit_mode.py`, `test_jobs.py`, `test_qa.py`, `test_campaign_ui.py` (suite 403). Details + progress log: `CAMPAIGN_PIPELINE_PLAN.md` §11.
> **Next:** a browser pass through the campaign UI (`OPEN_ISSUES.md` 5.1) and real campaign trials; then the fix roadmap for open HIGH/MEDIUM issues: `OPEN_ISSUES.md` §0 (parked by the user).

---

## 1. PROJECT OVERVIEW

Turns long-form video (podcasts, interviews, challenge videos, streams) into 9:16 vertical clips
for TikTok / Reels / Shorts, aiming at OpusClip-level output: complete stories (hook → payoff),
the speaker framed full-screen, smooth "virtual camera" motion, karaoke captions, one-pass polish.

**Stack:** Groq LLM (`openai/gpt-oss-120b`, OpenAI-compatible) for understanding text; **local**
SentenceTransformer embeddings (`models/Qwen3-Embedding-0.6B`, 1024-dim) in PostgreSQL +
pgvector; WhisperX (`small`, int8 on CPU) for word-level transcripts; OpenCV (YuNet faces,
spectral-residual saliency) + FFmpeg for cutting, framing, captions and polish; Gradio UI.

**Verified runs (real footage, this Mac):**

| Video | Length | Clips | End to end | Notes |
|---|---|---|---|---|
| MrBeast "Eat Everything In A Grocery Store" (plN7JMbadRg) | 58 min | 14 | ~3 h incl. re-runs | first run; exposed most bugs fixed in v4.5 |
| MrBeast "100 Kids Vs World's Strongest Man!" (oBXSvS2QKxU) | 27 min | 13 | 64 min | download 1 min, transcribe 4.5 min, LLM stages ~5 min, render ~50 min |

Video 2 framing audit (`tools/framing_audit.py`, 8,445 samples): holding a stale crop 4.9%
(video 1: ~24%), head cut 0.5%, split-screen 3.7% of samples, wrong-person proxy 24% (noisy —
see OPEN_ISSUES 1.1), faces sliced by the crop edge 9.2%. Both test videos' media were deleted
after review; only logs remain.

`discovery.target_clips: 15` is a **maximum** — the quality floor, dedup and "no payoff →
reject" intentionally return fewer.

---

## 2. ENVIRONMENT (this machine)

- macOS arm64, 8 GB RAM (swap ~1.9 GB during runs), CPU only (no CUDA).
- Python 3.12 virtualenv **`.venv/`** (`.venv/bin/python main.py …`); deps from `requirements.txt`
  (whisperx, torch 2.8, opencv 5.0 + contrib, mediapipe, sentence-transformers, gradio,
  psycopg2/pgvector, yt-dlp, librosa, pytest). Node v24 on PATH (yt-dlp JS runtime for HD).
- **Database:** Docker container `viral-clips-db` (`pgvector/pgvector:pg16`, volume
  `viral_clips_pgdata`, `--restart unless-stopped`) on **port 5433** — another Postgres already
  uses 5432 on this Mac. Schema via `db/init_db.sh --docker viral-clips-db`.
- **`.env`:** `GROQ_API_KEY`, `_2`, `_3`, `_FALLBACK` (free tier: 8,000 tokens/min/key),
  `DB_PORT=5433`, `DB_PASSWORD=postgres`. Template: `.env.example`. Never print key values.
- **Models:** `models/Qwen3-Embedding-0.6B/` (1.1 GB), `models/face_detection_yunet_2023mar.onnx`
  (auto-downloaded), Whisper `small` + English align model in the HuggingFace cache.
- Harmless startup noise: torchcodec "could not load libtorchcodec", objc duplicate-class
  warning (OpenCV vs PyAV libavdevice).

---

## 3. FILE STRUCTURE

```text
senna-dev/
├── main.py                 # orchestrator: run_pipeline() (11 stages, resume, run log, keep-awake),
│                           # render_clips(), reset_from_stage(), STAGES, CLI
├── ui.py                   # Gradio (port 7862): Generate + Review & Correct tabs (untested since v4.5)
├── rerun_from_embed.py     # thin wrapper: run_pipeline(video_id=…, from_stage=…) + decision table
├── topup_clips.py          # surgical deterministic top-up + similarity/refine/render for a video_id
├── validate_fullscreen.py  # re-render + render checks + crop-debug previews for a video_id
├── preview_framing.py      # layout previews for a video_id (verification only)
├── demo_faces.py / bench_transcribe.py   # verification scripts (legacy paths, see OPEN_ISSUES 5.3)
├── tools/framing_audit.py  # framing quality audit for a rendered video (see §6.9)
├── config.yaml             # SOURCE OF TRUTH for all tunables (§7)
├── config.py               # env + yaml merge (whitelisted top-level keys!), ffmpeg/ffprobe resolvers
├── logger.py               # setup_logging, bind(), log_stage(), run_log() (per-run log files)
├── conftest.py, test_*.py  # pytest suite, 300 tests (§8)
├── AGENT_CONTEXT.md        # this file      ├── OPEN_ISSUES.md     # open problems, prioritised
├── README.md               # user-facing setup/usage   ├── explanation.md   # beginner explainer
├── implementation_plan.md  # build history (phases)    ├── .env.example / .github/workflows/ci.yml
│
├── input/input_handler.py  # any source → local file; download cache; H.264 ≤1080p; 403 retry
├── campaign/               # campaign_parser.py (LLM → JSON config) + templates/*.json
├── db/
│   ├── init_db.sh          # extension + schema + migrations v4–v6 (idempotent; --docker <name>)
│   ├── connection.py       # ThreadedConnectionPool + pgvector
│   ├── schema.sql, migration_v4/5/6.sql (+ legacy migrate_embeddings_1024.sql)
│   └── repositories/       # video_repo (+find_latest_by_source, update_raw_path), chunk_repo,
│                           # clip_repo, run_repo (+fingerprints), candidate_repo, outline_repo,
│                           # event_repo, editplan_repo
├── pipeline/
│   ├── workspace.py        # per-video paths (§4) — activate()/current()
│   ├── transcriber.py      # WhisperX + align → <work>/transcript.json|txt (+ glossary)
│   ├── chunker.py, embedder.py
│   ├── audio_events.py, audio_classifier.py (YAMNet optional, off)
│   ├── outline.py, discovery.py, analyzer.py, fusion.py   # clip discovery (§5.5)
│   ├── similarity.py, boundaries.py, fingerprints.py
│   ├── llm_client.py       # key rotation, reasoning headroom, JSON mode, rate-limit waits (§5.11)
│   ├── clipper.py          # FFmpeg cut / concat
│   ├── framing.py          # detection, tracking, cuts, virtual camera, layouts, render, debug (§6)
│   ├── active_speaker.py   # lip motion, audio voicing, shot director (§6.5)
│   ├── captions.py, regen_srt.py, glossary.py, editplan.py, polish.py, review.py
│   ├── fast_burn.py, improvements.py   # legacy render fallbacks (export.consolidated: false)
│   └── speakers.py, diarization.py, visual.py   # optional adapters, default off
├── assets/                 # logos/, music/ (empty by default)
├── models/                 # embedding model, YuNet face model (git-ignored)
├── downloads/              # cached source videos  <source-key>.mp4   (generated, git-ignored)
├── work/                   # per-video intermediates                  (generated, git-ignored)
├── output/                 # per-video finished clips + report.json   (generated, git-ignored)
├── logs/                   # viral-clips.log (rotating) + runs/<timestamp>_<source>.log (git-ignored)
└── clips/, transcripts/    # legacy shared dirs — only used by tools with no active workspace
```

---

## 4. WORKSPACES, RESUME, RUN MANAGEMENT

**Per-video workspace** (`pipeline/workspace.py`). Label = `<source-key>_<video_id[:8]>`
(source-key = YouTube id, `gd_<driveid>`, or sanitised file stem):

```
downloads/<source-key>.mp4             cached source (YouTube/Drive/URL); local files used in place
work/<label>/audio.wav                 extracted audio
work/<label>/transcript.json|.txt      transcript (+ outline.json, clips_analysis.json, visual.json, glossary.json)
work/<label>/clips/                    clip_N.srt, clip_N_plan.json, clip_N_vertical_debug.mp4
                                       (+ clip_N.mp4 / _vertical.mp4 only with export.keep_intermediates)
output/<label>/clip_N_final.mp4        finished clips
output/<label>/report.json
```

`run_pipeline` calls `workspace.activate(video_id)` right after the video row exists; modules use
`workspace.current()` (transcript, audio, clips_dir, output_dir, `clip_file(n, suffix)`,
`final(n)`, `report`). With no active workspace they get the legacy shared paths (tests, tools).
`render_clips(video_id)` activates the workspace itself (Review-tab re-renders).
*Why:* before v4.5 every run shared `downloads/audio.wav`, `transcripts/`, `clips/`, `output/` —
a second video was transcribed from the first video's cached audio, and render skipped clips
whose `output/clip_N_final.mp4` existed from the previous video.

**Resume.** Same source again → `video_repo.find_latest_by_source()` → same `video_id`; finished
stages are skipped (`pipeline_runs`), the download is reused (`input_handler._cached_source_ok`:
exists, ≥720p, has audio). Verified: a full resume of a finished video takes ~2 s.
- `--fresh` → new video row, everything redone.
- `--from-stage <stage>` → `main.reset_from_stage()` clears that stage + later checkpoints and the
  rows they'd duplicate or reuse (chunks / audio_events / outlines / candidates / clips →
  cascades related_segments + edit_plans) and gating fingerprints (`audio_events`, `outline`,
  `discovery`). Stage order: `transcribe, chunk, audio_events, embed, outline, analyze,
  similarity, refine, render`.
- `--video-id <uuid>` → continue a specific row. `rerun_from_embed.py <id> [--from-stage]` wraps this.
- If `transcribe` is "done" but the workspace transcript is missing → automatic reset from transcribe.

**Render reuse rule:** existing finals are kept only when resuming a render that crashed
(`render` status running/failed). Any fresh render (after refine, framing change, Review edit)
deletes the stale final first. `export.keep_intermediates: false` deletes each clip's cut +
vertical + legacy-chain files after its final is written (~60 MB/clip).

**Run hygiene.** Each `run_pipeline` writes `logs/runs/<YYYYmmdd-HHMMSS>_<source-key>.log`
(`logger.run_log`). On macOS it holds a `caffeinate -i -w <pid>` assertion for the run
(`runtime.keep_awake`) — a sleeping Mac once paused a run for 15+ min.

**CLI:** `main.py <source> [--campaign "…"] [--template name] [--layout auto|speaker_crop|
stacked_split|center_crop] [--no-refine] [--debug-framing] [--fresh] [--from-stage S]
[--video-id ID] [--log-level DEBUG]`.

---

## 5. PIPELINE STAGES (`main.py:run_pipeline`)

`input → (register/resume + workspace) → campaign → transcribe → chunk → audio_events → embed →
outline → analyze → similarity → refine → render → report`. All from transcribe on are
checkpointed in `pipeline_runs`; `audio_events` and `outline` are fingerprint-gated
(`stage_fingerprints`); completing `refine` invalidates `render`.

### 5.1 Input (`input/input_handler.py`)
`detect_source_type`: any existing file path → local; youtube / youtube_live / google_drive /
direct_url. Local files are used **in place** (no copy). Remote sources download to
`downloads/<key>.mp4` and are reused when valid. YouTube format ladder (`HQ_FORMAT`): **H.264
(avc1) ≥720p and ≤ `YT_MAX_HEIGHT` (default 1080)** first, then any ≤1080p mp4, then any ≥720p.
(4K AV1 would make every decode ~3–4× slower on CPU; output is 1080×1920 anyway.) Attempts:
`youtube-hq` → `youtube-hq-retry` (same command, only if hq *errored* — fixes the audio-stream
HTTP 403 when its signed URL expires during a long video download; yt-dlp reuses the finished
video stream) → android fallback. Anything below `YT_MIN_HEIGHT` (720) fails loudly. Live: 300 s.

### 5.2 Campaign (`campaign/campaign_parser.py`)
LLM → strict JSON (logo/position, music/volume, subtitles, template, fade, layout…); assets only
from `assets/`; no key / 429 with empty description → safe defaults. Layout precedence:
`--layout` > campaign `layout` > `framing.default_layout`.

### 5.3 Transcription (`pipeline/transcriber.py`)
FFmpeg → `work/<label>/audio.wav`; WhisperX `small` (default since v4.5), compute `auto` = int8
on CPU / float16 on CUDA; librosa 16 kHz; alignment → per-word start/end. Measured on this Mac
(87 s of MrBeast audio): base 0.09× realtime but "21,700 galleries" + "spill it"; **small 0.16×
and both right** (~9 min per hour of video; real run: 27 min audio in 3.5 min + 1 min align);
medium 0.33×, not better. Glossary (`transcription.glossary`) rewrites display text only.

### 5.4 Chunk, audio events, embed
- Chunker: silence gaps (≥1.2 s) at sentence ends + 1000-token cap + 25 s overlap + 15 s min → DB.
- Audio events: 200 ms RMS vs rolling median, +8 dB, 0.4–8 s, merged; each spike proposes a
  transcript-aligned candidate (−25 s/+10 s, sentence-snapped, speech coverage ≥0.3), one
  batched LLM enrich. YAMNet labels optional (off). Fingerprint includes the audio path.
- Embeddings: local Qwen3 0.6B, 1024-dim, batched, idempotent; clip/query embeddings use the
  model's `query` prompt.

### 5.5 Clip discovery: outline → discovery → fusion (`analyzer.py`)
- **Outline** (`outline.py`): sentence windows (40, ≤12 windows — widened on long videos, ~120
  sentences/window at 58 min) → one LLM call each (sections, topics, Q&A ids, story arcs, key
  ids) + a combine call; deterministic fallback per failed window. JSON mode.
- **Discovery** (`discovery.py`): per-window candidates citing **sentence IDs only**; validated
  (known ids, 15–90 s spans, contiguous by default). JSON mode.
- **Fusion** (`fusion.py`): outline + audio candidates → IoU ≥0.7 merge → 7-part editorial score
  (opening .22, standalone .24, payoff .16, completeness .16, relevance .08, editability .09,
  audio .05) → diverse selection up to `target_clips` above `min_score 0.45`.
- No usable LLM at all → `_deterministic_section_clips` top-up.

### 5.6 Similarity (`similarity.py`)
pgvector top-k (3, threshold 0.5) per clip, LLM classifies stitch / standalone / noise →
`related_segments`. Stitch caps: gap ≤45 s, total ≤90 s. In practice few links render: video 1
rendered 1 of 14 clips as multi-part.

### 5.7 Boundary refinement (`boundaries.py`)
Words → sentences → deterministic snap; then **`_llm_validate`** (JSON mode, ±20 s context) asks
for start/end sentence IDs, `candidate_complete` and **`selection_complete`** (is the SELECTED
range a finished thought — the old single "complete" flag was about the original candidate, so
the code walked back from endings the model had already fixed). If the selection is still
incomplete → **`_llm_extend`** looks up to `max_payoff_extension_seconds` (30) past the
candidate end for where the payoff lands. Still none → **rejected** (`reject_incomplete: true`,
reason recorded). LLM-confirmed endings may extend up to 30 s; unverified ones keep the 15 s cap.
Over-length → trimmed from the **start** only. Continuations are refined separately and only
attached while the merged total ≤ `similarity.max_total_seconds` (`_merged_total`; before v4.6
three clips reached 96–97 s). Post-refine `fusion.deduplicate_refined` ignores already-rejected
clips (it once dropped 3 valid clips as "duplicates" of a rejected one).

### 5.8 Render (`main.render_clips`)
1. `clipper.cut_clips` → `work/<label>/clips/clip_N.mp4` (single slice or concat).
2. `framing.render_vertical` → 1080×1920 (§6). Exceptions → `render_center_crop` (static, never bars).
3. Captions: `captions.build_captions` (phrase cues ≤8 words / 42 chars / 4 s, ≤2 lines, ASS
   karaoke, placement auto top/bottom from face positions) + SRT via `regen_srt`.
4. Edit plan (`editplan.py`): versioned plan (ranges, timeline, cues, framing scene crops, manual
   ROIs, export) → `edit_plans` + `work/<label>/clips/clip_N_plan.json`.
5. Polish (`polish.py`, one FFmpeg pass): ASS + fades + logo + loudnorm −16 LUFS + music duck +
   limiter → `output/<label>/clip_N_final.mp4`. Legacy multi-pass chain on failure.
6. `report.json` per video (only rendered clips listed when render is skipped on resume).

### 5.9 Review & Correct (`review.py`, UI tab)
Boundary / layout / caption-text / crop-ROI overrides, candidate accept/reject; re-render one
clip via `render_clips(video_id, clip_numbers=[n])`. Manual ROIs always win over detection.

### 5.10 Optional adapters (default off, never fake)
`speakers.py` + `diarization.py` (explicit `speaker_map` + pyannote/HF_TOKEN only), `visual.py`
(keyframes; no vision backend available).

### 5.11 LLM client (`pipeline/llm_client.py`)
Every LLM call goes through `post_chat(payload, timeout, slot, purpose, json_mode=False)`:
- **Key slots:** key1 outline/discovery/enrich, key2 similarity, key3 campaign/analyzer/
  boundaries, key4 fallback; order slot → fallback → rest; placeholders/empties skipped; values
  never logged. 400/401/403/404 = dead key/model (try next key, no retry storm).
- **Reasoning headroom:** for reasoning models (`gpt-oss`, `qwen3`, …) `max_tokens` +=
  `LLM_REASONING_HEADROOM` (4000) — hidden reasoning shares the budget; 7 of 12 outline windows
  once failed with truncated JSON. `finish_reason == "length"` is logged.
- **JSON mode** (`response_format: json_object`) for outline, combine, discovery, boundaries;
  a 400 `json_validate_failed` retries the same key without JSON mode.
- **Rate limits:** when every key returns 429 on a per-minute budget, wait the reset Groq
  reports (retry-after / x-ratelimit-reset-* / "try again in Xs", 2–65 s) and retry, up to
  `LLM_RATE_LIMIT_MAX_WAIT` (300 s) per call; daily (TPD/RPD) limits fail fast.

---

## 6. FRAMING (`pipeline/framing.py`, `pipeline/active_speaker.py`)

**Hard requirement:** `full_screen_vertical: true` — every frame is a 9:16 crop filling
1080×1920; never letterbox / pad / blur-fill (`branded_fit` only renders with the flag off).

### 6.1 Sampling (`sample_clip`)
Sequential decode (no seeking — OpenCV seeks land on wrong frames). Every frame → cut detector;
every `sample_interval` (0.1 s) → face detection + tracking + lip motion + saliency. Sample
time = frame_index / fps (same clock as the renderer). Audio envelope → per-sample `voiced` +
`db`. Tracks seen < `min_track_hits` (3) dropped; `n` = confirmed faces. Content bounds
(embedded black bars) detected and removed from crop math.

### 6.2 Detection & tracking
- `detect_faces_bgr`: **YuNet** (`cv2.FaceDetectorYN`, downscaled to `detect_width` 960, gives
  5 landmarks) > MediaPipe (box bottom bug fixed: `y2 = y1 + height`) > Haar. Faces narrower
  than `min_face_ratio` (2%) dropped.
- `FaceTracker`: greedy matching on distance / face width, gated, constant-velocity guess; ids
  survive `track_max_missing_seconds` (1.5 s); reset at cuts.
- `_CutDetector` (every adjacent frame pair): cut = HSV content change ≥ 27 **and** colour-hist
  correlation < 0.85 **and** onset ≥ 2.5× the previous frame's change; cuts < 0.3 s apart
  merged. Rejects whip pans (change spread over many frames). Was: grey diff 0.1 s apart → 341
  fake "cuts" in 86 s.

### 6.3 Virtual camera (`plan_camera_axis`, `track_path`)
Per scene, offline: fill gaps → clamp to reachable range → median filter (5) → **deadzone**
(locked shot while the subject moves < 15% of the crop) → zero-phase Gaussian
(`path_smooth_seconds` 0.45) at frame rate → pan-speed cap (0.6 source widths/s) → light
re-smooth. No smoothing across cuts. `track_path` additionally: subject never seen in the shot
→ None (caller falls back); face lost > `hold_last_position_seconds` (1 s) → follow saliency;
**deslice** — nudge the target ≤ 25% of the crop so neighbouring faces are fully in or out.
Crops are sub-pixel (`warpAffine`, cubic).

### 6.4 Per-scene decision (`_render_plan`)
For each scene (split at cuts):
1. `present` = people visible ≥ 20% of the scene; `people` = those also ≥ `speaker_min_face_ratio`
   (4.5% width) and not cut off by the frame edge; `prominence` = Σ face width × centrality
   (1 − 0.7·offset; edge-clipped ×0.5). `crowd` = ≥ 3 present and nobody ≥ 2.5× more prominent.
2. ≥ 2 eligible people and layout auto/speaker_crop → **active-speaker director** (§6.5). In a
   crowd its shots must score ≥ `crowd_speaker_min_score` (0.25) or the scene is framed as a group.
3. No reliable speaker: explicit `stacked_split` keeps panels; auto stacking only with exactly
   two prominent people and no crowd.
4. Crowd → **group shot**: `_group_target` picks the crop position framing the most whole faces
   (size^0.7 × (1−offset)^1.5, edge faces ×0.2, sliced faces −1.5×); face-less gaps follow saliency.
5. One subject (prominence) → single face path with **auto-zoom** (`_shot_crop_size`: face =
   33% of output width, ≤ 2.4× upscale, never wider than full height) + **emphasis punch-ins**
   (6% eased zoom on voiced loudness peaks, ≥4 s apart, not near cuts).
6. No faces → **salient** path (§6.6); otherwise work_area / retained / centre crop.

**Split-screen** (`stacked`): panels cropped at the panel's own aspect, each with its own path;
requires both people on screen **simultaneously** ≥ `stack_min_concurrent` (60%) and panels
≥ 0.8 panel-width apart (prevents the same face twice).

### 6.5 Active speaker (`active_speaker.py`)
- `mouth_motion`: change in the mouth ROI (YuNet mouth/nose landmarks) between the sample frame
  and the frame before it, minus 0.8× eye-band change (head motion).
- `audio_envelope` / `voiced_at`: RMS per 0.1 s vs the clip's noise floor (+10 dB).
- `direct_shots(seg, people, cfg, prior)`: smoothed lip scores (`_nan_smooth`, window capped to
  scene length — a short-scene crash once dropped clips to static crops); below
  `speaker_min_activity` (0.08) = not talking; needs ≥ `speaker_min_clear_seconds` (0.3 s) of one
  clearly dominant mouth else None; Viterbi over people (+ two-shot state only when exactly two
  eligible) with `speaker_switch_cost` 3.0, prominence prior, `min_shot_seconds` 1.6, cuts
  snapped to the quietest point ±0.4 s. Returns shots with confidence scores.

### 6.6 Saliency (`salient_cx`)
Spectral-residual saliency on a 320 px thumbnail × motion weight (`0.1 + 0.9·motion/max` vs the
adjacent frame — moving players beat static bright lights), top 18% / bottom 15% bands ignored
(burned-in titles), result = centre of the **densest crop-width window** (not the centre of
mass, which lands between two clusters). Used for face-less shots and face-less gaps.

### 6.7 Output, debug, manual ROIs
`_frame_rects(plan, k, …)` gives the exact source windows per frame — shared by the renderer and
`_DebugWriter` (`--debug-framing`: source with face ids, lip bars, crop boxes + status line |
the vertical output, with audio). Anchor kinds reported: face / retain / group / salient /
stacked_split / center / manual. Manual ROIs (Review tab) override everything.

### 6.8 Layout decision (`decide_layout`)
Clip-level majority over samples (1 face → speaker_crop, 2+ → stacked_split, few faces →
center_crop); now mostly a label + input to `_render_plan`, which decides per scene.

### 6.9 Measuring framing (`tools/framing_audit.py <video_id> [--clips] [--json]`)
Re-cuts the clips from the source, rebuilds the exact plan, checks every 0.1 s sample: wrong
person (1 s window, big face — a noisy proxy), sliced faces, no face in crop, off-centre, head
cut, split-screen (random pairs), crowd/salient/holding shares. Use before/after on the same
video and confirm the flagged moments in the debug videos — by-eye checks beat the proxy.

---

## 7. CONFIGURATION REFERENCE (`config.yaml` — source of truth)

`config.py:get_config()` only passes through **whitelisted top-level keys** (add new sections
there). Env overrides: `GROQ_*`, `DB_*`, `EMBED_*`, `WHISPER_*`, `YT_MIN_HEIGHT`,
`YT_MAX_HEIGHT`, `LLM_REASONING_HEADROOM`, `LLM_RATE_LIMIT_MAX_WAIT`, `LOG_*`, `FFMPEG_PATH`.

```yaml
runtime:       {keep_awake: true}
transcription: {model: small, device: auto, compute_type: auto, language: en, glossary: []}
outline:       {window_sentences: 40, max_windows: 12}
discovery:     {target_clips: 15, max_candidates: 24, sentences_per_call: 120,
                min_span_seconds: 15, max_span_seconds: 90, allow_noncontiguous: false}
fusion:        {merge_iou: 0.7, diversity_iou: 0.5, diversity_sim: 0.92, min_score: 0.45, weights: …}
similarity:    {top_k: 3, threshold: 0.5, max_stitch_gap_seconds: 45, max_total_seconds: 90}
refine:        {context_seconds: 20, max_extension_seconds: 15, max_duration: 90,
                start_padding: 0.15, end_padding: 0.3, llm_validation: true,
                max_payoff_extension_seconds: 30, reject_incomplete: true}
export:        {consolidated: true, codec: libx264, preset: fast, crf: 23, keep_intermediates: false}
captions:      {renderer: ass, max_words: 8, max_chars: 42, max_duration: 4.0, max_lines: 2,
                font: Arial, font_size: 64, placement: auto}
framing:
  # core
  default_layout: auto, full_screen_vertical: true, debug_video: false, interpolation: cubic
  # detection / tracking / cuts
  face_backend: auto, face_confidence: 0.6, detect_width: 960, min_face_ratio: 0.02,
  sample_interval: 0.1, track_max_missing_seconds: 1.5, min_track_hits: 3,
  scene_cut_content: 27.0, scene_cut_corr: 0.85, scene_cut_onset: 2.5, min_scene_seconds: 0.3
  # virtual camera
  deadzone_ratio: 0.15, median_window: 5, path_smooth_seconds: 0.45, max_pan_speed: 0.6,
  hold_last_position_seconds: 1.0, headroom_ratio: 0.35
  # shot styling
  auto_zoom: true, target_face_ratio: 0.33, max_upscale: 2.4, emphasis_zoom: 0.06,
  emphasis_db: 4.0, emphasis_min_gap: 4.0, emphasis_hold: 1.2, cut_snap_seconds: 0.4
  # split-screen
  stack_min_presence: 0.3, stack_min_concurrent: 0.6, panel_face_scale: 3.2, panel_headroom_ratio: 0.4
  # active speaker
  active_speaker: true, speaker_min_presence: 0.2, speaker_max_people: 4,
  speaker_window_seconds: 0.5, speaker_min_activity: 0.08, speaker_voice_db: 10.0,
  speaker_switch_cost: 3.0, min_shot_seconds: 1.6, speaker_allow_group: true,
  speaker_group_threshold: 0.5, speaker_group_bias: 0.2
  # phase D
  speaker_min_face_ratio: 0.045, speaker_min_clear_seconds: 0.3, speaker_prior_weight: 0.1,
  crowd_min_people: 3, crowd_dominance: 2.5, crowd_speaker_min_score: 0.25,
  deslice_faces: true, deslice_max_shift: 0.25, saliency_fallback: true
```
All Phase D thresholds were tuned on MrBeast footage only (OPEN_ISSUES 1.9).

---

## 8. TESTS

**300 pytest tests, all green** (`.venv/bin/python -m pytest -q`, ~25 s; needs the Docker DB
for repo round-trip tests). Network/LLM always mocked; tests never create workspaces or run logs
(`test_main` mocks `workspace.activate`, `run_log`, `_keep_awake`). Largest suites:
test_framing 69 (detection geometry, tracker, cut detector, camera path, deslice, group target,
saliency, split-screen safety, render plan, render smoke with debug video), test_fusion 23,
test_active_speaker 20, test_editplan_polish 19, test_boundaries 15 (incl. payoff extension and
the stitched total cap), test_llm_client 15 (headroom, JSON mode retry, rate-limit waits),
test_main 13 (workspaces, resume, --fresh/--from-stage), test_input_handler 13 (cache, local in
place, 403 retry). CI (`.github/workflows/ci.yml`): lint job + pytest job with a pgvector
service — **never run yet** (repo not pushed).

---

## 9. HISTORY (condensed; details in `implementation_plan.md`)

- **≤ v4.1 (before 09 Oct 2026):** MiniMax → NVIDIA → Groq + local embeddings; 11-stage
  pipeline; outline/discovery/fusion; audio events; boundaries; ASS captions; edit plans; one-pass
  polish; Review tab; 4-key rotation; HD-enforced downloads; 220 tests.
- **v4.2 Framing Phase A:** YuNet, MediaPipe box fix, sequential sampling, persistent tracker,
  offline virtual camera, sub-pixel crops, correct split-screen aspect.
- **v4.3 Phase B:** lip-motion + audio active-speaker director with hard cuts and two-shots.
- **v4.4 Phase C:** face-sized close-ups, emphasis punch-ins, cuts on pauses, debug video.
- **v4.5 (first real run):** per-video workspaces, resume/`--fresh`/`--from-stage`, download
  cache + 403 retry, run logs, keep-awake, intermediates cleanup; LLM headroom / JSON mode /
  rate-limit waits; payoff extension + reject incomplete; Whisper small/int8; new cut detector;
  Phase D (eligible speakers, crowd mode, prominence, deslice, saliency); CI test job; init script.
- **v4.6 (second real run):** H.264 ≤1080p downloads, stitched total cap, never-seen-subject
  crash fix, split-screen concurrency/overlap guards, motion-weighted densest-window saliency,
  crowd gaps follow saliency, framing audit tool.

---

## 10. INVARIANTS & GOTCHAS

- LLMs choose **sentence IDs, never timestamps**; timestamps come from aligned words.
- Every frame fills 9:16; no bars. Manual ROIs beat detection. Speech is never faded.
- New config sections must be added to `config.py`'s whitelist or they silently disappear.
- Paths: never hard-code `transcripts/`, `clips/`, `output/`, `downloads/audio.wav` — use
  `workspace.current()`. Legacy `main()` blocks in `regen_srt.py` / `fast_burn.py` still do.
- `transcript_signature` is mtime+size: moving a transcript keeps caches; copying invalidates.
  The audio-events fingerprint includes the audio **path** — moving audio needs a re-store.
- Free-tier Groq: expect "all 4 keys rate-limited; waiting …" lines — normal, not an error.
- yt-dlp on Mac needs Node on PATH for HD formats. macOS zsh: a non-matching glob aborts the
  whole command (`rm a/* a/.x*` deleted nothing once) — use `find … -delete`.
- Don't trust the wrong-person proxy alone — verify in `--debug-framing` videos.
- `.env` `DB_PORT` must be 5433 for the Docker DB on this Mac.
