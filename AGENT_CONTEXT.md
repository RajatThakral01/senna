# 🤖 Agent Implementation Context: Viral Clips Automator (Detailed Architecture)

**Version:** 4.0
**Date:** 06 Oct 2026
**Status:** 11-stage pipeline implemented & verified end-to-end; 183/183 tests green
**Purpose:** This document is the single source of truth for the AI agent maintaining this project. It provides exhaustive technical details of how the Viral Clips Automator works under the hood so that the agent understands the entire system.

> v4.0 changelog vs v3.2: full rewrite to match reality. Provider is now **Groq LLM (`openai/gpt-oss-120b`) + local SentenceTransformer embeddings (`models/Qwen3-Embedding-0.6B`, 1024-dim)** — all NVIDIA NIM / Kimi / MiniMax references removed from live code. Pipeline is **11 stages** (was 8): added `audio_events`, `outline`, `refine` stages; `analyze` is now two-pass contextual discovery (outline → candidates → fusion). New modules: `boundaries.py`, `framing.py`, `outline.py`, `discovery.py`, `fingerprints.py`, `audio_events.py`, `audio_classifier.py`, `fusion.py`, `captions.py`, `glossary.py`, `editplan.py`, `polish.py`, `review.py`, `speakers.py`, `visual.py`, `diarization.py`. DB migrations v4–v6 applied (refine columns, outlines/candidates/audio_events/fingerprints/provenance, edit_plans). Render path is now framing + ASS captions + consolidated single-pass polish with versioned edit plans, plus a Review & Correct UI tab. `resume.py`, `pipeline/downloader.py`, `subtitler.py`, `karaoke_burn.py` no longer exist — use `rerun_from_embed.py`. Test suite repaired: 183/183 pass.

---

## 1. PROJECT OVERVIEW

The Viral Clips Automator is an end-to-end pipeline that converts long-form videos into short-form viral clips optimized for TikTok, Instagram Reels, and YouTube Shorts.

It uses a Groq-served LLM (`openai/gpt-oss-120b`, OpenAI-compatible `/chat/completions`) for campaign parsing, clip discovery, boundary validation, and similarity classification, plus **local** SentenceTransformer embeddings (`models/Qwen3-Embedding-0.6B`, native 1024-dim, no API key) stored in PostgreSQL + pgvector (HNSW). WhisperX provides word-level transcription; FFmpeg + OpenCV/MediaPipe handle cutting, face-aware 9:16 framing, ASS karaoke captions, and single-pass polish.

**Latest verified runs (324s DIY sample):** final config produces 3 clips (35–50s, fusion scores 0.53–0.81, 1 near-dup rejected with reason, 2 audio cues fused, 18–31 phrase-caption cues each, edit plans v1, single-pass polish, render 406s). Earlier baseline on the same source: 4 clips/25–65s with word-captions and blurred-background default. Target is 8 (`discovery.target_clips`) treated as a **maximum** — quality floor + dedup intentionally return fewer when quality is insufficient.

---

## 2. FILE STRUCTURE

```text
viral_clips_automator/
├── main.py                        # Orchestrator — run_pipeline() 11 stages + render_clips()
├── ui.py                          # Gradio UI (port 7862): Generate tab + Review & Correct tab
├── rerun_from_embed.py            # Re-run helper: cleanup + embed→render for an existing video_id
├── preview_framing.py             # Verification only: layout previews + boundary demo → output/previews/
├── demo_faces.py                  # Verification only: synthetic face demos (public-domain portrait)
├── bench_transcribe.py            # Verification only: WhisperX model benchmark → output/bench_transcription.json
├── config.yaml                    # SOURCE OF TRUTH for all tunables (see §4)
├── config.py                      # Groq + local-embedding settings, get_config(), ffmpeg/ffprobe resolvers
├── requirements.txt               # yt-dlp, whisperx, gradio, psycopg2, pgvector, sentence-transformers, opencv, mediapipe, pytest
├── conftest.py                    # pytest path shim + cleanup_video() helper
├── explanation.md                 # Beginner-friendly end-to-end explainer (non-agent docs)
├── AGENT_CONTEXT.md               # This file — agent source of truth
├── implementation_plan.md         # Phase 0–7 completion log (see §8)
│
├── input/
│   ├── __init__.py
│   └── input_handler.py           # LIVE: normalizes any source into input/raw_video.mp4
│
├── campaign/
│   ├── __init__.py
│   ├── campaign_parser.py         # LIVE: Groq parser, natural language → JSON config (+ mock fallback)
│   └── templates/                 # podcast_clip.json, tiktok_reaction.json, motivational_reel.json
│
├── db/
│   ├── connection.py              # ThreadedConnectionPool (1-10) + pgvector registration
│   ├── schema.sql                 # Base tables (videos, chunks, clips, related_segments, pipeline_runs)
│   ├── migrate_embeddings_1024.sql
│   ├── migration_v4.sql           # Refine/layout columns on clips (refine_status/reason, source_ranges, timeline, layout)
│   ├── migration_v5.sql           # outlines, candidates, audio_events, stage_fingerprints, clips.provenance
│   ├── migration_v6.sql           # edit_plans (versioned edit plans)
│   └── repositories/              # video_repo, chunk_repo (+pgvector search), clip_repo,
│                                  # run_repo (+fingerprints), candidate_repo, outline_repo,
│                                  # event_repo, editplan_repo
│
├── pipeline/
│   ├── __init__.py
│   ├── transcriber.py             # LIVE: WhisperX → transcripts/transcript.json (+ glossary corrections)
│   ├── chunker.py                 # LIVE: silence-gap + sentence + token-budget + overlap chunker → DB
│   ├── embedder.py                # LIVE: local SentenceTransformer → DB (batch, idempotent)
│   ├── outline.py                 # LIVE (pass 1): whole-video structure (sections, Q&A, story arcs)
│   ├── discovery.py               # LIVE (pass 2): sentence-ID-cited candidates, validated spans
│   ├── analyzer.py                # LIVE: two-pass (outline→discovery→fusion) + legacy per-chunk fallback
│   ├── audio_events.py            # LIVE: RMS spike detector → transcript-aligned event cues
│   ├── audio_classifier.py        # LIVE (optional, default off): YAMNet laughter/applause/music labels
│   ├── fusion.py                  # LIVE: merge → editorial scoring → diverse select → post-refine dedup
│   ├── similarity.py              # LIVE: pgvector search + 3-way LLM stitch/standalone/noise
│   ├── boundaries.py              # LIVE: sentence-complete refinement (deterministic + LLM-validated)
│   ├── fingerprints.py            # Stage fingerprints for checkpoint invalidation
│   ├── clipper.py                 # LIVE: FFmpeg cut / concat stitch (respects stitch caps)
│   ├── framing.py                 # LIVE: face-aware 9:16 (auto/speaker_crop/stacked_split/branded_fit)
│   ├── diarization.py             # OPTIONAL: pyannote wrapper (needs HF_TOKEN; never maps audio→face alone)
│   ├── speakers.py                # OPTIONAL (off): explicit-map-only speaker-switch adapter
│   ├── visual.py                  # OPTIONAL (off): keyframe sampler + vision-backend adapter (never fakes)
│   ├── captions.py                # LIVE: phrase-level ASS karaoke (face-aware placement, SRT preserved)
│   ├── glossary.py                # LIVE: traceable display-text corrections (transcripts/glossary.json)
│   ├── regen_srt.py               # LIVE: word-timestamps → per-clip .srt with concat timeline mapping
│   ├── fast_burn.py               # FALLBACK: drawtext-chain word burn (multi-pass for long lists)
│   ├── editplan.py                # LIVE: versioned structured edit plans (preview == export)
│   ├── polish.py                  # LIVE: consolidated single-pass FFmpeg polish
│   ├── improvements.py            # FALLBACK: add_fades()/apply_logo()/mix_music() (export.consolidated=false)
│   └── review.py                  # LIVE: Review & Correct ops (boundary/layout/caption overrides)
│
├── assets/
│   ├── logos/logo.png
│   ├── music/                     # empty by default
│   └── style.css                  # unused
│
├── test_*.py                      # pytest suite, 183/183 green (see §8)
├── transcripts/                   # transcript.json/txt, clips_analysis.json, outline.json, visual.json, glossary.json (generated)
├── clips/                         # intermediates: clip_N.mp4 → _vertical → .srt → _plan.json (generated)
├── downloads/                     # audio.wav for WhisperX (generated)
├── input/raw_video.mp4            # normalized input, overwritten each run (generated)
└── output/                        # clip_N_final.mp4 + report.json (generated); previews/, phase snapshots, bench JSON
```

Live render path: `clipper.py + framing.py + captions.py (+ regen_srt) + editplan.py + polish.py`, with `improvements.py + fast_burn.py` as fallback when `export.consolidated` is false or polish fails. Never route new code through deleted legacy files.

---

## 3. TECHNICAL DEEP DIVE: HOW IT WORKS

Stages in `main.py:run_pipeline()` (progress 0.05 → render): **input → campaign → transcribe → chunk → audio_events → embed → outline → analyze → similarity → refine → render**. All except input/campaign are checkpointed in `pipeline_runs`; `audio_events`/`outline` are fingerprint-gated (`stage_fingerprints`); completing `refine` invalidates stale `render` checkpoints; already-rendered finals are skipped (resumable).

### 3.1 Input Handling & DB Registration
`input/input_handler.py` (`detect_source_type()` + `handle_input()`): YouTube (`yt-dlp`, Android-client 360p fallback on 403), YouTube-live (300s cap), Google Drive (`gdown`), local files (`shutil.copy`), direct URLs (streamed `requests`) → always `input/raw_video.mp4`. `video_repo.insert_video()` registers the row; `should_skip_stage()` reads `pipeline_runs`.

### 3.2 Campaign Parsing
Groq LLM (`temperature=0.1`, `max_tokens=1000`) turns natural language into strict JSON (logo/logo_position/music/music_volume/subtitles/subtitle_style/aspect_ratio/template/fade/min_clip_duration + **`layout`**: `auto | speaker_crop | stacked_split | branded_fit`). Generic asset requests map to first available file in `assets/` (never hallucinated); only explicitly requested features are set. `merge_config()` = `{**template, **non_null_campaign}`. No key (or Groq 429 with empty description) → keyword mock / safe defaults so UI/tests/resume keep working. Layout precedence: `--layout` CLI > campaign `layout` > `framing.default_layout`, always via `validate_layout()`.

### 3.3 Transcription
`pipeline/transcriber.py`: FFmpeg extracts `downloads/audio.wav`; `whisperx.load_model()` with `config.yaml [transcription]` model/device/compute/language (`device: auto` = CUDA when available else CPU — evidence: 60s sample CUDA 1.5s vs CPU 11.0s, identical words) + `librosa` 16kHz load; alignment gives per-word `start/end`; GPU memory freed afterwards for embeddings. Glossary entries (`transcription.glossary`) rewrite only `display` text; corrections logged to `transcripts/glossary.json`. Outputs `transcript.json` + `transcript.txt`. `bench_transcribe.py` compares models without changing defaults (uncached models skipped unless `--allow-download`).

### 3.4 Chunking
`pipeline/chunker.py:build_chunks()`: silence gaps (≥1.2s) confirmed at sentence boundaries (`.?!`, 20-word lookahead, gap fallback) + token-budget split (>1000 tokens at mid-point sentence) + 25s sliding overlap + 15s minimum filter → DB with `token_count`, `is_overlap_tail`, `silence_gap_before`.

### 3.5 Audio Events (checkpointed, fingerprint-gated)
`pipeline/audio_events.py` runs on pre-normalization source audio with **source timestamps preserved**: 200ms RMS windows vs rolling-median baseline (±10s) + silence floor, 8.0dB threshold, 0.4s min (click suppression), 8.0s cap (sustained regions truncated, confidence lowered), 1.0s merge. Each spike only *prompts* transcript inspection: window [peak−25s, peak+10s] snapped to sentence bounds, kept iff speech coverage ≥0.3 and span within discovery bounds. One batched LLM `enrich_event_candidates()` adds hook/main_idea/payoff (no invented transcript). `audio_classifier.py` optionally labels laughter/applause/music/speech via YAMNet (`backend: none` default — reports disabled, never fakes). Events → `audio_events` table; cues → `candidates(source=audio_event, status=proposed)`.

### 3.6 Embedding (local)
`pipeline/embedder.py`: lazily-loaded SentenceTransformer (`models/Qwen3-Embedding-0.6B`, device auto, dtype auto = bf16 CUDA / fp32 CPU, batch 8, L2-normalized, `EMBED_DIM` 1024 head-slice). Batched, length-sorted encode; idempotent (skips embedded); no rate limits. Stored vectors use no prompt prefix; clip/query embeddings use `prompt_name="query"` when the model defines it. `EMBED_DIM` must match `schema.sql vector(1024)`.

### 3.7 Outline + Discovery + Fusion (two-pass analyze)
- **Outline** (`outline.py`, pass 1): sentences windowed (40/window, ≤12, widened for even coverage on long videos) → one LLM call per window (topics, Q&A with IDs, story setup/development/payoff, key IDs) + video-level combine pass. No key / 429 → deterministic fallback per window, `method` marked (`llm|partial|deterministic`). Persisted to `outlines` + `transcripts/outline.json`; fingerprint-gated (`outline_fingerprint` over transcript sig + model + prompt versions + window size).
- **Discovery** (`discovery.py`, pass 2): per-section transcript windows → candidates citing **sentence IDs only** (`start_id/end_id`, optional `join_ids` only when `allow_noncontiguous: true`, default false). `validate_candidate()` drops unknown IDs, orders spans, enforces 15–90s spans (this caught a live 283s-candidate bug), ≤60s join gaps. Timestamps always derived from word data. Persisted to `candidates(status=proposed)` + `discovery` fingerprint.
- **Fusion** (`fusion.py`, Phase 3): pool = outline candidates + proposed audio-event cues → IoU≥0.7 overlap merge (lower uncertainty wins) → 7-component editorial heuristic scores (`opening .22 / standalone .24 / payoff .16 / completeness .16 / relevance .08 / editability .09 / audio_support .05`; audio is a 5% bonus only, never rescues incoherence) → diverse select (temporal IoU<0.5 unless text sim<0.92; same-topic-distinct moments preserved) up to `target_clips: 8` (a **maximum**) with `min_score: 0.45` floor and recorded skip reasons. Winners → `clips` (hook/reason embedded with query prompt, `provenance` JSON). Legacy per-chunk extraction (`_analyze_legacy`, 0–2 clips/chunk, 5s-start dedupe) runs only when discovery yields nothing and `legacy_fallback: true`.

### 3.8 Similarity Search & 3-Way Classification
`find_related_segments(video_id, clips, top_k=3, threshold=0.50)`: pgvector `ORDER BY embedding <=> query LIMIT k` (threshold applied in Python), own source chunks excluded, LLM `temperature=0.1` classifies `stitch | standalone | noise` (4 attempts, 429-aware backoff) → `related_segments(confirmed_by_llm = decision=="stitch")`, `ON CONFLICT DO NOTHING`. **Stitch caps** (`max_stitch_gap_seconds: 45`, `max_total_seconds: 90`) enforced in both `clipper.get_time_ranges_for_clip()` (via `limit_stitch_ranges`) and `refine_all_clips()` — added after live evidence of a 120s Frankenstein stitch; distant passages are related context, never continuations.

### 3.9 Boundary Refinement (checkpointed; `--no-refine` to skip)
`pipeline/boundaries.py` runs after similarity, before render; only *adjacent* transcript may move a boundary: flat word list → sentence/utterance IDs (punctuation + ≥1.0s pause fallback, missing timestamps skipped) → deterministic snap (start→sentence start −0.15s, end→sentence end +0.3s, bounded by neighbours/video) → LLM **validates/selects sentence IDs** from ±20s context (timestamps never from LLM; punctuation alone insufficient for completeness). Over-extension capped at +15s to a complete ending; over-duration shortened from the **start** only; no complete ending → earlier complete ending or `rejected` with reason. Each confirmed continuation refined independently; ranges merged + output `timeline` mapping stored. Persisted via `clip_repo.update_refinement()` (v4 columns, graceful fallback). Render skips `refine_status=rejected`; `refine` completion invalidates `render`. Post-refine `deduplicate_refined()` (IoU≥0.5 or ≥0.6 smaller-clip coverage) marks dupes rejected.

### 3.10 FFmpeg Rendering Pipeline (`render_clips()`)
1. **Cut/stitch (`clipper.py`)**: `get_time_ranges_for_clip()` prefers refined `source_ranges` (+ explicit `timeline` order), else primary + sorted confirmed continuations with stitch caps; single range → slice re-encode, multi → `concat` filter (`libx264/fast/crf23/aac`).
2. **Vertical (`framing.py`)**: face-aware 9:16 (1080×1920), **no blurred background** — `branded_fit` uses solid `#0B0F1A`. Auto: 1 face → `speaker_crop`, 2 faces → `stacked_split` (top/bottom + divider), uncertain → `branded_fit`; per-scene awareness, MediaPipe BlazeFace (auto-download ~230KB) → Haar fallback → branded fit; stable track association, deadzone/smoothing/jitter/snap, scene-cut reset, no cross-cut interpolation, last-position hold then fallback, headroom/zoom/bounds limits, stability window. Manual layout override supported per clip and globally. (Smoothing/track/panel/scene-cut concepts adapted from MIT-licensed `NaufalRizqullah/opensource-clipping` — see attribution header in `framing.py`; fresh implementation.) Audio muxed from source (`-c:a aac`, `-shortest`).
3. **Captions (`captions.py` + `regen_srt.py`)**: words in ranges (+0.5s end tol) → concat `timeline` mapping (gap-free output times) → phrase cues (≤8 words/42 chars/4s, sentence-aware, ≤2 balanced lines) → ASS karaoke (`\kf` word fill, yellow highlight, configurable font/size/colors/margins) via one-pass `ass` filter when available else `fast_burn` drawtext fallback; `generate_srt_for_ranges()` SRT kept for export/debug. Placement `auto` = top when median face sits low (reuses framing analysis, zero extra cost), else bottom. Glossary `display` text preferred.
4. **Edit plan (`editplan.py`)**: `build_edit_plan()` assembles ranges/timeline/cues/placement/layout/audio/branding/export + model/config versions + fingerprint (crop paths recomputed deterministically, noted in `limits`); persisted versioned to `edit_plans` + `clips/clip_N_plan.json`.
5. **Polish (`polish.py`, one FFmpeg pass)**: ASS + video fades + logo overlay + `setsar=1` (+optional fps); audio = loudnorm (−16 LUFS) + optional denoise + music sidechain-duck + mix + limiter (+optional audio fade, **default off** — speech never faded; `fades: {video: true, audio: false}`); `-an` on silent sources, `-stream_loop -1` music. Legacy chain (`_render_legacy_chain`: drawtext → logo → music → fades via `improvements.py`) used when `export.consolidated: false` or polish fails.
6. Per-clip `update_output_path()` (+refine/layout persist), then `output/report.json` regenerated: clip_number/hook/refined start/end/duration/output_duration/source_ranges/timeline/layout(+reason)/refine_status(+reason)/reason/title/output_file.

### 3.11 Review & Correct (`pipeline/review.py` + UI tab)
User edits win: `validate_boundary()` (numbers, order, ≥1s, within video) → `apply_boundary_edit()` (single-range clips; clears final, `refine_status='manual'`) / `apply_layout_override()` / `apply_caption_text()` (cue text only, word counts must match — karaoke timings preserved; new plan version) / candidate accept-reject. Re-render only the clip: `render_clips(video_id, clip_numbers=[n])`.

### 3.12 Optional adapters (default OFF, never fake)
- **Speakers** (`speakers.py` + `diarization.py`): stable turns (≥1.5s, 0.5s merge, 2.0s hold), voice→face mapping **only** with explicit `speaker_map` (fixed seats); `auto_match` reports `unavailable_unevaluated`; overlap holds current speaker. `run_diarization()` needs `HF_TOKEN` + pyannote, else raises with instructions.
- **Visual** (`visual.py`): candidate-focused keyframes (scene cuts + midpoints, ≤12, `clips/`), `backend: none` default; `groq_vision` reports unavailable (no vision model on key); results cached in `transcripts/visual.json`.

---

## 4. CONFIGURATION REFERENCE

### config.yaml (key blocks — source of truth)
```yaml
ai: {provider: groq, llm_model: openai/gpt-oss-120b,
     llm_endpoint: https://api.groq.com/openai/v1/chat/completions}
embeddings: {provider: local, model: models/Qwen3-Embedding-0.6B,
  device: auto, dtype: auto, batch_size: 8, normalize: true,
  truncate_dim: 1024, max_seq_length: 1024}   # truncate_dim must match vector(1024)
similarity: {top_k: 3, threshold: 0.5, llm_confirmation: true,
  max_stitch_gap_seconds: 45, max_total_seconds: 90}
chunking: {silence_threshold_seconds: 1.2, max_tokens_per_chunk: 1000,
  overlap_seconds: 25, min_chunk_duration_seconds: 15}
outline: {window_sentences: 40, max_windows: 12}
discovery: {target_clips: 8, max_candidates: 24, sentences_per_call: 120,
  min_span_seconds: 15, max_span_seconds: 90,
  allow_noncontiguous: false, legacy_fallback: true}
audio_events: {enabled: true, window_ms: 200, baseline_seconds: 10, threshold_db: 8.0,
  min_duration_seconds: 0.4, max_event_seconds: 8.0, merge_gap_seconds: 1.0,
  silence_floor: 0.0001, pre_seconds: 25, post_seconds: 10,
  min_speech_coverage: 0.3, classifier: {backend: none}}
fusion: {merge_iou: 0.7, diversity_iou: 0.5, diversity_sim: 0.92, min_score: 0.45,
  weights: {opening: 0.22, standalone: 0.24, payoff: 0.16, completeness: 0.16,
            relevance: 0.08, editability: 0.09, audio_support: 0.05}}
transcription: {model: base, device: auto, compute_type: float32, language: en, glossary: []}
captions: {enabled: true, renderer: ass, max_words: 8, max_chars: 42, max_duration: 4.0,
  max_lines: 2, font: Arial, font_size: 64, placement: auto}
refine: {enabled: true, context_seconds: 20, max_extension_seconds: 15,
  min_duration: 20, max_duration: 90, start_padding: 0.15, end_padding: 0.3,
  llm_validation: true}
framing: {default_layout: auto, sample_interval: 0.25, deadzone_ratio: 0.15,
  smooth_factor: 0.30, jitter_px: 5, snap_ratio: 0.25, branded_background: "#0B0F1A",
  allow_active_speaker_switch: false, speaker_map: {}}
speakers: {enabled: false, auto_match: false, min_turn_seconds: 1.5, hold_seconds: 2.0}
visual: {enabled: false, backend: none, max_frames: 12}
audio: {normalize_loudness: true, peak_limit_db: -1.5, duck_music: true, denoise: false}
export: {consolidated: true, codec: libx264, preset: fast, crf: 23, audio_bitrate: 128k, fps: null}
fades: {video: true, audio: false, duration: 0.5}
```

### config.py (named constants)
```python
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", ...)      # LLM only; "" when unset (mock/fallback paths)
GROQ_BASE_URL  = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_LLM_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
LLM_API_KEY, LLM_MODEL, LLM_API_URL  # provider-agnostic aliases used by all LLM modules
EMBED_MODEL / EMBED_DEVICE / EMBED_BATCH_SIZE / EMBED_DTYPE / EMBED_DIM / EMBED_MAX_SEQ
WHISPER_MODEL / WHISPER_DEVICE / ...  # legacy fallbacks; config.yaml [transcription] + WHISPER_* env win
get_config()   # cached YAML + env merge; ffmpeg()/ffprobe_path() resolve FFMPEG_PATH > yaml > PATH
```
**Every LLM module imports** `from config import LLM_API_KEY, LLM_MODEL, LLM_API_URL`. **Do not hardcode keys/URLs/models elsewhere.** No secrets are hardcoded — empty key triggers mock/fallback paths.

---

## 5. DATABASE SCHEMA SUMMARY

Apply once in order: `schema.sql` → `migrate_embeddings_1024.sql` → `migration_v4.sql` → `migration_v5.sql` → `migration_v6.sql` (requires `CREATE EXTENSION vector;`). Repo functions degrade gracefully on unmigrated DBs (legacy column reads, times-only refinement fallback).

| Table | Key Columns |
|---|---|
| `videos` | `id`, `source_url`, `raw_path`, `duration_seconds`, `status`, `campaign_id` |
| `chunks` | `id`, `video_id`, `chunk_index`, `start_time`, `end_time`, `text`, `embedding vector(1024)` (+ HNSW) |
| `clips` | `id`, `video_id`, `clip_number`, `start/end_time`, `duration_seconds`, `hook`, `reason`, `suggested_title`, `suggested_hashtags`, `embedding vector(1024)` (+ HNSW), `output_path`, `refine_status/reason` (v4), `source_ranges/timeline` jsonb (v4), `layout/layout_reason` (v4), `provenance` jsonb (v5) |
| `related_segments` | `id`, `clip_id`, `related_chunk_id`, `similarity_score`, `confirmed_by_llm`, `decision` (stitch\|standalone\|noise), `stitched_into_clip` |
| `candidates` (v5) | `id`, `video_id`, `source` (outline\|audio_event\|similarity\|manual\|legacy), `sentence_ids`, `source_ranges`, `hook/main_idea/payoff/required_context/rationale`, `uncertainty`, `scores`, `status` (proposed\|shortlisted\|selected\|rejected), `clip_id` |
| `outlines` (v5) | `id`, `video_id`, `level` (section\|video), `idx`, `start/end_time`, `title`, `content` jsonb |
| `audio_events` (v5) | `id`, `video_id`, `start/end/peak_time`, `energy_increase`, `confidence`, `method/version`, `label/label_confidence`, `config_fingerprint` |
| `edit_plans` (v6) | `id`, `video_id`, `clip_id`, `version`, `plan` jsonb |
| `pipeline_runs` | `id`, `video_id`, `stage`, `status` (pending\|running\|done\|failed), `error_message`, `started_at`, `completed_at` |
| `stage_fingerprints` (v5) | `video_id`, `stage`, `fingerprint` |

Cosine similarity = `1 - (embedding <=> query)` (`chunk_repo.find_similar_chunks`); HNSW indexes on both embedding columns.

---

## 6. SYSTEM INVARIANTS

- **Provider**: Groq LLM for all language tasks; embeddings strictly local (no embedding API, no embedding rate limits). Groq 429s possible on long videos — analyzer/similarity/boundaries/discovery retry or fall back (deterministic outline, dropped candidates, `noise` mapping, deterministic snap); campaign parser falls back to defaults on empty description.
- **Embeddings**: 1024-dim effective (`EMBED_DIM` head-slice) must equal DB `vector(1024)`. Batch local encode; `query` prompt only when the model defines it.
- **Timestamps**: always derived from aligned word data. LLMs select sentence IDs, never timestamps. Stitched ranges carry an explicit output `timeline`.
- **Speech**: never faded by default (`fades.audio: false`); fades live in `end_padding` silence. Polish normalizes loudness (−16 LUFS) + limits peaks; never invents labels/captions/faces (adapters report disabled/unavailable).
- **FFmpeg**: `libx264/fast/crf23`, `aac 128k`, resolved binary; avoid negative timestamps; reset timestamps on cuts.
- **Checkpoints**: `run_repo.{start,complete,fail}_stage` + `get_stage_status` + fingerprints; `rerun_from_embed.py` is the supported re-run path (dup-safe cleanup of clips/related/candidates/outlines/events + stage invalidation).

---

## 7. KNOWN GOTCHAS

- **Groq 429s on long videos**: free-tier bursts; retries + polite 2s delays between outline/discovery calls; very long videos may yield fewer clips — by design (floor + dedup), not failure.
- **Duplicate clip rows**: `rerun_from_embed.py:cleanup()` deletes clips/related/candidates/outlines/events before re-running analyze. Never re-run analyze without that cleanup.
- **`torchcodec` / `libavutil` warnings** on startup: harmless.
- **Sample with ~0 confident faces** (desk close-ups, dark scenes): `auto → branded_fit` is correct behavior, not a miss. Eyeball one preview — no visual inspection was possible in this environment (machine checks only: dims/duration/audio/timing).
- **Unexercised live**: stacked layout on real 2-person footage, speaker switching + diarization, YAMNet on real footage, large-v3/turbo benchmark, selective re-render, audio-enrich under 429.
- **Standalone-script paths**: `regen_srt.py` / `fast_burn.py` `main()` blocks carry old hardcoded demo paths — only affect direct script runs, not `main.py`.
- **Stale `ensure_v5()` mention**: `migration_v5.sql:4` says app code applies the v5 DDLs at startup via `ensure_v5()` — that function does not exist in code. Apply all migrations manually with `psql`; unmigrated DBs only get the graceful read/write fallbacks, not the tables.
- **Stale references to delete on sight**: any `NVIDIA_*` / `MINIMAX_*` / `kimi` / `nv-embedqa` names, `resume.py`, `pipeline/downloader.py`, `subtitler.py`, `karaoke_burn.py`, 8-stage diagrams, `threshold: 0.65`, blur-background default.

---

## 8. TEST SUITE STATUS (accurate as of v4.0)

**183/183 pytest pass** (was 33 with 7 failures + 2 collection errors). Phase 0 repaired 5 stale files (wrong import paths, teardown deleting real assets, obsolete APIs). `unittest.mock` throughout — no real API/DB/FFmpeg in unit tests.

- `test_boundaries` + `test_framing`: 24/24 (mid-sentence end, unfinished thought, missing punctuation/timestamps, end-of-video bound, duration conflict, stitched SRT sync, 1/2-face, camera cut, detection loss, crop bounds, stability, diarization-no-map).
- Fusion/captions/editplan+polish/speakers+visual/audio-events/outline+discovery/review/glossary suites: all green.
- E2E (needs DB + media, not in pytest): three full runs on the 324s sample via `rerun_from_embed.py`; `output/phase0|phase1|phase4_monster` snapshots, `output/previews/`, `output/bench_transcription.json` kept as evidence.
- `implementation_plan.md` Phase 0–7 log is the current spec history — the old MiniMax-era plan it replaced is superseded.
