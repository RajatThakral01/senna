# 🤖 Agent Implementation Context: Viral Clips Automator (Detailed Architecture)

**Version:** 3.2  
**Date:** 03 Oct 2026  
**Status:** Core pipeline implemented & working end-to-end; test suite partially stale (see §8)  
**Purpose:** This document is the single source of truth for the AI agent maintaining this project. It provides exhaustive technical details of how the Viral Clips Automator V3 works under the hood so that the agent understands the entire system.

> v3.2 changelog vs v3.1: corrected MiniMax-removal claim (remnants remain), documented deprecated/legacy files (`pipeline/downloader.py`, `subtitler.py`, `karaoke_burn.py`), documented broken `test_main.py`, clarified embedding-model source of truth (1024-dim `nv-embedqa-e5-v5`; stale 4096-dim comments remain in code), clarified `similarity.py` default-arg vs `config.yaml` effective threshold, added `explanation.md` / tests / `conftest.py` to file tree, flagged hardcoded secrets in `config.py` and `resume.py`.

---

## 1. PROJECT OVERVIEW

The Viral Clips Automator is an end-to-end pipeline that converts long-form videos into short-form viral clips optimized for TikTok, Instagram Reels, and YouTube Shorts. Version 3.1 built on the V3.0 PostgreSQL-backed architecture with targeted improvements to the similarity classification system and a full codebase-wide variable rename for clarity. v3.2 is a docs/correctness pass with no pipeline behavior change.

It uses NVIDIA NIM endpoints (`moonshotai/kimi-k2.6` for LLM and `nvidia/nv-embedqa-e5-v5` for embeddings) to accurately parse natural language campaigns, build vector-embedded semantic chunks, extract hooks, stitch topically related clip continuations using similarity search, and dynamically process the output through FFmpeg.

**End-to-end test completed on:** `https://youtu.be/HAnw168huqA` (58-minute podcast)  
**Result:** 15 fully-rendered 9:16 vertical clips in `output/` (subtitled, fades applied).

---

## 2. FILE STRUCTURE

```text
viral_clips_automator/
├── main.py                        # Pipeline orchestrator — CLI entry point, run_pipeline() 8 stages
├── resume.py                      # Manual resume helper — HARDCODED video_id + URL, edit before reuse
├── ui.py                          # Gradio web interface (port 7862)
├── config.yaml                    # SOURCE OF TRUTH for DB, AI models, FFmpeg, chunking, similarity
├── config.py                      # Python wrapper mapping config.yaml and .env → named constants
├── requirements.txt               # Python dependencies (yt-dlp, whisper, gradio, psycopg2, pgvector, etc.)
├── conftest.py                    # pytest path shim (adds project root to sys.path)
├── explanation.md                 # Beginner-friendly end-to-end explainer (non-agent docs)
├── AGENT_CONTEXT.md               # This file — agent source of truth
├── implementation_plan.md         # Historical phased plan (references old MiniMax API, outdated)
│
├── input/
│   ├── __init__.py
│   └── input_handler.py           # LIVE path: normalizes any source into input/raw_video.mp4
│
├── campaign/
│   ├── __init__.py
│   ├── campaign_parser.py         # NVIDIA NIM parser, natural language → JSON config (+ mock fallback)
│   └── templates/                 # podcast_clip.json, tiktok_reaction.json, motivational_reel.json
│
├── db/                            # V3 Database Module (PostgreSQL + pgvector)
│   ├── connection.py              # ThreadedConnectionPool (1-10) + pgvector registration
│   ├── schema.sql                 # Table schemas + HNSW indexes (header still says v3.0, content is v3.1+)
│   └── repositories/              # Abstraction layer for data access
│       ├── video_repo.py          # videos CRUD
│       ├── chunk_repo.py          # chunks CRUD + pgvector similarity search
│       ├── clip_repo.py           # clips CRUD
│       └── run_repo.py            # pipeline_runs checkpointing (start/complete/fail/status)
│
├── pipeline/                      # Core video processing and AI modules
│   ├── __init__.py
│   ├── downloader.py              # DEPRECATED: old yt_dlp helper, NOT used by main.py (uses input_handler)
│   ├── transcriber.py             # LIVE: WhisperX transcription → transcripts/transcript.json
│   ├── chunker.py                 # LIVE: semantic + silence-gap + overlap chunker → DB
│   ├── embedder.py                # LIVE: NVIDIA embeddings → DB (docstring stale, see §7)
│   ├── analyzer.py                # LIVE: Kimi K2.6 → viral clips per chunk (3-retry backoff)
│   ├── similarity.py              # LIVE: pgvector search + 3-way LLM classification
│   ├── clipper.py                 # LIVE: FFmpeg cut / concat stitch
│   ├── regen_srt.py               # LIVE: word-timestamps → per-clip .srt (has hardcoded test main())
│   ├── fast_burn.py               # LIVE: drawtext-chain karaoke burn (has hardcoded test main())
│   └── improvements.py            # LIVE: add_fades(), apply_logo(), mix_music(), remove_silences()
│
├── subtitler.py                   # LEGACY: LLM-formatted SRT + subtitles-filter burn, NOT in main flow
├── karaoke_burn.py                # EXPERIMENT: standalone karaoke script, hardcoded paths, NOT in main flow
│
├── assets/                        # Optional campaign assets
│   ├── logos/logo.png
│   ├── music/                     # empty by default
│   └── style.css                  # unused
│
├── test_*.py                      # pytest suite, partially stale (see §8)
├── transcripts/                   # transcript.json, transcript.txt, clips_analysis.json (generated)
├── clips/                         # intermediates: clip_N.mp4 → _vertical → _subbed → .srt (generated)
├── downloads/                     # audio.wav for WhisperX (generated)
├── input/raw_video.mp4            # normalized input, overwritten each run (generated)
└── output/                        # clip_N_final.mp4 + report.json (generated, overwritten each run)
```

Live render path used by `main.py` is `clipper.py + regen_srt.py + fast_burn.py + improvements.py`. Do not route new code through `subtitler.py` or `karaoke_burn.py`.

---

## 3. TECHNICAL DEEP DIVE: HOW IT WORKS

### 3.1 Input Handling & DB Registration
The system normalizes all input into `input/raw_video.mp4` via `input/input_handler.py` (`detect_source_type()` + `handle_input()`). It supports YouTube (`yt-dlp`), YouTube-live (300s cap), Google Drive (`gdown`), local files (`shutil.copy`), and direct URLs (streamed `requests`).  
Once normalized, `main.py` inserts a new record into the `videos` PostgreSQL table. The `run_repo` manages state checkpointing (transcribe → chunk → embed → analyze → similarity → render) so the pipeline can gracefully resume if crashed via `should_skip_stage()`.

### 3.2 Campaign Parsing
Uses the NVIDIA NIM API (`moonshotai/kimi-k2.6`, `temperature=0.1`, `max_tokens=1000`) to turn a natural language campaign description into a rigorous JSON structure. The LLM must map generic requests to files available in `assets/` (first-available fallback, never hallucinate). The dynamic config overrides static templates found in `campaign/templates/` via `merge_config()` (`{**template, **non_null_campaign}`). If no API key, a keyword-based mock config is returned so UI/tests still run.

### 3.3 Transcription
Uses `whisperx` via `pipeline/transcriber.py` (`base` model, `cpu`, `float32`, `librosa` 16kHz load to bypass torchcodec issues) to extract audio (`downloads/audio.wav` via FFmpeg) and generate word-level timestamped `transcripts/transcript.json` (`segments[].words[].start/end`) + readable `transcript.txt`.

### 3.4 Chunking
`pipeline/chunker.py:build_chunks()` reads the transcript and splits it into discrete semantic chunks. It uses a sliding window with overlap (default 25 seconds) and cuts primarily on silence gaps (>1.2s) and sentence boundaries (`.?!`, 20-word lookahead, gap fallback) to avoid splitting mid-sentence, plus a token-budget safety valve (>1000 tokens → mid-point sentence split) and a 15s minimum-duration filter. Chunks are stored in the DB with `token_count`, `is_overlap_tail`, `silence_gap_before`.

### 3.5 Embedding
`pipeline/embedder.py` passes the text of each chunk to the NVIDIA embeddings endpoint (`nvidia/nv-embedqa-e5-v5` per `config.yaml:embeddings`). This produces a 1024-dimensional vector saved into the `chunks` table as a `vector(1024)` pgvector type. The schema uses an `hnsw` index for fast cosine-distance retrieval. Idempotent (skips already-embedded), 1.5s inter-request delay, 3 retries on 429. Storage uses `input_type="passage"`.

### 3.6 Content Analysis (Viral Clip Extraction)
`pipeline/analyzer.py:analyze()` feeds each chunk independently into the Kimi LLM (`temperature=0.3`, 120s timeout) to identify 0–2 viral moments per chunk. Each API call is wrapped in a **3-attempt exponential backoff retry loop** (`2^attempt` s) to handle NVIDIA NIM `429 Too Many Requests` and `ReadTimeout` errors gracefully.  
The LLM returns a strict JSON: `clip_number`, `start_time`, `end_time`, `hook`, `reason`, `suggested_title`, `suggested_hashtags`. `extract_json_from_response()` strips `<think>` blocks and markdown fences. Clips <15s or malformed are dropped, hook+reason is embedded immediately, rows go to `clips`, then start-time dedupe (5s window, keep longer) runs. Also writes `transcripts/clips_analysis.json` (embeddings stripped) for debugging.

### 3.7 Similarity Search & 3-Way Classification (v3.1)
`pipeline/similarity.py:find_related_segments()` detects if a clip abruptly cuts off before a topic is finished.

**Effective configuration** (`config.yaml` — source of truth):
```yaml
similarity:
  top_k: 3
  threshold: 0.65     # lowered from 0.80 in v3.1 to cast a wider net
  llm_confirmation: true
```
Note: the function signature default is still `threshold=0.80`; `main.py` always passes the config value, so effective runtime threshold is **0.65**. Do not rely on the signature default.

**Flow:**
1. Queries `pgvector` (`<=>` cosine distance, threshold 0.65) to find upcoming chunks similar to the clip's tail.
2. Excludes the clip's own source chunk(s).
3. For each candidate, calls the LLM at `temperature=0.1` for a **three-way classification**:
   - `"stitch"` → Direct narrative continuation. Appended to the clip via FFmpeg concat.
   - `"standalone"` → Same topic, different angle. Recorded in DB but not stitched.
   - `"noise"` → Coincidental overlap. Rejected.
4. Results are stored in the `related_segments` table with `confirmed_by_llm` (bool, true only for `stitch`) and `decision` (text) columns. INSERT uses `ON CONFLICT DO NOTHING` so re-runs are safe.

**Why 3-way instead of binary?** The LLM can now distinguish between a clip that *completes* a thought (stitch) and a related segment that is independently valuable (standalone) — enabling future "related clips" suggestions without forcing a stitch.

### 3.8 FFmpeg Rendering Pipeline (`main.py` render stage)
1. **Clipping & Stitching (`clipper.py`)**: `get_time_ranges_for_clip()` = primary + sorted confirmed continuations, overlapping ranges merged. Single range → single-slice re-encode; multi-range → multi-input `concat=n=:v=1:a=1` filter. Always re-encodes (`libx264`, `preset fast`, `crf 23`).
2. **Vertical Conversion** (inline in `main.py`): Transforms landscape into 9:16 (1080×1920) by blurring the background (`scale+crop+gblur=sigma=20`) and fitting the foreground (`scale=1080:-1`) centered via `overlay`.
3. **Subtitles (`regen_srt.py` & `fast_burn.py`)**: `generate_srt_for_clip()` collects words in `[clip_start, clip_end+0.5]`, shifts to relative time, writes one-word-per-entry `.srt`. `burn_subtitles()` parses it into a chained `drawtext` filter (yellow `0xFFE000`, fontsize 95, Helvetica, uppercased, `y=h*0.82`, `enable=between(t,start,end)`).
4. **Improvements (`improvements.py`)**:
   - Overlays logos (`scale=120:-1`, positions top-right/top-left/bottom-right/bottom-left).
   - Mixes background music via `amix=inputs=2:duration=first` with configurable volume, `-codec:v copy`, `-shortest`.
   - Applies audio/video fade+afade transitions (1.0s in main flow) after `ffprobe` duration lookup.
5. `clip_repo.update_output_path()` per clip, then `output/report.json` is regenerated from DB.

---

## 4. CONFIGURATION REFERENCE

### config.yaml (key blocks — source of truth)
```yaml
similarity:
  top_k: 3
  threshold: 0.65           # cosine similarity floor for candidate chunks
  llm_confirmation: true    # always run 3-way LLM classification

chunking:
  silence_threshold_seconds: 1.2
  max_tokens_per_chunk: 1000
  overlap_seconds: 25
  min_chunk_duration_seconds: 15

embeddings:
  model: "nvidia/nv-embedqa-e5-v5"  # 1024-dim; storage input_type=passage, search=query
```

### config.py (named constants)
All LLM-facing modules import from this single file. **Do not hardcode any URL, key, or model name elsewhere.**

```python
# NVIDIA NIM credentials (source of truth)
NVIDIA_API_KEY    = os.getenv("NVIDIA_API_KEY", "<key>")
NVIDIA_BASE_URL   = "https://integrate.api.nvidia.com/v1"
NVIDIA_LLM_MODEL  = "moonshotai/kimi-k2.6"
NVIDIA_EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"  # 1024-dim, matches schema vector(1024)

# Provider-agnostic aliases used by all pipeline modules
LLM_API_KEY = NVIDIA_API_KEY
LLM_MODEL   = NVIDIA_LLM_MODEL
LLM_API_URL = f"{NVIDIA_BASE_URL}/chat/completions"
```

**Every live pipeline module imports:** `from config import LLM_API_KEY, LLM_MODEL, LLM_API_URL` (embedder uses `NVIDIA_API_KEY + get_config()`).

> MiniMax migration status: the rename to NVIDIA/Kimi is complete for all live code paths, but string remnants remain: `campaign/campaign_parser.py:53` fallback check `"your_minimax_api_key_here"`, `test_campaign_parser.py:42` env patch, and `M2.7` wording in `subtitler.py` comments and old `htmlcov/` reports. `config.py:15` module docstring also still names the old `llama-nemotron-embed-1b-v2` model — ignore it; `config.yaml` + `NVIDIA_EMBED_MODEL` above are authoritative. Do not reintroduce `MINIMAX_*` names.

---

## 5. DATABASE SCHEMA SUMMARY

| Table | Key Columns |
|---|---|
| `videos` | `id`, `source_url`, `raw_path`, `duration_seconds`, `status`, `campaign_id` |
| `chunks` | `id`, `video_id`, `chunk_index`, `start_time`, `end_time`, `text`, `embedding vector(1024)` |
| `clips` | `id`, `video_id`, `clip_number`, `start_time`, `end_time`, `hook`, `reason`, `suggested_title`, `suggested_hashtags`, `embedding vector(1024)`, `output_path` |
| `related_segments` | `id`, `clip_id`, `related_chunk_id`, `similarity_score`, `confirmed_by_llm`, `decision` (stitch\|standalone\|noise), `stitched_into_clip` |
| `pipeline_runs` | `id`, `video_id`, `stage`, `status` (pending\|running\|done\|failed), `error_message`, `started_at`, `completed_at` |

Both `chunks` and `clips` use HNSW indexes on their `embedding` column for fast cosine-distance search. Run `psql viral_clips -f db/schema.sql` once (requires `CREATE EXTENSION vector;`). Note: `schema.sql` header comment still says v3.0 but the body already includes the v3.1 `decision` column — header is stale, body is correct.

---

## 6. SYSTEM INVARIANTS & HARDWARE SPECIFICS

- **Database**: Strictly requires PostgreSQL with the `pgvector` extension (`CREATE EXTENSION vector;`). Vector dimension is locked at **1024** to match `nv-embedqa-e5-v5`. Ignore stale `4096` mentions in `pipeline/embedder.py:25` docstring and `config.py:15` docstring.
- **NVIDIA NIM API**: All credentials come from `NVIDIA_API_KEY` env var (with a hardcoded fallback currently in `config.py:24-27` — rotate/remove it, see §7). Do not hardcode keys/URLs/models elsewhere.
- **Embeddings**: Asymmetric. Storage uses `input_type="passage"`; search queries use `input_type="query"`.
- **FFmpeg Engine**: `libx264`, `preset fast`, `crf 23`, path `~/miniforge3/bin/ffmpeg` with system-`ffmpeg` fallback. Avoid negative timestamps; reset timestamps during cuts.
- **Retry Logic**: `analyzer.py` retries NVIDIA API calls up to 3 times with exponential backoff (`2^attempt` seconds); `embedder.py` waits `20*(attempt+1)`s on 429. Handles rate limits on long videos.
- **Fail-Safes**: All DB stages are logged in `pipeline_runs`. The pipeline uses `should_skip_stage()` to jump over successful phases on restart. Use `resume.py` to manually resume from a specific checkpoint (must edit its hardcoded IDs first).

---

## 7. KNOWN GOTCHAS

- **Rate limits on long videos**: NVIDIA NIM free tier aggressively rate-limits on videos with >30 chunks. The retry loop handles most cases but some chunks may be skipped on very long videos. Clips extracted from successful chunks are still rendered correctly.
- **`torchcodec` / `libavutil` warnings**: These appear on startup in the miniforge3/Python 3.12 environment but do not block execution.
- **Duplicate clip rows**: If `analyze` stage is re-run without clearing the DB, duplicate `clips` rows can appear. Always clear with `DELETE FROM clips WHERE video_id = '<id>';` before re-running analysis on the same video.
- **`related_segments` deduplication**: The INSERT uses `ON CONFLICT DO NOTHING` — safe to re-run similarity stage without duplicating rows.
- **Hardcoded secrets/paths (must fix before sharing):** `config.py:24-27` embeds a real `nvapi-...` fallback key — move to `.env` (`NVIDIA_API_KEY=...`) and delete the literal. `resume.py:13-15` hardcodes a past `video_id` + YouTube URL. `karaoke_burn.py` / `regen_srt.py` / `fast_burn.py` `main()` blocks hardcode old `/Users/rajatthakral/Desktop/clipper/viral-clips` paths — only affects standalone runs, not `main.py`.
- **Stale code comments:** `pipeline/embedder.py:24-26` claims 4096-dim/`vector(4096)`; truth is 1024-dim/`vector(1024)`. `pipeline/similarity.py:21` signature default `threshold=0.80` is overridden by config `0.65` at runtime.

---

## 8. TEST SUITE STATUS (accurate as of v3.2)

`test_*.py` use `unittest.mock` (no real API/DB/FFmpeg). Coverage artifacts live in `htmlcov/`, `.coverage`, `.pytest_cache/`.

- ✅ Current: `test_input_handler`, `test_campaign_parser` (except MiniMax env-name remnant), `test_transcriber`, `test_analyzer`, `test_clipper`, `test_downloader`, `test_subtitler`.
- ❌ Broken: `test_main.py` imports `print_banner` / `print_summary` and mocks old `download_video` / `analyze_transcript` flow that no longer exists in `main.py` (now `run_pipeline` + `handle_input` + `transcribe_audio` + `analyze` + `cut_clips` + `process_clip_subtitles` split). Must be rewritten against `run_pipeline()` before trusting `pytest`.
- `implementation_plan.md` is historical (Phase 1–6, MiniMax-era) — do not treat as current spec.
