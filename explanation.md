# Viral Clips Automator — End-to-End Explanation

> This document explains the project from zero: what it does in simple words, then every technical piece, how files connect, what happens in the background, and what tech it uses.

---

## 1. What Does This Project Do? (Simple Language)

Imagine you have a **1-hour podcast / YouTube video** and you want **15 short vertical videos** for TikTok / Reels / Shorts.

Doing that by hand means:
1. Watching the whole video
2. Finding funny / emotional / surprising moments
3. Cutting those 30–60 sec parts
4. Making them vertical (9:16)
5. Adding big yellow subtitles, logo, music, fades

**This project does all of that automatically.**

You give it:
- A video (YouTube link, Google Drive link, direct URL, or a file on your computer)
- Optionally: instructions in plain English like `"add logo top-right, podcast style, no music"`

It gives you:
- `output/clip_1_final.mp4`, `clip_2_final.mp4`, ... — ready-to-post vertical videos with subtitles
- `output/report.json` — list of clips with hook, timestamps, title, hashtags, reason

You can run it two ways:
- **Command line:** `python main.py <video_path> --campaign "..." --template podcast_clip`
- **Web UI:** `python ui.py` → opens Gradio page on `http://localhost:7862`

Example tested: `https://youtu.be/HAnw168huqA` (58-min podcast) → 15 vertical clips.

---

## 2. The 8-Step Pipeline (High Level)

Every run goes through the same 8 stages in `main.py:run_pipeline()`:

```
1. Normalize input      input/input_handler.py      → input/raw_video.mp4
2. Parse campaign        campaign/campaign_parser.py → JSON config (logo? music? subtitles?)
3. Transcribe            pipeline/transcriber.py     → transcripts/transcript.json (speech → text + timestamps)
4. Chunk                 pipeline/chunker.py         → split text into ~topic chunks, save to DB
5. Embed                 pipeline/embedder.py        → turn each chunk into a 1024-number vector, save to DB
6. Analyze               pipeline/analyzer.py        → LLM finds 0-2 viral moments per chunk, save to DB
7. Similarity            pipeline/similarity.py      → find continuations, decide stitch/standalone/noise
8. Render                pipeline/clipper.py + regen_srt.py + fast_burn.py + improvements.py
                        → cut → make vertical → burn subtitles → logo → music → fades → output/
```

Each stage is **checkpointed in the database** (`pipeline_runs` table). If the program crashes at step 6, re-running skips steps 1–5. That logic is `should_skip_stage()` in `main.py:45` + `resume.py`.

---

## 3. Architecture Diagram

```
                    ┌─────────────┐
                    │  User input │
                    │ YouTube /   │
                    │ Drive / file│
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐      ┌──────────────────┐
                    │ input_handler│     │ campaign_parser  │
                    │ download/    │      │ NVIDIA LLM turns │
                    │ copy to      │      │ English → JSON   │
                    │ raw_video.mp4│      └────────┬─────────┘
                    └──────┬──────┘               │
                           ▼                      ▼
                    ┌─────────────────────────────────────┐
                    │              main.py                │
                    │        orchestrator (8 stages)       │
                    └──────┬──────────────────────────────┘
                           ▼
        ┌────────────────────────────────────────────────────┐
        │                PostgreSQL + pgvector                 │
        │ videos | chunks | clips | related_segments |        │
        │ pipeline_runs                                       │
        └────────────────────────────────────────────────────┘
                           ▲   ▲   ▲
                           │   │   │
   ┌──────────────┐  ┌─────┴───┴───┴──────┐  ┌──────────────┐
   │ WhisperX     │  │ NVIDIA NIM APIs    │  │ FFmpeg       │
   │ speech→text  │  │ LLM + embeddings   │  │ cut/resize/  │
   │ (local)      │  │ (cloud)            │  │ burn/mix     │
   └──────────────┘  └────────────────────┘  └──────────────┘
```

Two kinds of “AI” are used:
1. **Local AI (WhisperX):** converts audio to text on your machine. No internet needed after model download.
2. **Cloud AI (NVIDIA NIM):** `moonshotai/kimi-k2.6` for understanding text, `nvidia/nv-embedqa-e5-v5` for embeddings. Needs `NVIDIA_API_KEY`.

One database (PostgreSQL) glues everything together. One video tool (FFmpeg) does all video work.

---

## 4. Tech Stack

| Layer | Technology | Why it exists |
|---|---|---|
| Language | **Python 3.12** | Whole project |
| Video download | **yt-dlp**, `gdown`, `requests`, `shutil` | YouTube / Drive / direct URL / local copy |
| Speech-to-text | **WhisperX** (`openai-whisper` + alignment models) + `librosa` | Word-level timestamps (`start`, `end` per word) |
| LLM (brain) | **NVIDIA NIM** `moonshotai/kimi-k2.6` via OpenAI-compatible `/chat/completions` | Campaign parsing, viral-moment detection, similarity classification, subtitle cleanup |
| Embeddings | **NVIDIA NIM** `nvidia/nv-embedqa-e5-v5`, 1024-dim | Turn text into numbers so “similar meaning” = “close numbers” |
| Database | **PostgreSQL + pgvector extension** via `psycopg2-binary`, `pgvector` Python lib | Store videos, chunks, clips, progress. Vector search with HNSW index + cosine distance `<=>` |
| Video editing | **FFmpeg / FFprobe** (`~/miniforge3/bin/ffmpeg`, fallback to system `ffmpeg`) with `libx264`, `preset fast`, `crf 23`, `aac` | Cut, concat, scale/crop/blur to 1080x1920, drawtext subtitles, logo overlay, amix music, fade/afade |
| Config | **YAML** (`config.yaml`) + `python-dotenv` + `config.py` | All tunables in one place, secrets in `.env` |
| Web UI | **Gradio** (`ui.py`, port 7862) | Textboxes + gallery, progress bar |
| Tests | **pytest** (`test_*.py`, `conftest.py`) | Unit tests with mocks — no real API/DB needed |

`requirements.txt` is short: `yt-dlp, openai-whisper, requests, gdown, gradio, pyyaml, python-dotenv, psycopg2-binary, pgvector`. Everything else (torch, librosa, whisperx, FFmpeg) is expected in the conda/miniforge env.

---

## 5. File Structure — What Each File Does

```
viral-clips/
├── main.py                  # THE BOSS. run_pipeline() runs all 8 stages. CLI entry point.
├── ui.py                    # Gradio web page. Calls run_pipeline() + shows config preview + gallery.
├── resume.py                # Manual resume helper. Hardcodes a video_id and re-runs from analyze onward.
├── config.yaml              # All settings: ffmpeg path, AI models, DB, chunking, similarity.
├── config.py                # Reads config.yaml + .env. Exposes NVIDIA_API_KEY, LLM_MODEL, get_config().
├── requirements.txt         # pip dependencies.
├── conftest.py              # Makes tests importable (adds project root to sys.path).
├── test_*.py                # Unit tests (see §11).
│
├── input/
│   ├── __init__.py
│   └── input_handler.py     # detect_source_type() + handle_input() → always returns input/raw_video.mp4
│
├── campaign/
│   ├── campaign_parser.py   # parse_campaign(description, assets) → JSON via LLM. Has mock fallback if no key.
│   └── templates/           # podcast_clip.json, tiktok_reaction.json, motivational_reel.json
│
├── pipeline/                # Core processing
│   ├── downloader.py        # OLD helper (yt_dlp direct). Mostly superseded by input_handler.py.
│   ├── transcriber.py       # transcribe_audio() — WhisperX + align → transcript.json + transcript.txt
│   ├── chunker.py           # build_chunks() — silence-gap + sentence-boundary + token-budget + overlap → DB
│   ├── embedder.py          # embed_text() + embed_all_chunks() — NVIDIA embeddings → DB vector column
│   ├── analyzer.py          # analyze() — LLM per chunk → 0-2 clips → dedupe → DB + clips_analysis.json
│   ├── similarity.py        # find_related_segments() — pgvector search + LLM stitch/standalone/noise → DB
│   ├── clipper.py           # cut_clips() — FFmpeg cut or concat (primary + continuations)
│   ├── regen_srt.py         # generate_srt_for_clip() — words in time window → .srt text (word-per-line)
│   ├── fast_burn.py         # burn_subtitles() — parse .srt → chain of drawtext filters → yellow karaoke burn
│   └── improvements.py      # add_fades(), apply_logo(), mix_music(), remove_silences()
│
├── db/
│   ├── connection.py        # ThreadedConnectionPool (1-10 conns) + pgvector registration
│   └── repositories/
│       ├── video_repo.py    # insert/get video rows
│       ├── chunk_repo.py    # insert chunks, update embeddings, find_similar_chunks (pgvector SQL)
│       ├── clip_repo.py     # insert clips, update output_path/embedding, get clips
│       └── run_repo.py      # start/complete/fail/get_stage_status — resume logic
│
├── subtitler.py             # LEGACY subtitle path: LLM formats SRT + FFmpeg subtitles filter. Not used in main flow now.
├── karaoke_burn.py          # STANDALONE experiment script for karaoke effect (has hardcoded paths + test clips).
│
├── assets/logos/logo.png    # Optional overlay image
├── assets/music/            # (empty) — put .mp3s here for background music
├── assets/style.css         # Unused styling
│
├── input/raw_video.mp4      # Normalized input (overwritten each run)
├── downloads/audio.wav      # Extracted audio for WhisperX
├── transcripts/             # transcript.json (full), transcript.txt (readable), clips_analysis.json
├── clips/                   # Intermediates: clip_N.mp4 → _vertical → _subbed → _logo → _music → .srt
├── output/                  # Finals: clip_N_final.mp4 + report.json
```

Legacy / experimental files you can ignore for normal runs: `pipeline/downloader.py`, `subtitler.py`, `karaoke_burn.py`. The live render path is `regen_srt.py + fast_burn.py + improvements.py` called from `main.py:142-220`.

---

## 6. End-to-End Data Flow (What Happens in the Background)

### Stage 1 — Input normalizing (`input/input_handler.py:79`)
- `detect_source_type()` looks at the string: `youtube.com/youtu.be` → youtube, `drive.google.com` → drive, ends with `.mp4/.mov/...` + starts with `/` or `./` → local, ends with video ext otherwise → direct URL.
- YouTube → `yt-dlp -f bestvideo+bestaudio --merge-output-format mp4 -o input/raw_video.mp4`
- Live → records max 300s. Drive → `gdown`. Local → `shutil.copy`. Direct → streaming `requests.get`.
- **Always outputs the same path** so downstream never cares where it came from.

### Stage 2 — Campaign parsing (`campaign/campaign_parser.py:35`)
- `scan_assets()` lists `assets/logos/` + `assets/music/`.
- Prompt = system rules (“return ONLY JSON, never invent filenames, only include what user asked”) + user description + available files.
- LLM `moonshotai/kimi-k2.6`, `temperature 0.1`, `max_tokens 1000` → JSON like:
  ```json
  { "logo": "logo.png", "logo_position": "top-right", "music": null,
    "subtitles": true, "aspect_ratio": "9:16", "template": "podcast_clip", "fade": true }
  ```
- If `template` set and file exists in `campaign/templates/*.json`, merge: `merged = {**template, **campaign_non_null}`.
- If no API key → mock keyword matching (`"logo" in description`, etc.) so UI/tests still work.

### Stage 3 — Transcription (`pipeline/transcriber.py:7`)
1. `ffmpeg -i raw_video.mp4 -map a downloads/audio.wav` (audio only).
2. `whisperx.load_model("base", device="cpu", compute_type="float32")` → `model.transcribe(audio_16kHz)`.
3. `whisperx.load_align_model() + whisperx.align()` → word-level `start/end` per word.
4. Save `transcripts/transcript.json` = `{segments: [{start, end, text, words: [{word, start, end}]}]}` + `transcript.txt` readable version.
- Why word-level? Subtitles and chunking need exact timing, not just sentences.

### Stage 4 — Chunking (`pipeline/chunker.py:83`)
Goal: split a 1-hour transcript into searchable topic blocks without cutting mid-thought.
- **A. Silence gaps:** if gap between words ≥ `silence_threshold_seconds` (1.2s) → candidate cut.
- **B. Semantic validation:** confirm cut only if previous word ends with `.?!`. Else look ahead 20 words for next sentence end; fallback to gap itself.
- **C. Token safety valve:** if chunk text > `max_tokens_per_chunk` (1000 ≈ 4000 chars), split at mid-point sentence boundary.
- **D. Overlap:** prepend last `overlap_seconds` (25s) of previous chunk to next chunk, so a viral moment straddling a boundary appears in both.
- Skip chunks shorter than `min_chunk_duration_seconds` (15s).
- Each chunk → `chunk_repo.insert_chunk(video_id, index, start, end, text, tokens, is_overlap, gap)` → Postgres.

### Stage 5 — Embedding (`pipeline/embedder.py:99`)
- For each chunk: `POST https://integrate.api.nvidia.com/v1/embeddings {model, input:[text], input_type:"passage", ...}` → 1024-float vector.
- `chunk_repo.update_embedding(chunk_id, vector)` → `chunks.embedding vector(1024)` column.
- Idempotent (skips chunks that already have embedding), 1.5s delay + 3 retries on 429 rate-limit. Free tier ≈ 40 req/min.

### Stage 6 — Analysis (`pipeline/analyzer.py:203`)
- For **each chunk independently**: send text + its time bounds to LLM (`temperature 0.3`, 120s timeout, 3 retries with exponential backoff).
- Prompt asks for “0 to 2 best moments, 20–90s, strong hook, no abrupt start/end” → JSON array with `start_time, end_time, hook, reason, suggested_title, hashtags`.
- `extract_json_from_response()` strips `<think>` blocks and markdown fences — Kimi often wraps answers.
- Filter: duration ≥ 15s, valid timestamps. Embed `hook + reason` via `embed_text()` → `clip_repo.insert_clip(...)`.
- **Dedupe:** sort by start; if two clips start within 5s, keep longer one (handles overlap-induced duplicates).
- Also writes `transcripts/clips_analysis.json` (embeddings stripped) for debugging.

### Stage 7 — Similarity (`pipeline/similarity.py:20`)
Problem: a clip may cut off mid-story (“...and then he—” end). Solution: find its continuation later in the video.
1. `chunk_repo.find_similar_chunks(clip_embedding, video_id, top_k=3, threshold=0.65)`:
   ```sql
   SELECT ..., 1 - (embedding <=> query::vector) AS similarity
   WHERE embedding IS NOT NULL AND similarity >= 0.65
   ORDER BY embedding <=> query ASC LIMIT 3
   ```
   (`<=>` = cosine distance in pgvector; HNSW index makes it fast.)
2. Exclude the clip’s own source chunk.
3. For each candidate → LLM `temperature 0.1` 3-way classify:
   - `stitch` → direct continuation → will be concatenated
   - `standalone` → same topic, different angle → saved but not stitched
   - `noise` → coincidental → rejected
4. `INSERT INTO related_segments ... ON CONFLICT DO NOTHING` (safe re-run) with `confirmed_by_llm=(decision=="stitch")`.

### Stage 8 — Render (`main.py:142-220`, `pipeline/*.py`)
For each clip:
1. **Cut/stitch (`clipper.py`):** `get_time_ranges_for_clip()` = primary + sorted confirmed continuations, merged if overlapping. One range → single FFmpeg slice (re-encode `libx264/fast/crf23/aac`). Multiple → multi-input + `concat=n=:v=1:a=1` filter → `clips/clip_N.mp4`.
2. **Vertical (`main.py:171`):** single FFmpeg `filter_complex`: scale up + crop to 1080x1920 + `gblur=sigma=20` background + overlay scaled foreground centered → `clip_N_vertical.mp4`.
3. **Subtitles (`regen_srt.py` + `fast_burn.py`):** collect words with `clip_start <= start/end <= clip_end+0.5`, shift to relative time, write one-word-per-entry `.srt`. Then build a long `drawtext` chain (one per word, yellow `0xFFE000`, fontsize 95, Helvetica, centered at `y=h*0.82`, `enable=between(t,start,end)`) → `clip_N_subbed.mp4`. Upper-cases text for TikTok style.
4. **Logo (`improvements.py:58`):** `scale=120:-1` + `overlay=W-w-20:20` (etc. per position) → `clip_N_logo.mp4`. Audio copied.
5. **Music (`improvements.py:86`):** `[music]volume=X + [orig][music]amix=inputs=2:duration=first` → `clip_N_music.mp4`. Video copied (`-codec:v copy`), `-shortest`.
6. **Fades (`improvements.py:25`):** `ffprobe` duration → `fade in/out + afade in/out` (1.0s in main flow) → `output/clip_N_final.mp4`. If fade disabled, rename.
7. `clip_repo.update_output_path(clip_id, final_path)`.
8. After loop: query all clips → `output/report.json` with `clip_number, hook, start/end, duration, reason, title, output_file`.

---

## 7. Database — Tables & Why

PostgreSQL + `CREATE EXTENSION vector;`. Pool: `ThreadedConnectionPool(min1/max10)` in `db/connection.py`. Every repo does `get_conn() → cursor → commit → release_conn()`.

| Table | Key columns | Purpose |
|---|---|---|
| `videos` | `id uuid, source_url, raw_path, duration_seconds, status, campaign_id` | One row per run. `insert_video()` in main. |
| `chunks` | `id, video_id, chunk_index, start_time, end_time, text, token_count, is_overlap_tail, silence_gap_before, embedding vector(1024)` + HNSW index | Searchable transcript blocks. |
| `clips` | `id, video_id, clip_number, start_time, end_time, duration_seconds, hook, reason, suggested_title, suggested_hashtags text[], source_chunk_ids uuid[], embedding vector(1024), output_path` + HNSW | Viral moments. |
| `related_segments` | `id, clip_id, related_chunk_id, similarity_score, confirmed_by_llm bool, decision (stitch/standalone/noise), stitched_into_clip` | Continuation decisions. |
| `pipeline_runs` | `id, video_id, stage, status (pending/running/done/failed), error_message, started_at, completed_at` | Resume checkpoints. |

Key SQL trick (`db/repositories/chunk_repo.py:80`): cosine similarity = `1 - (embedding <=> query)`. HNSW index on `embedding` makes top-K fast even with hundreds of chunks.

Resume logic: `run_repo.start_stage()` deletes non-done prior row then inserts `running`; `complete_stage()` → `done`; `fail_stage()` → `failed` + message. `main.py:should_skip_stage()` skips `done` stages. `resume.py` is a hardcoded manual version of the same.

---

## 8. Configuration

**`config.yaml`** — edit this to tune behavior without touching code:
- `ffmpeg.path, encoder, preset, crf` — video quality/speed.
- `paths.*` — input/output/assets dirs.
- `ai.provider/llm_model/llm_endpoint` — currently NVIDIA NIM.
- `defaults.min_clip_duration, aspect_ratio, music_volume, fade_duration`.
- `chunking.silence_threshold_seconds (1.2), max_tokens (1000), overlap (25s), min_duration (15s)`.
- `embeddings.model/endpoint/input_type_storage(passage)/input_type_query(query)` — asymmetric models need different modes for store vs search.
- `similarity.top_k (3), threshold (0.65), llm_confirmation (true)`.

**`config.py`** — two faces:
- Legacy constants (`WHISPER_MODEL="base"`, `MIN_CLIP_DURATION`, subtitle style) for older modules.
- `get_config()` (cached) reads YAML + overrides `DB_USER`/`DB_PASSWORD` from `.env`.
- NVIDIA block is source of truth: `NVIDIA_API_KEY` (env or hardcoded fallback — **rotate this key, it is exposed**), `NVIDIA_BASE_URL`, `LLM_MODEL=kimi-k2.6`, `EMBED_MODEL=nv-embedqa-e5-v5`, plus aliases `LLM_API_KEY/MODEL/URL` imported everywhere.

---

## 9. User Interfaces

**CLI (`main.py:242`):**
```bash
python main.py <video_path_or_url> --campaign "add logo top right, no music" --template podcast_clip
```

**Web (`ui.py`):**
- Left: `video_input` textbox, `campaign_description` textbox, `template` radio (None + 3), Generate button, status box.
- Right: parsed campaign JSON preview (runs `parse_campaign` before pipeline so you see what AI understood).
- Bottom: `Gallery` of final mp4s.
- `generate_clips_wrapper()` yields 3 times: input validation → “Starting” + config → final clips or error. Progress passed into `run_pipeline(progress)` for the 8-step bar (0.05 → 0.8).

---

## 10. Background Processes & External Calls (What “runs where”)

| What | Where | When |
|---|---|---|
| `yt-dlp` subprocess | Your machine | Stage 1, downloads video |
| `ffmpeg`/`ffprobe` subprocesses | Your machine (`~/miniforge3/bin/`) | Audio extract, cut, vertical, burn, logo, music, fades — heaviest CPU part |
| WhisperX `base` model + align model | Your machine (CPU, float32) | Stage 3, minutes for long video |
| `POST /chat/completions` (Kimi) | `integrate.api.nvidia.com` | Campaign parse (1×), analyzer (1× per chunk), similarity confirm (1× per candidate) — most rate-limit risk |
| `POST /embeddings` (Nemotron) | `integrate.api.nvidia.com` | Embedder (1× per chunk) + analyzer (1× per clip hook) |
| PostgreSQL queries | `localhost:5432/viral_clips` | Every stage — inserts, vector search, checkpoints |
| Gradio server | `localhost:7862` | Only when running `ui.py` |

Failure modes to know: NVIDIA 429 rate-limit (handled with 20s×attempt waits in embedder, 2^attempt backoff in analyzer; very long videos >30 chunks may still drop some); `torchcodec/libavutil` warnings on startup (harmless); re-running `analyze` without clearing DB creates duplicate clip rows; FFmpeg needs the miniforge path or system fallback.

---

## 11. Tests

`test_*.py` use `unittest.mock` — they **do not** call real APIs, DB, or FFmpeg:

- `test_input_handler` — source detection + dispatch.
- `test_campaign_parser` — JSON parsing, mock fallback.
- `test_transcriber / test_analyzer / test_clipper / test_downloader / test_subtitler` — per-module logic with fakes.
- `test_main` — full `main()` flow with all 5 stages mocked (note: it references old `print_banner/print_summary/download_video` functions that no longer exist in current `main.py`, so it will fail until updated).

Run: `pytest` (see `.coverage`, `htmlcov/`, `.pytest_cache/` — coverage artifacts from last run).

---

## 12. Folders That Fill Up As You Run

- `input/` → `raw_video.mp4` (overwritten every run — old input lost).
- `downloads/` → `audio.wav`.
- `transcripts/` → `transcript.json`, `transcript.txt`, `clips_analysis.json`.
- `clips/` → per-clip intermediates (`clip_N.mp4`, `_vertical`, `_subbed`, `.srt`, ...).
- `output/` → finals + `report.json` (overwritten every run).
- `assets/logos, assets/music` → you supply these; parser matches names.

---

## 13. How to Run / Modify (Cheat Sheet)

```bash
# setup
pip install -r requirements.txt
# needs: postgres running, CREATE EXTENSION vector;, ffmpeg, .env with NVIDIA_API_KEY + DB_PASSWORD

# CLI
python main.py "https://youtu.be/XXXX" --campaign "podcast style, add logo" --template podcast_clip

# Web
python ui.py   # → http://localhost:7862

# Resume after crash (edit video_id inside first)
python resume.py

# Tests
pytest
```

To change clip style: edit `campaign/templates/*.json` or pass different `--campaign` English. To get more/fewer clips: tune `chunking.*` + `similarity.threshold` in `config.yaml`. To change subtitle look: `pipeline/fast_burn.py:build_filter_complex()` (fontsize 95, yellow `0xFFE000`, Helvetica). To change vertical look: `main.py:171-178` blur/crop filter.

---

## 14. Security / Cleanup Notes

- `config.py:24` contains a hardcoded `nvapi-...` fallback key. Anyone with this file can use your NVIDIA quota. Move it to `.env` (`NVIDIA_API_KEY=...`) and remove the literal.
- `DB_PASSWORD` is empty by default (macOS trust auth). Set via `.env` on shared machines.
- `resume.py:13` hardcodes a real `video_id` + YouTube URL from a past run — update before reuse.
- `karaoke_burn.py:35,101` and `regen_srt.py:35` hardcode `/Users/rajatthakral/Desktop/clipper/viral-clips` (old path) — only matters if you run those standalone scripts.
