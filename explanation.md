# Viral Clips Automator — End-to-End Explanation

> This document explains the project from zero: what it does in simple words, then every technical piece, how files connect, what happens in the background, and what tech it uses. (Last synced 06 Oct 2026 — 11-stage pipeline, Groq LLM + local embeddings.)

---

## 1. What Does This Project Do? (Simple Language)

Imagine you have a **1-hour podcast / YouTube video** and you want **a handful of short vertical videos** for TikTok / Reels / Shorts.

Doing that by hand means:
1. Watching the whole video
2. Finding funny / emotional / surprising moments
3. Cutting those 20–90 sec parts (without chopping sentences in half)
4. Making them vertical (9:16) with the speaker nicely framed
5. Adding readable karaoke subtitles, logo, music, fades

**This project does all of that automatically.**

You give it:
- A video (YouTube link, YouTube live, Google Drive link, direct URL, or a file on your computer)
- Optionally: instructions in plain English like `"add logo top-right, podcast style, no music"` + a style template + a framing layout

It gives you:
- `output/clip_1_final.mp4`, `clip_2_final.mp4`, ... — ready-to-post vertical videos
- `output/report.json` — hook, refined timestamps, source ranges, layout + reasons, titles

You can run it two ways:
- **Command line:** `python main.py <video> --campaign "..." --template podcast_clip --layout auto`
- **Web UI:** `python ui.py` → `http://localhost:7862` (Generate + Review & Correct tabs)

Latest verified example: a 324s DIY sample → 3 clips (35–50s each). The system aims for up to 8 but deliberately returns fewer when quality is insufficient — the quality floor + duplicate removal explain that, by design.

---

## 2. The 11-Step Pipeline (High Level)

Every run goes through the same stages in `main.py:run_pipeline()`:

```
1.  Normalize input      input/input_handler.py      → input/raw_video.mp4
2.  Parse campaign        campaign/campaign_parser.py → JSON config (logo? music? layout? ...)
3.  Transcribe            pipeline/transcriber.py     → transcripts/transcript.json (words + timestamps)
4.  Chunk                 pipeline/chunker.py         → topic blocks, saved to DB
5.  Audio events          pipeline/audio_events.py    → applause/laughter spikes → extra candidates
6.  Embed                 pipeline/embedder.py        → each chunk → 1024 numbers, saved to DB
7.  Outline               pipeline/outline.py         → whole-video structure (topics, Q&A, stories)
8.  Analyze               pipeline/analyzer.py        → outline + events → ranked clip picks (fusion)
9.  Similarity            pipeline/similarity.py      → continuations? stitch / standalone / noise
10. Refine boundaries      pipeline/boundaries.py      → snap to complete sentences, LLM-checked
11. Render                 clipper + framing + captions + editplan + polish
                         → cut → face-aware vertical → ASS subtitles → logo/music/fades → output/
```

Each stage is **checkpointed in the database** (`pipeline_runs` table + fingerprints). If the program crashes at step 9, re-running skips finished steps. `rerun_from_embed.py <video_id>` safely re-runs from step 6 onward.

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
                    │ download/    │      │ Groq LLM turns   │
                    │ copy to      │      │ English → JSON   │
                    │ raw_video.mp4│      └────────┬─────────┘
                    └──────┬──────┘               │
                           ▼                      ▼
                    ┌─────────────────────────────────────┐
                    │              main.py                │
                    │        orchestrator (11 stages)      │
                    └──────┬──────────────────────────────┘
                           ▼
        ┌────────────────────────────────────────────────────┐
        │                PostgreSQL + pgvector                 │
        │ videos | chunks | clips | related_segments |        │
        │ candidates | outlines | audio_events | edit_plans | │
        │ pipeline_runs + fingerprints                        │
        └────────────────────────────────────────────────────┘
                           ▲   ▲   ▲
                           │   │   │
   ┌──────────────┐  ┌─────┴───┴───┴──────┐  ┌──────────────┐
   │ WhisperX     │  │ Groq LLM (cloud)   │  │ FFmpeg +     │
   │ speech→text  │  │ + local embeddings │  │ OpenCV faces │
   │ (local)      │  │ (Qwen3, local)     │  │ cut/frame/   │
   └──────────────┘  └────────────────────┘  │ burn/mix     │
                                             └──────────────┘
```

Two kinds of “AI” are used:
1. **Local AI:** WhisperX (speech→text) and Qwen3 embeddings (text→numbers) run on your machine. No internet needed after model download, no embedding bills.
2. **Cloud AI (Groq):** `openai/gpt-oss-120b` for understanding text — campaign parsing, finding clip candidates, checking boundaries, classifying continuations. Needs `GROQ_API_KEY` (+ `_2`, `_3`, `_FALLBACK` — stages are split across keys with automatic fallback).

One database (PostgreSQL) glues everything together. One video tool (FFmpeg, plus OpenCV/MediaPipe for faces) does all video work.

---

## 4. Tech Stack

| Layer | Technology | Why it exists |
|---|---|---|
| Language | **Python 3.10+** | Whole project |
| Video download | **yt-dlp**, `gdown`, `requests`, `shutil` | YouTube (+360p fallback) / Drive / direct URL / local copy |
| Speech-to-text | **WhisperX** + `librosa` | Word-level timestamps (`start`, `end` per word) |
| LLM (brain) | **Groq** `openai/gpt-oss-120b` via OpenAI-compatible API | Campaigns, candidates, boundary checks, stitch decisions |
| Embeddings | **Local** `Qwen3-Embedding-0.6B`, 1024-dim | “Similar meaning” = “close numbers”, no API needed |
| Database | **PostgreSQL + pgvector** via `psycopg2-binary`, `pgvector` lib | Videos, chunks, clips, candidates, plans, progress; HNSW vector search |
| Video editing | **FFmpeg/FFprobe** (`libx264`, `fast`, `crf 23`, `aac`), **OpenCV**, **MediaPipe** | Cut, face-track, 1080×1920, ASS subtitles, logo, ducked music, fades |
| Config | **YAML** (`config.yaml`) + `python-dotenv` + `config.py` | All tunables in one place, secrets in `.env` |
| Web UI | **Gradio** (`ui.py`, port 7862) | Generate + Review & Correct tabs |
| Tests | **pytest** (220/220 green, mocked) | Fast unit tests without API/DB/FFmpeg |

---

## 5. File Structure — What Each File Does

```
viral-clips/
├── main.py                  # THE BOSS. run_pipeline() runs all 11 stages. CLI entry point.
├── ui.py                    # Gradio page: generate clips, then Review & Correct single clips.
├── rerun_from_embed.py      # Re-run helper: Dup-safe cleanup + embed→render for a video_id.
├── preview_framing.py / demo_faces.py / bench_transcribe.py   # Verification scripts only.
├── config.yaml              # ALL settings (chunking, similarity, outline, discovery, audio,
│                            # fusion, captions, refine, framing, export, ...).
├── config.py                # Reads config.yaml + .env. Groq key/model, embedding settings.
│
├── input/input_handler.py   # detect_source_type() + handle_input() → input/raw_video.mp4
│
├── campaign/
│   ├── campaign_parser.py   # parse_campaign(description, assets) → JSON. Mock fallback if no key.
│   └── templates/           # podcast_clip.json, tiktok_reaction.json, motivational_reel.json
│
├── pipeline/
│   ├── transcriber.py       # WhisperX + align → transcript.json/txt (+ glossary fixes).
│   ├── chunker.py           # silence-gap + sentence + token-budget + overlap → DB.
│   ├── audio_events.py      # loud-moment detector → extra candidates from transcript.
│   ├── audio_classifier.py  # optional laughter/applause labels (off by default, never faked).
│   ├── embedder.py          # local embeddings → DB vector column (batched, skips done).
│   ├── outline.py           # whole-video map: sections, questions, story arcs.
│   ├── discovery.py         # proposes candidates citing exact sentence IDs (validated).
│   ├── analyzer.py          # combines outline + audio cues, ranks, saves clips to DB.
│   ├── fusion.py            # merges overlaps, scores quality, picks diverse winners.
│   ├── similarity.py        # finds continuations: stitch / standalone / noise.
│   ├── boundaries.py        # snaps clip edges to complete sentences (LLM-checked).
│   ├── fingerprints.py      # detects changed inputs so stale stages re-run.
│   ├── clipper.py           # FFmpeg cut / concat (with anti-Frankenstein stitch caps).
│   ├── framing.py           # face-aware 9:16: 1 face→follow, 2 faces→stacked, unsure→brand color.
│   ├── captions.py          # phrase karaoke subtitles (ASS), top/bottom auto-placement.
│   ├── glossary.py          # fixes names/terms in subtitles, logged to glossary.json.
│   ├── regen_srt.py         # words → .srt subtitles with stitch-aware timing.
│   ├── editplan.py          # versioned per-clip plan (preview and export share it).
│   ├── polish.py            # ONE ffmpeg pass: subs + logo + music + loudness + fades.
│   ├── review.py            # powers the Review tab: boundary/layout/caption corrections.
│   ├── diarization.py / speakers.py / visual.py   # optional, all OFF by default.
│   └── improvements.py / fast_burn.py             # fallback render chain (older style).
│
├── db/                      # connection.py, schema.sql, migrations v4–v6, repositories/
├── assets/logos, assets/music  # you supply these; the parser matches names.
│
├── input/raw_video.mp4      # Normalized input (overwritten each run)
├── downloads/audio.wav      # Extracted audio for WhisperX
├── transcripts/             # transcript.json/txt, outline.json, clips_analysis.json, ...
├── clips/                   # Per-clip work files (cuts, subtitles, plans)
├── output/                  # Finals: clip_N_final.mp4 + report.json
```

---

## 6. End-to-End Data Flow (What Happens in the Background)

### Stages 1–2 — Input + campaign
- Source detection → download/copy → always `input/raw_video.mp4`; video row registered in DB.
- Campaign LLM reads your English + available logos/music → strict JSON (logo? music? subtitles? **layout**?). Template merged in; no key → keyword mock so the app still runs.

### Stage 3 — Transcription
1. `ffmpeg -i raw_video.mp4 downloads/audio.wav` (audio only).
2. WhisperX transcribes (CUDA if available — ~7× faster, identical words) + aligns to per-word timestamps.
3. Optional glossary fixes display text only (timings untouched, logged). Saves `transcript.json` + readable `transcript.txt`.

### Stage 4 — Chunking
Silence (≥1.2s) + sentence punctuation (+20-word lookahead) + 1000-token safety split + 25s overlap + 15s minimum → searchable topic blocks in the DB.

### Stage 5 — Audio events
Loud spikes (laughter, applause, reactions) detected against a rolling baseline; each spike looks at the surrounding 25s-before/10s-after transcript and becomes a candidate only if real speech supports it. Never uses spike timing as clip edges — always sentence-aligned.

### Stage 6 — Embedding
Each chunk → 1024 numbers via the local model (batched, skips already-done). Stored in Postgres with an HNSW index for fast “find similar meaning” search.

### Stages 7–8 — Outline → discovery → fusion (the “analyze” step)
1. **Outline:** transcript windows → section map (titles, topics, Q&A, story setup→payoff). Falls back to a simple deterministic map without an API key.
2. **Discovery:** proposes candidates that cite exact sentence IDs; every ID validated, spans forced into 15–90s (this validation once caught a 283-second monster candidate).
3. **Fusion:** merges near-duplicates, scores each on 7 editorial components (audio evidence is only a 5% bonus), picks a diverse set up to 8, enforcing a 0.45 quality floor — weak ones are rejected *with written reasons*. Winners become `clips` rows.

### Stage 9 — Similarity
For each clip: vector-search similar chunks → LLM judges each as `stitch` (direct continuation → concatenated), `standalone` (same topic, separate — saved, not joined), or `noise` (rejected). Safety caps: no join across >45s gaps, no output over 90s (added after a 120s “Frankenstein” stitch was caught live).

### Stage 10 — Boundary refinement
Snaps every clip edge to sentence starts/ends (±0.15s/0.3s padding inside neighbouring silence), then the LLM double-checks completeness from ±20s context using sentence IDs (timestamps always come from the transcript, never the LLM). Incomplete endings → earlier complete ending or rejection with reason. Stitched parts refined separately. `--no-refine` skips this.

### Stage 11 — Render (per clip)
1. **Cut/stitch:** refined ranges cut via FFmpeg (multi-part → concat).
2. **Vertical:** hard requirement — every shot is a face-/subject-tracked 9:16 crop filling 1080×1920 (face scenes follow the speaker, workbench scenes hold the work area or centre crop; manual 9:16 ROIs from the Review tab always win). Never letterboxed, never stretched; embedded source bars are detected and removed before cropping. Audio carried over.
3. **Subtitles:** words → gap-free timing → phrase cues (≤8 words, 2 lines) → yellow karaoke ASS burn in one pass (top placement when faces sit low).
4. **Edit plan:** structured plan saved versioned (DB + `clip_N_plan.json`) — preview and export use the same plan.
5. **Polish:** ONE ffmpeg pass — subtitles, fades (video only; speech never faded), logo, ducked background music, loudness normalization, limiter.
6. Final saved, DB updated, `output/report.json` regenerated with refined times, ranges, layouts, and reasons.

### Review & Correct (UI tab)
Ranked candidates + per-clip detail → edit boundaries (validated), layout, or caption words (timings preserved) → each edit bumps the plan version → re-render just that clip.

---

## 7. Database — Tables & Why

PostgreSQL + `CREATE EXTENSION vector;`. Pool of 1–10 connections; every repo does `get_conn() → cursor → commit → release_conn()`. Apply `schema.sql` then migrations v4→v5→v6.

| Table | Purpose |
|---|---|
| `videos` | One row per run (source URL, raw path, status) |
| `chunks` | Searchable transcript blocks + 1024-dim vectors (HNSW) |
| `clips` | Chosen viral moments + hook/reason/title + refined ranges/timeline/layout + output path |
| `related_segments` | Continuation decisions (`stitch/standalone/noise`) |
| `candidates` | The full pool: outline + audio-event proposals with scores and statuses |
| `outlines` | Whole-video section map |
| `audio_events` | Detected loud moments (source timestamps) |
| `edit_plans` | Versioned per-clip render plans |
| `pipeline_runs` / `stage_fingerprints` | Resume checkpoints + change detection |

Resume logic: finished stages are skipped; changed inputs (new transcript, new config) invalidate dependent stages automatically.

---

## 8. Configuration (cheat sheet)

- **More/fewer clips:** `discovery.target_clips`, `fusion.min_score`, `similarity.threshold`.
- **Clip shape:** `refine.max_duration / max_extension_seconds`, `discovery.min/max_span_seconds`.
- **Look:** `framing.default_layout` (+ `--layout` flag / UI radio), `captions.*` (font, size, placement), `fades.*` (audio stays `false` — speech never faded).
- **Speed/quality:** `export.consolidated` (single pass vs legacy chain), `transcription.device/model`, `EMBED_BATCH_SIZE`.
- **Models without code edits:** `GROQ_MODEL`, `EMBED_MODEL`, `EMBED_DIM` in `.env`.

---

## 9. Background Processes & Costs

| What | Where | Cost |
|---|---|---|
| `yt-dlp` / `ffmpeg` / WhisperX | Your machine | CPU/GPU time only |
| Groq LLM calls | `api.groq.com` | Campaign (1×), outline (1×/window), discovery (1×/window), similarity + boundaries (small) — 429-aware retries; polite delays |
| Local embeddings | Your machine | Free, no rate limits |
| PostgreSQL | `localhost:5432/viral_clips` | Every stage |
| Gradio | `localhost:7862` | Only with `ui.py` |

Groq free-tier bursts can 429 on long videos — the pipeline retries, then degrades gracefully (deterministic outline, dropped candidates, `noise` default, deterministic snap). Fewer clips under heavy rate-limiting is expected, not a crash.

---

## 10. Tests & Evidence

- `pytest` → **220/220 green** (mocked unit tests, no API/DB/FFmpeg needed).
- Real runs need DB + media: three full end-to-end runs on a 324s sample via `rerun_from_embed.py`; snapshots in `output/phase0|phase1|phase4_monster`, layout previews in `output/previews/`, transcription benchmark in `output/bench_transcription.json`.
- Not yet exercised on real footage (honest gaps): 2-person stacked layout, speaker switching + diarization, YAMNet labels, large WhisperX models, selective re-render. And nobody has eyeballed a preview here yet — machine checks (dims/duration/audio/timing) all pass, please watch one.

---

## 11. How to Run / Modify (Cheat Sheet)

```bash
pip install -r requirements.txt
# needs: postgres + vector extension, ffmpeg, node/deno (HD YouTube), .env with GROQ_API_KEYs

python main.py "https://youtu.be/XXXX" --campaign "podcast style, add logo" --template podcast_clip --layout auto
python ui.py                                   # → http://localhost:7862
python rerun_from_embed.py <video_id>          # re-run embed→render dup-safe
pytest                                         # 220 unit tests
```

Change clip style: templates or `--campaign` English. Change look: `captions.*` / `framing.*` in `config.yaml`. Change pickiness: `fusion.min_score` / `discovery.target_clips`.

---

## 12. Security Notes

- Never commit `.env`. No secrets are hardcoded — empty key slots are skipped (mock/fallback paths when none usable).
- `DB_PASSWORD` empty by default (local trust auth); set it on shared machines.
- Old hardcoded demo paths in `regen_srt.py` / `fast_burn.py` `main()` blocks only matter for direct script runs.
