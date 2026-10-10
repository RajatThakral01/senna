# Viral Clips Automator — End-to-End Explanation

> This document explains the project from zero: what it does in simple words, then every technical piece, how files connect, what happens in the background, and what tech it uses. (Last synced 10 Oct 2026 — 11-stage pipeline, Groq LLM + local embeddings, per-video workspaces + resume, OpusClip-style framing, campaign pipeline (§13). Full technical detail: `AGENT_CONTEXT.md`; known problems: `OPEN_ISSUES.md`.)

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

It gives you (one folder per video, e.g. `output/oBXSvS2QKxU_6f9ddaec/`):
- `clip_1_final.mp4`, `clip_2_final.mp4`, ... — ready-to-post vertical videos
- `report.json` — hook, refined timestamps, source ranges, layout + reasons, titles

You can run it two ways:
- **Command line:** `python main.py <video> --campaign "..." --template podcast_clip --layout auto`
- **Web UI:** `python ui.py` → `http://localhost:7862` (Generate + Review & Correct tabs)

Latest verified examples (real MrBeast videos): a 58-minute video → 14 clips, and a 27-minute video → 13 clips in about an hour on an 8 GB Mac. The system aims for up to 15 (`discovery.target_clips`) but deliberately returns fewer when quality is insufficient — the quality floor, duplicate removal and "no payoff → drop it" explain that, by design.

**Same link twice?** The second run *resumes*: finished steps are skipped and the download is reused (a finished video "re-runs" in about 2 seconds). Use `--fresh` to start over or `--from-stage refine` (etc.) to redo one step onward.

---

## 2. The 11-Step Pipeline (High Level)

Every run goes through the same stages in `main.py:run_pipeline()`:

```
1.  Normalize input      input/input_handler.py      → downloads/<youtube-id>.mp4 (cached) or your local file
2.  Parse campaign        campaign/campaign_parser.py → JSON config (logo? music? layout? ...)
3.  Transcribe            pipeline/transcriber.py     → work/<video>/transcript.json (words + timestamps)
4.  Chunk                 pipeline/chunker.py         → topic blocks, saved to DB
5.  Audio events          pipeline/audio_events.py    → applause/laughter spikes → extra candidates
6.  Embed                 pipeline/embedder.py        → each chunk → 1024 numbers, saved to DB
7.  Outline               pipeline/outline.py         → whole-video structure (topics, Q&A, stories)
8.  Analyze               pipeline/analyzer.py        → outline + events → ranked clip picks (fusion)
9.  Similarity            pipeline/similarity.py      → continuations? stitch / standalone / noise
10. Refine boundaries      pipeline/boundaries.py      → snap to complete sentences, LLM-checked
11. Render                 clipper + framing + captions + editplan + polish
                         → cut → virtual-camera vertical → ASS subtitles → logo/music/fades → output/<video>/
```

Each stage is **checkpointed in the database** (`pipeline_runs` table + fingerprints). If the program crashes at step 9, re-running the same link skips finished steps. `--from-stage <step>` redoes one step and everything after it; `rerun_from_embed.py <video_id>` does the same for a video id.

### Where your files go

```
downloads/<youtube-id>.mp4               the source video (kept, reused next time)
work/<video>/                            audio, transcript, outline, clips/ (subtitles, edit plans, debug videos)
output/<video>/clip_N_final.mp4          the finished clips + report.json
logs/runs/<date-time>_<source>.log       a log file for every run
```

`<video>` is the source name plus a short id, e.g. `oBXSvS2QKxU_6f9ddaec`. Each video has its own folders, so two videos never mix (before this, a second video could pick up the first video's audio!).

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
                    │ download once│      │ Groq LLM turns   │
                    │ → downloads/ │      │ English → JSON   │
                    │ (cached)     │      └────────┬─────────┘
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
1. **Local AI:** WhisperX (speech→text), Qwen3 embeddings (text→numbers) and the face / speaker / saliency detection for framing run on your machine. No internet needed after model download, no embedding bills.
2. **Cloud AI (Groq):** `openai/gpt-oss-120b` for understanding text — campaign parsing, finding clip candidates, checking boundaries, classifying continuations. Needs `GROQ_API_KEY` (+ `_2`, `_3`, `_FALLBACK` — stages are split across keys with automatic fallback).

One database (PostgreSQL, in Docker) glues everything together. One video tool (FFmpeg, plus OpenCV for faces, speakers and saliency) does all video work.

---

## 4. Tech Stack

| Layer | Technology | Why it exists |
|---|---|---|
| Language | **Python 3.10+** | Whole project |
| Video download | **yt-dlp** (+ Node for HD), `gdown`, `requests` | YouTube (H.264 ≤1080p, HD required) / Drive / direct URL; downloads cached |
| Speech-to-text | **WhisperX** `small` + `librosa` | Word-level timestamps (`start`, `end` per word) |
| LLM (brain) | **Groq** `openai/gpt-oss-120b` via OpenAI-compatible API | Campaigns, candidates, boundary checks, stitch decisions |
| Embeddings | **Local** `Qwen3-Embedding-0.6B`, 1024-dim | “Similar meaning” = “close numbers”, no API needed |
| Database | **PostgreSQL + pgvector** via `psycopg2-binary`, `pgvector` lib | Videos, chunks, clips, candidates, plans, progress; HNSW vector search |
| Video editing | **FFmpeg/FFprobe** (`libx264`, `fast`, `crf 23`, `aac`), **OpenCV** (YuNet faces, saliency) | Cut, track faces/speakers, 1080×1920 virtual camera, ASS subtitles, logo, ducked music, fades |
| Config | **YAML** (`config.yaml`) + `python-dotenv` + `config.py` | All tunables in one place, secrets in `.env` |
| Web UI | **Gradio** (`ui.py`, port 7862) | Generate + Review & Correct tabs |
| Tests | **pytest** (300/300 green, LLM mocked) | Fast tests; repo tests use the Docker database |
| Database server | **Docker** `pgvector/pgvector:pg16` on port 5433 | One command to run Postgres + pgvector |

---

## 5. File Structure — What Each File Does

```
viral-clips/
├── main.py                  # THE BOSS. run_pipeline() runs all 11 stages. CLI entry point.
├── ui.py                    # Gradio page: generate clips, then Review & Correct single clips.
├── rerun_from_embed.py      # Re-run an existing video from a chosen step (dup-safe).
├── tools/framing_audit.py   # Measures how well the clips are framed (wrong person, sliced faces...).
├── preview_framing.py / demo_faces.py / bench_transcribe.py   # Verification scripts only.
├── config.yaml              # ALL settings (chunking, similarity, outline, discovery, audio,
│                            # fusion, captions, refine, framing, export, ...).
├── config.py                # Reads config.yaml + .env. Groq key/model, embedding settings.
│
├── input/input_handler.py   # detect_source_type() + handle_input() → cached download / local file
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
│   ├── framing.py           # the "virtual cameraman": faces, scene cuts, camera moves, layouts.
│   ├── active_speaker.py    # who is talking? (lip movement + audio) → which shot to show.
│   ├── workspace.py         # gives every video its own work/ and output/ folders.
│   ├── llm_client.py        # talks to Groq: 4 keys, waits out rate limits, strict JSON.
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
├── downloads/               # cached source videos
├── work/<video>/            # audio, transcript, outline, per-clip subtitles/plans/debug videos
├── output/<video>/          # Finals: clip_N_final.mp4 + report.json
└── logs/                    # run logs
```

---

## 6. End-to-End Data Flow (What Happens in the Background)

### Stages 1–2 — Input + campaign
- Source detection → download once to `downloads/<id>.mp4` (or use your local file as-is); the video row is registered in the DB — or the previous row for the same link is resumed — and the video's workspace folders are created.
- Campaign LLM reads your English + available logos/music → strict JSON (logo? music? subtitles? **layout**?). Template merged in; no key → keyword mock so the app still runs.

### Stage 3 — Transcription
1. `ffmpeg` extracts the audio to `work/<video>/audio.wav`.
2. WhisperX `small` transcribes (int8 on CPU; CUDA if available) + aligns to per-word timestamps. `small` was chosen after `base` misheard "calories" as "galleries"; it takes ~9 min per hour of video on this Mac.
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
3. **Fusion:** merges near-duplicates, scores each on 7 editorial components (audio evidence is only a 5% bonus), picks a diverse set up to 15, enforcing a 0.45 quality floor — weak ones are rejected *with written reasons*. Winners become `clips` rows.

### Stage 9 — Similarity
For each clip: vector-search similar chunks → LLM judges each as `stitch` (direct continuation → concatenated), `standalone` (same topic, separate — saved, not joined), or `noise` (rejected). Safety caps: no join across >45s gaps, no output over 90s (added after a 120s “Frankenstein” stitch was caught live).

### Stage 10 — Boundary refinement
Snaps every clip edge to sentence starts/ends (±0.15s/0.3s padding inside neighbouring silence), then the LLM picks the best start and end sentence IDs from ±20s of context and says whether *its chosen range* finishes the thought (timestamps always come from the transcript, never the LLM). If the punchline comes later, a second LLM pass looks up to 30s further ahead and extends the clip to where it lands. If the story never resolves, the clip is dropped with a written reason — a clip without its payoff is worse than no clip. Stitched parts are refined separately and the whole clip stays ≤90s. `--no-refine` skips this.

### Stage 11 — Render (per clip)
1. **Cut/stitch:** refined ranges cut via FFmpeg (multi-part → concat).
2. **Vertical:** hard requirement — every shot is a 9:16 crop filling 1080×1920, never letterboxed or stretched (see "The virtual cameraman" below). Manual 9:16 crops from the Review tab always win. Audio carried over.
3. **Subtitles:** words → gap-free timing → phrase cues (≤8 words, 2 lines) → yellow karaoke ASS burn in one pass (top placement when faces sit low).
4. **Edit plan:** structured plan saved versioned (DB + `clip_N_plan.json`) — preview and export use the same plan.
5. **Polish:** ONE ffmpeg pass — subtitles, fades (video only; speech never faded), logo, ducked background music, loudness normalization, limiter.
6. Final saved, DB updated, `output/report.json` regenerated with refined times, ranges, layouts, and reasons.

### The virtual cameraman (how the framing decides, in simple words)

Think of a camera operator who has already watched the clip and now decides, shot by shot, where to point a tall 9:16 frame:

1. **Find the faces and the cuts.** Every 0.1s it finds faces (YuNet) and remembers who is who, even if someone looks away for a second. It also notices every hard cut in the source video, so each shot is planned separately.
2. **Who is talking?** It watches mouths move (compared with the eyes, so nodding doesn't count) and checks the audio is actually speech. The clearest talker gets the shot; when the speaker changes it **cuts** (on a pause, never mid-word), but never makes a shot shorter than 1.6s. If two people talk over each other, it shows both in a **split-screen**.
3. **No clear talker?** It frames the most important-looking person (big face, near the middle). In a **crowd**, it frames the group or the action in the centre instead of guessing one face. If there are **no faces** (b-roll, products, someone turned away), it follows what's moving and eye-catching.
4. **Move like a real camera.** While the person stays roughly in place the frame doesn't move at all (locked shot). When they walk, the camera eases into the move and eases out — planned for the whole shot at once, so there's no jitter or "lag". It avoids cutting a neighbour's face in half at the edge.
5. **Polish the shot.** Close-ups are zoomed so the face fills about a third of the width; on emphatic loud moments it does a subtle 6% punch-in.

Want to see its decisions? Run with `--debug-framing` and open `work/<video>/clips/clip_N_vertical_debug.mp4`: the source video with every face, mouth-activity bar and the chosen crop drawn on it, next to the final vertical clip.

### Review & Correct (UI tab)
Ranked candidates + per-clip detail → edit boundaries (validated), layout, or caption words (timings preserved) → each edit bumps the plan version → re-render just that clip.

---

### Campaign tabs
See §13 — Campaigns → New job → Jobs → Review & deliver sit in front of the two tabs above.

---

## 7. Database — Tables & Why

PostgreSQL + `CREATE EXTENSION vector;`. Pool of 1–10 connections; every repo does `get_conn() → cursor → commit → release_conn()`. Apply `schema.sql` then migrations v4→v5→v6→v7 (`db/init_db.sh` does all of it).

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
| `campaigns` / `campaign_assets` | Campaign brief, saved recipe (versioned), platforms; its logo/audio/font/video/image files (migration v7) |
| `jobs` / `job_inputs` | One run of a campaign over a set of inputs (frozen recipe); each input's mode, probe, status, error |
| `deliverables` | One output file per input × clip × variant × platform, with its QA report and approve/reject status |

Resume logic: finished stages are skipped; changed inputs (new transcript, new config) invalidate dependent stages automatically.

---

## 8. Configuration (cheat sheet)

- **More/fewer clips:** `discovery.target_clips`, `fusion.min_score`, `similarity.threshold`.
- **Clip shape:** `refine.max_duration / max_payoff_extension_seconds / reject_incomplete`, `discovery.min/max_span_seconds`.
- **Look:** `framing.default_layout` (+ `--layout` flag / UI radio), `captions.*` (font, size, placement), `fades.*` (audio stays `false` — speech never faded).
- **Camera feel:** `framing.path_smooth_seconds` (higher = calmer), `deadzone_ratio` (bigger = more locked shots), `speaker_switch_cost` / `min_shot_seconds` (fewer/more speaker cuts), `auto_zoom`, `emphasis_zoom` (0 = no punch-ins).
- **Speed/quality:** `transcription.model` (`small` default), `export.consolidated`, `export.keep_intermediates`, `framing.debug_video`, `YT_MAX_HEIGHT` (download resolution cap).
- **Models without code edits:** `GROQ_MODEL`, `EMBED_MODEL`, `EMBED_DIM` in `.env`.

---

## 9. Background Processes & Costs

| What | Where | Cost |
|---|---|---|
| `yt-dlp` / `ffmpeg` / WhisperX | Your machine | CPU/GPU time only |
| Groq LLM calls | `api.groq.com` | Campaign (1×), outline (1×/window), discovery (1×/window), similarity + boundaries (small) — 429-aware retries; polite delays |
| Local embeddings | Your machine | Free, no rate limits |
| PostgreSQL | Docker `viral-clips-db`, `localhost:5433/viral_clips` | Every stage |
| Gradio | `localhost:7862` | Only with `ui.py` |

Groq's free tier allows ~8,000 tokens per minute per key (~1 big call per key per minute). When all 4 keys are busy the pipeline **waits** for the per-minute budget to reset (log lines "all 4 keys rate-limited; waiting 3s…" are normal) instead of failing. A 27-minute video's LLM stages take ~5 minutes this way.

**Typical timing (27-minute video, this Mac):** download 1 min, transcription 4.5 min, LLM stages ~5 min, rendering ~50 min for 13 clips → ~64 min total.

---

## 10. Tests & Evidence

- `.venv/bin/python -m pytest -q` → **403 green** (~3.5 min; LLM/network mocked; repository tests use the Docker DB; campaign tests render real video).
- Campaign runs: edit mode on real-footage samples (logo, captions, CTA, intro card, colour, speed, watermark, end card, 2 hook versions) and clip mode on the MrBeast video (2 clips × TikTok + YouTube) — every QA check passing.
- Real runs: two MrBeast videos end to end (58 min → 14 clips, 27 min → 13 clips), reviewed by eye and with `tools/framing_audit.py`. The second video's clips looked clearly better than the first; remaining weak spots (action that isn't a face, sliced faces, graphics) are listed in `OPEN_ISSUES.md`.
- Not yet exercised: clicking through the Gradio UI in a browser (handlers are tested), the CI test job, diarization, YAMNet labels.

---

## 11. How to Run / Modify (Cheat Sheet)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# needs: Docker DB (README step 4), ffmpeg, node (HD YouTube), .env with GROQ_API_KEYs + DB_PORT=5433

.venv/bin/python main.py "https://youtu.be/XXXX" --debug-framing   # full run (resumes if run before)
.venv/bin/python main.py "https://youtu.be/XXXX" --from-stage render   # redo just the rendering
.venv/bin/python ui.py                                    # → http://localhost:7862
.venv/bin/python tools/framing_audit.py <video_id>        # how good is the framing?
.venv/bin/python main.py campaign create --name "Acme" --asset logo.png --brief "…"   # campaign
.venv/bin/python main.py job run --campaign acme --input clip1.mp4 --input folder/     # run a job
.venv/bin/python -m pytest -q                             # 403 tests
```

Change clip style: templates or `--campaign` English. Change look: `captions.*` / `framing.*` in `config.yaml`. Change pickiness: `fusion.min_score` / `discovery.target_clips`.

---

## 12. Security Notes

- Never commit `.env`. No secrets are hardcoded — empty key slots are skipped (mock/fallback paths when none usable).
- The Docker DB uses `postgres`/`postgres` on localhost only; change it on shared machines.
- Old hardcoded demo paths in `regen_srt.py` / `fast_burn.py` `main()` blocks only matter for direct script runs.

---

## 13. Campaign Pipeline (clipping campaigns)

**In simple words:** a client gives you a *brief* ("replace the music with ours, logo top right, captions, 'Link in bio' at the end, TikTok, under 60 s") plus either a **long video to cut into clips** or **short videos to modify**. You create a campaign once, upload its files, and the tool turns the brief into an **edit recipe** you can read and correct. Then every job applies that recipe to whatever videos you feed it, checks each result, and hands you a zip.

**Flow:**
1. **Campaign** (`campaign/service.py`) — name, brief, platforms, asset library copied to `assets/campaigns/<slug>/`.
2. **Brief → recipe** (`campaign/brief_parser.py` → `campaign/recipe.py`) — the LLM (keyword fallback) writes a recipe; the validator only allows real asset names, fills defaults, clamps numbers, and turns anything unclear into a **question**. Deterministic guards fix common LLM slips (dropped durations / platforms, invented text, made-up file names). The recipe is shown in plain English and saved with a version number.
3. **Job** (`campaign/jobs.py`) — inputs (links, files, folders) are probed and **routed**: longer than 3 min → **clip mode**, else **edit mode** (overridable). One background worker; failures stay on their input; re-runs resume.
4. **Edit mode** (`campaign/edit_mode.py`) — the whole short video is the deliverable; transcribed once if captions are on.
5. **Clip mode** (`campaign/clip_mode.py`) — the existing 11-stage pipeline finds the clips (clip count / length / focus from the recipe), the virtual camera frames them, then the recipe is applied.
6. **Compositor** (`pipeline/compositor.py`) — both modes end here: trim, reframe, audio replace / music (ducked) / mute, logo, captions, text overlays, watermark, colour, speed, intro / outro / end card, one file per platform preset (TikTok, Reels, Shorts, YouTube, X, Facebook), loudness-normalised. **Variants** (e.g. two hooks) render side by side.
7. **QA** (`campaign/qa.py`) — measures every file: duration, size, codecs, loudness, audio really replaced, logo visible where it should be, text drawn, intro/outro present.
8. **Delivery** (`campaign/delivery.py`) — `manifest.json`, `posting.txt` (title + CTA + hashtags), approve / reject, re-render one file with a tweaked recipe, zip.

**UI:** tabs 1–4 of `ui.py` (`campaign_ui.py`). **CLI:** `python main.py campaign …` / `python main.py job …`. **Output:** `output/campaigns/<slug>/<job8>/`. Design + build log: `CAMPAIGN_PIPELINE_PLAN.md`.

