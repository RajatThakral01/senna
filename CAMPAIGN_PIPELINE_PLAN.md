# Campaign Pipeline — Implementation Plan

Status: **in progress** — **all phases C1–C8 ✅ (10 Oct 2026)**; open items in `OPEN_ISSUES.md` (1.10–1.13, 5.1). Builds on the existing clip pipeline
(`AGENT_CONTEXT.md`); open quality issues of the clip pipeline are tracked in `OPEN_ISSUES.md`.

## 1. Goal

Turn the project from "long video → clips" into a **campaign tool** for clipping campaigns:

> Give it a campaign brief (requirements) plus the material you received — a long video to cut,
> or one or many finished short videos to modify — and get back campaign-compliant deliverables
> ready to post, with a checklist proving each requirement was met.

Two kinds of jobs, one system:

| Mode | Input | What happens |
|---|---|---|
| **Clip** | 1 long-form video (link or file) | existing pipeline finds + cuts clips, then the campaign's edits are applied to every clip |
| **Edit** | 1..N short videos (files, links, a folder) | no cutting — each video gets the campaign's edits (replace/add audio, logo, text, captions, reframe, trim…) |
| *(Mixed)* | both in one job | each input is routed to the right mode automatically |

## 2. Core concepts

**Campaign** — reusable definition of a client/brand campaign:
- `brief` (free text, as given by the campaign owner) + structured `requirements` parsed from it
- **asset library** for the campaign: logos, audio tracks (music / voice-over / sound replacement),
  fonts, intro/outro videos, end-card images
- platform targets (TikTok / Reels / Shorts / X …) → export presets
- output rules: aspect ratio(s), duration min/max, max file size, filename pattern, variants

**Edit recipe** — the machine-readable version of the requirements: an ordered, versioned list of
operations (§4), validated against a schema, editable in the UI before running. The brief parser
produces it; the user can correct it; every deliverable stores the exact recipe version used.

**Job** — one run of a campaign over a set of inputs. Has inputs, a mode per input, progress,
and deliverables. Runs in the background; survives restarts (resume like today's pipeline).

**Deliverable** — one output file (per input × variant × platform) + its compliance report.

## 3. Architecture

```
            ┌──────────────── Campaign (brief + assets + recipe + presets) ────────────────┐
inputs ──▶  Job ──▶ probe each input ──▶ Router ──┬─▶ CLIP MODE: existing 11-stage pipeline ─┐
(links,                                           │   (transcribe → discover → refine → frame)│
 files,                                           └─▶ EDIT MODE: per video                    │
 folder)                                              (probe → optional transcribe/reframe)   │
                                                                                              ▼
                                         Compositor: apply recipe in ONE ffmpeg graph per output
                                                                                              │
                                         QA / compliance checks ──▶ deliverables + manifest + zip
```

Key decision: **both modes end in the same Compositor.** In clip mode a clip is just a segment
cut from the long video; in edit mode the whole short video is the segment. The recipe is applied
the same way, so "add logo + replace music" works identically for clips and for supplied videos.
The current `polish.py` becomes the core of the Compositor (it already does logo + music ducking
+ captions + fades + loudness in one pass).

### Router
Per input: probe (duration, resolution, aspect, audio present, fps). Default rule:
`duration > campaign.clip_threshold (e.g. 3 min)` **and** the recipe asks for clips → clip mode;
otherwise edit mode. Always overridable per input in the UI / CLI.

## 4. Edit operations (recipe schema v1)

Must-have first (P1 = phase 3 below), nice-to-have later (P2):

| Operation | Parameters | Priority | Built on |
|---|---|---|---|
| `audio.replace` | track, start offset, loop/trim, fade, loudness target | P1 | polish audio chain |
| `audio.add_music` | track, volume, duck under speech | P1 | existing (ducking) |
| `audio.mute_original` / `audio.keep_original_volume` | level | P1 | new filter |
| `logo` | asset, position (corners/custom), size %, opacity, margin, start/end time | P1 | existing overlay (extend) |
| `captions` | on/off, style preset (karaoke / plain / bold box), font, size, colour, position | P1 | `captions.py` (+transcribe short videos) |
| `text_overlay` | text (hook, CTA, handle, disclaimer), font, style, position, start/end, animation (fade) | P1 | new (`drawtext` / ASS) |
| `reframe` | target aspect 9:16 / 1:1 / 4:5 / 16:9, method auto/face/center/fit-with-blur* | P1 | `framing.render_vertical` |
| `trim` | start/end, max duration, cut silence at ends | P1 | ffmpeg |
| `export` | platform preset (codec, bitrate, fps, max size, resolution), filename pattern | P1 | new presets |
| `intro` / `outro` | asset video/image + duration, transition | P2 | ffmpeg concat |
| `end_card` | image/text CTA for N seconds | P2 | overlay + concat |
| `speed` | factor (with pitch-correct audio) | P2 | `setpts`/`atempo` |
| `watermark` | text/handle, opacity, moving/fixed | P2 | drawtext |
| `color` | brightness/contrast/saturation preset or LUT | P2 | `eq`/`lut3d` |
| `variants` | N versions differing in hook text / music / captions style (A/B) | P2 | recipe fan-out |

\*`fit-with-blur` is only allowed when a campaign explicitly asks for it — clip mode keeps the
full-screen-crop rule.

Recipe example (what the parser would produce from *"Replace the audio with our track, add the
logo top-right at 15% width, add captions, put 'Link in bio' for the last 3 seconds, 9:16 for
TikTok and Reels, max 60 s"*):

```json
{"version": 1,
 "ops": [
  {"op": "trim", "max_duration": 60},
  {"op": "reframe", "aspect": "9:16", "method": "auto"},
  {"op": "audio.replace", "track": "assets/campaigns/acme/track.mp3", "fade_out": 1.0},
  {"op": "captions", "style": "karaoke"},
  {"op": "logo", "asset": "assets/campaigns/acme/logo.png", "position": "top-right", "size_pct": 15},
  {"op": "text_overlay", "text": "Link in bio", "position": "bottom-center", "from_end": 3}
 ],
 "export": [{"platform": "tiktok"}, {"platform": "reels"}],
 "questions": []}
```
Note: with `audio.replace` the captions come from the original speech only if the brief says so;
the parser must flag such conflicts in `questions` instead of guessing.

## 5. Brief → recipe parsing

- LLM (existing `llm_client`, JSON mode) with a strict schema + the campaign's asset list; rules
  carried over from today's parser: **only what the brief explicitly asks for**, assets only from
  the library (never invented names).
- Output also contains **`questions`** — ambiguities/conflicts the user must answer (e.g. "replace
  audio — keep the speaker's voice underneath?", "logo size not specified, using 12%").
- Deterministic validation + defaults; keyword fallback without an LLM (like today).
- The user reviews/edits the recipe in the UI (form view + JSON view) before the job runs; the
  approved recipe is versioned and stored with the campaign.

## 6. QA / compliance

Every deliverable is checked against the recipe, automatically:
- duration within limits, resolution / aspect / fps / codec / file size per platform preset
- audio present, loudness ≈ target, original audio actually replaced/muted when required
- logo present (template match of the logo in sampled frames), text overlays present (frames at
  their times), captions present when required
- result: a per-deliverable checklist (✅/❌ with the measured value) in the UI and `manifest.json`;
  failed checks block "approve" until the user overrides.

## 7. Data model (migration v7)

| Table | Purpose |
|---|---|
| `campaigns` | id, name, brief, requirements jsonb, recipe jsonb + version, presets, created/updated |
| `campaign_assets` | id, campaign_id, kind (logo/audio/font/video/image), path, metadata (duration, size) |
| `jobs` | id, campaign_id, status, mode_policy, progress, created/started/finished, error |
| `job_inputs` | id, job_id, source (url/path), probe jsonb, mode (clip/edit), video_id (clip mode → existing `videos` row), status |
| `deliverables` | id, job_id, job_input_id, clip_id (clip mode), variant, platform, path, recipe_version, qa jsonb, status (rendered/approved/rejected) |

Files: `assets/campaigns/<campaign>/…` (uploaded assets), `output/<campaign>/<job>/<deliverable>.mp4`
+ `manifest.json` + a zip for download. Clip-mode intermediates keep using the per-video
workspaces that exist today.

## 8. UI (Gradio, campaign-first)

Recommendation: stay on **Gradio** for now (fast to build, already used); keep the backend behind a
clean service layer (`campaigns/`, `jobs/`) so a FastAPI + web frontend can replace it later
without touching the pipeline.

| Tab | Contents |
|---|---|
| **1. Campaigns** | list + create/edit: name, brief text, upload assets (logos, audio, fonts, intro/outro), platform targets, output rules → "Parse brief" → recipe shown as an editable form + JSON, with the parser's **questions** to answer → save |
| **2. New job** | pick campaign → add inputs (paste links, upload one or many files, or a folder path) → each input shows its probe (duration, aspect) and auto-chosen mode (clip/edit, overridable) → for clip mode: number of clips / clip length → **Run** |
| **3. Jobs** | queue with status/progress per input, live log tail, cancel / resume |
| **4. Review & deliver** | grid of deliverables with player, compliance checklist, recipe tweaks per deliverable (e.g. move the logo) → re-render just that one → approve → **download zip** (+ captions/titles/hashtags text for posting) |
| **5. Clip review** | today's Review & Correct tab (boundaries, layout, captions) for clip-mode outputs |

CLI equivalents: `main.py campaign create --brief … --assets …`, `main.py job run --campaign X
--input a.mp4 --input https://…`, `main.py job status`.

## 9. Implementation phases

| Phase | Deliverable | Depends on | Effort |
|---|---|---|---|
| **C1** Data model ✅ | migration v7, repositories, `campaigns/` service (create/list/assets), asset storage + probing | — | ½ day |
| **C2** Recipe + parser ✅ | recipe schema v1 (pydantic-style validation), LLM brief parser with asset matching + `questions`, keyword fallback, tests | C1 | 1 day |
| **C3** Compositor ✅ | generalise `polish.py` into `pipeline/compositor.py`: P1 ops (audio replace/add/mute, logo with size/opacity/timing, captions, text overlays, reframe hook, trim, export presets) as one ffmpeg graph; golden-file tests per op | C2 | 2–3 days |
| **C4** Edit mode ✅ | per short video: probe → (transcribe if captions) → (reframe if aspect differs) → compositor → deliverable; batches of N inputs | C3 | 1 day |
| **C5** Router + jobs ✅ | input classification, clip mode = existing pipeline + compositor on each clip, background job runner (one worker, persisted progress, resume), CLI | C4 | 1–1½ days |
| **C6** QA + delivery ✅ | compliance checks (§6), `manifest.json`, zip, posting text (title/hashtags from existing clip metadata) | C5 | 1 day |
| **C7** UI ✅ | the 5 tabs (§8) on the service layer | C6 | 2 days |
| **C8** P2 ops + hardening ✅ | intro/outro, end card, speed, watermark, colour, variants; docs, real campaign trial runs | C7 | 2 days |

Total ≈ 11–13 working days. Each phase ends with tests + a real run where it applies, like the
framing phases.

## 10. Decisions needed before starting

1. **Must-have operations** — is the P1 list in §4 right for your first campaigns? Anything missing
   (e.g. green-screen, subtitles in another language, removing original captions/watermarks)?
2. **Platforms** — which ones, and do you need one output per platform or one universal 9:16 file?
3. **Original speech when replacing audio** — usually removed, or kept under the new track?
4. **Text overlays** — what kinds do campaigns ask for (hooks, CTA, @handle, disclaimers)? Any brand fonts?
5. **Variants** — do campaigns pay per video so you'd want several variants of each input?
6. **UI** — Gradio first (recommended) or a proper web app from the start?

## 11. Progress log

**Defaults used (§10 unanswered):** P1 op list as in §4; platforms TikTok / Reels / Shorts (+ generic,
YouTube, X, Facebook presets), one universal 9:16 file unless the brief names platforms; replacing
audio removes the original voice by default and the parser asks to confirm; generic text roles
(hook / cta / handle / disclaimer), Arial unless a campaign font is uploaded; variants in C8; Gradio UI.

**C1 — data model ✅** `db/migration_v7.sql` (campaigns, campaign_assets, jobs, job_inputs,
deliverables; applied by `db/init_db.sh`), `db/repositories/campaign_repo.py` (campaigns, recipe
versioning, assets), `db/repositories/job_repo.py` (jobs, inputs, deliverables),
`pipeline/media.py` (probe any file: kind, duration, size, audio/video, alpha, aspect label),
`campaign/service.py` (create/list/update campaigns with unique slugs; asset library copied to
`assets/campaigns/<slug>/<kind>/`, kind inferred — a transparent PNG is a logo; catalog for the
parser). Tests: `test_campaigns.py` (11, real ffmpeg media + DB).

**C2 — recipe + brief parser ✅** `campaign/recipe.py` (schema v1: 15 ops — 9 rendered from C3,
6 stored for C8; validation with clamping, enums, aliases, asset resolution against the catalog,
single-op de-dup, conflicts, platform limit warnings, filename guard, questions; plain-English
`describe_recipe`), `campaign/presets.py` (platform presets, aspect → output size; limits are
conservative defaults), `campaign/brief_parser.py` (LLM via `llm_client`, JSON mode, temperature 0,
strict rules + keyword fallback). Live checks with the Groq keys: the plan's example brief → exactly
the 6 expected ops; a clip brief → 8 clips 30–60s with focus, uppercase captions, logo, timed hook,
Shorts; a brief needing files that don't exist → errors + upload questions. Fixed during testing:
the LLM filling unstated settings and asking about source videos / filenames (prompt rules 9–10),
and the validator silently substituting the only audio file for a missing "voiceover" (auto-pick
now only for the keyword fallback). Tests: `test_recipe.py` (28). Suite: 339 passed.

**C3 — compositor ✅** `pipeline/compositor.py`: `compose(input, recipe, output, platform)` renders one
deliverable in a single ffmpeg pass (after an optional framing pass); `compose_all` renders every
export platform, reusing one render for platforms with the same aspect. Implemented ops: trim
(cut snapped to the end of the last spoken word when a transcript exists, never mid-word),
reframe (wide → 9:16 auto/face = the face-tracking virtual camera; other aspect changes = static
crop centred on the main face, or centre; fit_blur), audio.replace (offset, loop, volume, fades,
original kept at keep_original), audio.add_music (ducked under speech via sidechaincompress),
audio.mute_original, audio.volume, logo (size %, opacity, margin, 9 positions, start/end/from_end),
captions (karaoke / plain / boxed, font, size, colours, position, uppercase; words from a supplied
transcript or WhisperX on the delivered segment), text_overlay (role sizes, 9 positions, box,
fade/pop, timing; lifted above bottom captions). Captions + text overlays share one ASS script
sized to the output (libass: brand fonts via `fontsdir`, boxes, fades); loudness normalised to the
preset (−14 LUFS), silent track added when there is no audio, H.264/AAC 48 kHz, faststart.
C8 ops are reported in `skipped`. Verified: every P1 op at once on a synthetic clip (1.6 s render,
−13.7 LUFS); 25 s of real footage → 9:16 via the virtual camera with real Whisper captions and
1:1 via the face-centred crop; an uploaded font not installed on the system rendered from the
campaign's font file. Tests: `test_compositor.py` (18, real renders; audio content checked by
FFT — replaced track present, original gone). Suite: 357 passed.

**C4 — edit mode ✅** `campaign/edit_mode.py`: `render_input(path, recipe, out_dir, work_dir,
basename)` (DB-free core) and `process_input(job, campaign, input, label)` (resolves the source via
`input_handler` — local file or URL — probes it, records `job_inputs.local_path/probe/status`,
renders, writes one `deliverables` row per platform with the compose report as `qa`; a re-run
replaces the input's rows; failures mark the input failed and re-raise for the job runner to
isolate). Captions: the whole input is transcribed once (cached `transcript.json` in the work dir)
and shared by every platform render; Whisper models now stay loaded across a batch
(`transcribe_audio(keep_models=True)` + `release_models()`; the long-video pipeline is unchanged).
Naming: `output.filename` pattern via `output_basename()`; `input_labels()` gives unique stems
(`promo`, `promo_2`); `compose_all` fills `{platform}`. Layout: `output/campaigns/<slug>/<job8>/`,
work in `work/campaigns/<slug>/<job8>/<idx>/`. Compositor fixes found by the real run (20 s 16:9
and 15 s 9:16 samples cut from real footage, logo + karaoke captions + CTA, TikTok + YouTube):
(1) a vertical source exported to 16:9 was cropped to a blurry 32 % strip → `auto` now falls back
to fit-over-blur whenever a crop would keep < 50 % of the picture (with a warning); (2) a bottom CTA
touched the captions on 16:9 → bottom overlays are placed from the measured caption block height,
and the bottom safe zone depends on the aspect (22 % vertical, 12 % square, 7 % wide); (3) when
`audio.replace` removes the original voice, captions now transcribe the replacement track (what the
viewer hears), also for silent inputs. Second input transcribed in 3 s with the cached model.
Tests: `test_edit_mode.py` (10, incl. DB round trip + re-run) and 3 new compositor tests.
Suite: 369 passed.

**C5 — router + jobs + clip mode ✅** `campaign/jobs.py`: `create_job(campaign, sources,
mode_policy, modes)` (folders expand to their videos; the campaign recipe is re-validated against
today's assets and frozen on the job with its version), `run_job(job_id, progress, cancel_event)`
(per input: resolve/download → probe → route → clip or edit mode → QA; failures isolated on the
input, the rest carry on; a re-run skips done inputs; status done / failed / cancelled with the
failures listed), `route()` (per-input override > job policy > recipe.mode > auto: longer than
180 s → clip mode), one background `Worker` thread (submit / cancel between inputs) and
`resume_interrupted()` (UI start-up). `campaign/clip_mode.py`: `run_pipeline(render=False)` with
the recipe's clip spec as temporary config overrides (`discovery.target_clips` = count + 2,
span + refine min/max) and `clips.focus` as the analysis focus — `run_pipeline` gained
`campaign_config` (skips the old brief parser), `render` and `info`; cached analysis is reused
when the spec matches; the top `count` refined clips are framed by `render_clips(clip_numbers=…)`
with no captions/logo/music into the job's work dir (the standalone `output/<video>/` finals are
untouched and the render stage is invalidated afterwards); then per clip, 9:16 platforms compose
from the face-tracked vertical render and other aspects from a full-resolution cut of the
source ranges (`cut_ranges`), captions from the source transcript mapped onto the clip timeline
(`clip_segments`, stitched ranges supported). CLI: `python main.py campaign create|list|show|brief`
and `python main.py job run|resume|status` (`campaign/cli.py`). Bug found by the real run: a
finished standalone render was "reused" (the old finals with their own captions) → the render
stage is now reset before the campaign render and only finals inside the job folder are
accepted. Real run (cached MrBeast analysis, LLM-parsed brief "2 clips under 60 s, logo top
right, karaoke captions, Link in bio for the last 3 s, TikTok + YouTube"): 2 clips × 2
platforms in 7.2 min, all QA checks pass. Tests: `test_jobs.py` (14).

**C6 — QA + delivery ✅** `campaign/qa.py` measures each FILE: duration (vs plan, platform
limit, trim / clip-spec min-max), resolution + H.264/AAC, file size, audio present + integrated
loudness (ebur128, ±2 LU of the preset), audio actually replaced (log-band spectrum + envelope
correlation against the replacement track, compared with the original), logo pixels present at
the exact spot the compositor puts them (±6 px search, opacity-aware), captions / text overlays
drawn, C8 ops listed as not rendered. Each check is shown failing on a file that breaks it.
Results are stored on the deliverable (`qa.compliance`, `qa.checks`). `campaign/delivery.py`:
`manifest.json` + `posting.txt` per job (title = clip title/hook, the campaign's CTA/handle
texts, hashtags from the clip + #tags in the brief), `build_zip(approved_only)`,
`set_status(approved/rejected)`, `rerender(deliverable, recipe)` (one file again, optionally
with its own tweaked recipe, from the same composed input; QA re-runs). Tests: `test_qa.py` (8).

**C7 — UI ✅** `campaign_ui.py` (handlers are plain functions; wiring in
`build_campaign_tabs()`), mounted first in `ui.py`: **1 Campaigns** (pick/create, brief,
platforms, asset upload/remove, Parse brief → plain-English recipe + questions, editable JSON,
Save recipe), **2 New job** (links / paths / folders / uploads, Check inputs → duration, aspect,
auto mode + reason, per-input override column, Run → queued on the worker), **3 Jobs** (queue,
auto-refresh every 4 s, inputs + errors, cancel, resume/retry), **4 Review & deliver**
(deliverables with QA, preview, compliance checklist, posting text, approve / reject, recipe
JSON for one file → re-render, zip of approved or all). The old Generate tab is **5 Quick clips
(no campaign)** and Review & Correct is **6 Clip review**. Start-up re-queues interrupted jobs.
The keyword parser now recognises more CTA verbs (shop, buy, order, download, sign up, …).
Verified: handler flow test (create → … → zip) and the real app served all 6 tabs. Tests:
`test_campaign_ui.py` (3). Suite: 394 passed.

**C8 — P2 ops, variants, hardening ✅** Compositor: `speed` (pre-pass: setpts + atempo, pitch
kept; transcript words retimed so captions stay in sync), `color` (presets vivid / warm / cool /
bw + brightness / contrast / saturation via `eq`), `watermark` (translucent text in the shared
ASS script), `intro` / `outro` (video or image, fitted over a blurred fill, loudness-normalised,
optional fade into / out of the body) and `end_card` (dark card + optional image + text) joined
in a post-pass; the report gains `body_offset` / `body_duration` so QA checks the logo, the
replacement audio and the trim limits on the body, and QA fails a missing intro / outro / end
card. **Variants:** `recipe.variants = [{name, ops, remove}]` — each op replaces the base op of
the same kind (text overlays by role), `remove` drops ops; edit and clip mode render main + every
variant (transcribed once), files get `_<variant>` before the platform (or the `{variant}`
token), deliverables carry the variant, QA / posting text / re-render use the variant's recipe;
two variants that only ADD ops → the first becomes the main version ("2 versions with hook A /
hook B" = 2 files, not 3). Brief parser hardening from live runs: the prompt knows variants and
C8 ops; deterministic guards on the LLM output — duration limits and platforms the LLM dropped
are copied back from the brief (keyword parser), on-screen text that is not in the brief is
dropped (the LLM once invented "HOOK_TEXT"), a "custom" role is corrected from the brief's
wording, and an LLM-made filename pattern is ignored unless the brief asks for one. QA logo check
is alpha-aware (semi-transparent logos). Bugs found by the real trial (2 real-footage samples ×
2 hook versions; intro card, warm, x1.1, logo, captions, watermark, end card, TikTok, under 20 s):
intro / end-card pieces were written to a relative path while ffmpeg ran inside the work dir
(tests used absolute paths) → absolute paths + a relative-work-dir test; the LLM's filename
pattern produced `…_tiktok_tiktok.mp4`. After the fixes: 4 files, durations exact
(15 s ÷ 1.1 + 1.5 s intro + 2 s end card = 17.2 s), every QA check passes. Tests: C8 cases in
`test_compositor.py`, `test_recipe.py`, `test_edit_mode.py`. Suite: 403 passed.
