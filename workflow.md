# Workflow — using the UI and editing videos

How to run campaigns, cut clips from long videos, edit short videos, and fix
or re-render results in the Viral Clips Automator UI.

---

## 0. Start the app

```bash
cd ~/Desktop/senna-dev
docker compose up -d            # add --build after code or requirements changes
```

Open **http://localhost:7862** (this Mac only, while the containers run).

- Watch what it's doing: `docker logs -f senna-dev-app-1`
- Stop: `docker compose down` (database and outputs are kept)

LLM: every LLM call goes to **Claude** (Sonnet 5.5 for clip discovery, Haiku 5.5
for everything else). If Claude is down, the same call goes to **Groq**
automatically. You don't need to do anything. A log line containing
`falling back to Groq` tells you it happened.

---

## 1. The big picture

```
1 · Set up campaign → 2 · Start a job → 3 · Progress → 4 · Review & download
 (brief + files +      (videos in)        (live view)     (approve, tweak,
  edit steps)                                               re-render, zip)
```

A **status banner above every tab** always shows what is running, its
percentage, its current step and how long ago it last did something. If it
says "No jobs running", nothing is running.

- A **campaign** = one client/brand: its brief, its assets (logo, music,
  fonts…), target platforms, and a **recipe** (the brief turned into exact
  edit steps).
- A **job** = a batch of videos run through one campaign's recipe.
- Each input runs in one of two **modes**:
  - **clip**: a long video (over 3 minutes) is cut into the best short clips.
  - **edit**: a short video (3 minutes or less) is edited as a whole (logo,
    music, captions, text, reframe…).
  - `auto` picks by length. You can force either one.

The last two tabs, **Quick clips** and **Fix a clip (advanced)**, are the
older single-video tools: quick clips without a campaign, and clip-level
corrections.

---

## 2. Tab 1 · Set up campaign

1. **Create**: type a **Name**, paste the client's **Brief** exactly as
   written, tick **Platforms** (tiktok, reels, shorts, youtube, x, facebook,
   generic), then click **Create as new campaign**.
   To reopen one later, pick it in the **Campaign** dropdown (↻ refreshes the
   list).
2. **Add assets**: upload logos, audio tracks, fonts, intro/outro videos and
   images under **Asset library**, leave **Kind = auto** (or set it), then
   click **Add assets**. To remove one, type its exact name and click
   **Remove**.
   Upload the assets *before* parsing, because the recipe can only use assets
   that are in the library. It never invents file names.
3. **Parse brief**: keep **Use the LLM** ticked (Claude reads the brief; untick
   it for the simple keyword parser), then click **Parse brief**. You get:
   - a plain-English summary of every edit step,
   - ⚠️ warnings and ❌ errors,
   - **Questions** for anything the brief didn't say (with defaults).
4. **Fix and save**: answer questions by editing the brief and parsing again,
   or edit the **Recipe JSON** directly. Then click **Save recipe**.
   - ✅ **Ready to run** = valid.
   - 📝 **Draft** = still has open questions.
   Every save creates a new recipe version.
5. **Save details** saves name/brief/platforms changes without touching the
   recipe.

### How the on-screen text looks

| `style` | Look |
|---|---|
| `box` (default) | TikTok's native style: a white box hugging each line, black text |
| `clean` | White text, thick black outline and soft shadow, no box |
| `dark_box` | White text on a translucent dark panel |

The font is **Montserrat ExtraBold** (bundled in `assets/fonts`, free OFL
licence). Lines are balanced: a new sentence starts on a new line, there are
at most 3 even lines, and a word is never left alone on the last line. Long
lines get a slightly smaller size. Words wrapped in `*asterisks*` are
**highlighted** (yellow, or pink on the white box). The AI highlights the key
name or song, e.g. `Came for *Kacey*, stayed for *Estevie*`.

### On-screen text written by AI for every clip

If the brief says text on screen is required and the creator should write it
("write your own", "get creative", example lines to imitate), the parser adds
a text overlay whose text is **`{auto}`**. For every clip, the AI then writes
one fresh line (at most 90 characters). It follows the brief's rules, such as
tone, "no negativity" and "don't just announce what happened", and avoids
repeating lines used on other clips. If the recipe's `clips.focus` names the
subject (e.g. you reused a U2 brief for Mumford & Sons), the focus wins.

- The same AI call writes the **post caption** for that clip. It's shown as
  **Posting text** in Tab 4, ready to paste. It includes every @tag,
  #hashtag, required name and song to tag that the brief asks for (e.g.
  "@daniaustin @jordanjramirez @deinfluencedpodcast",
  "#TheBarbariansOfCalifornia", "Careless Whisper (Live in Paris 1988)").
  Captions are written even when the brief needs no on-screen text.
- Example lines from the brief are used as inspiration: the AI picks the
  example group that fits the clip (e.g. discovery vs. talent), rephrases,
  and varies wording across clips so the page looks organic.
- The written line is shown in **Tab 4 → Recipe for THIS file**. Edit it there
  and click **Re-render this file** to change it.
- Re-rendering keeps the same line. A new one is only written for new renders.
- If the AI is unavailable, the overlay is left out and the file gets a warning.
- Top text never covers the speaker. If the main face is in the top half of
  the frame, the line moves to the bottom and the file says so in its
  warnings. Set `"avoid_faces": false` on the overlay to force it to stay on
  top.

### Writing a good brief

Be concrete: positions, sizes, timings, platforms, length.

> Replace the audio with our summer track, add the logo top right at 15% width,
> add captions, put "Link in bio" for the last 3 seconds, vertical 9:16 for
> TikTok and Reels, under 60 seconds.

Name the assets the way they're called in the library ("summer track" →
`summer_track.mp3`).

---

## 3. Editing videos: what a recipe can do

The recipe is a list of `ops`. The parser writes it from the brief, and you
can edit the JSON yourself. Every op, with its main settings:

| Op | What it does | Main settings (defaults) |
|---|---|---|
| `trim` | Cut start/end or cap the length | `start`, `end`, `max_duration`, `min_duration` (seconds) |
| `reframe` | Change aspect ratio | `aspect` 9:16 / 4:5 / 1:1 / 16:9; `method` auto / face / center / fit_blur |
| `logo` | Place a logo/image | `asset`, `position` (top-right), `size_pct` 3–60 (12), `opacity`, `margin_pct`, `start` / `end` / `from_end` |
| `captions` | Burned-in subtitles | `style` karaoke / plain / boxed; `font`, `size` 20–160, `color`, `highlight_color`, `position` auto/top/center/bottom, `uppercase` |
| `text_overlay` | Text on screen (`"text": "{auto}"` = AI writes it per clip) | `text` (≤200 chars), `role` hook / cta / handle / disclaimer / custom, `position`, timing, `font`, `size`, `color`, `box`, `animation` none/fade/pop |
| `audio.replace` | Swap the soundtrack | `track`, `volume` 0–2, `keep_original` 0–1, `loop`, `fade_in`, `fade_out` |
| `audio.add_music` | Music under the voice | `track`, `volume` (0.15), `duck` (lowers music under speech) |
| `audio.mute_original` | Silence the original audio | none |
| `audio.volume` | Change loudness | `level` 0–3 |
| `watermark` | Faint text mark | `text`, `opacity` (0.35), `position` |
| `intro` / `outro` | Put a clip/image before/after | `asset`, `duration`, `transition` cut/fade |
| `end_card` | Closing card | `asset` and/or `text`, `duration` |
| `speed` | Speed up/slow down | `factor` 0.5–2.0 |
| `color` | Colour grade | `preset` vivid/warm/cool/bw; `brightness`, `contrast`, `saturation` |

Positions: `top-left`, `top-right`, `bottom-left`, `bottom-right`, `top`,
`bottom`, `center`, `top-center`, `bottom-center`.
Timing: `start` / `end` in seconds, or `from_end: 3` for "the last 3 seconds".

Other recipe fields:
- `mode`: `auto` | `clip` | `edit`.
- `clips`: (clip mode) `count`, `min_duration`, `max_duration`, `focus`
  (what kind of moments to look for).
- `export`: platforms. Platforms that take the same format share **one file**:
  TikTok, Reels and Shorts are all 1080×1920, so a clip becomes one file
  labelled "TikTok · Reels · Shorts", checked against the strictest limit of
  the group (Reels 90 s). `youtube` is 16:9, so it gets its own file.
- `output.filename`: e.g. `{campaign}_{input}_{clip}_{platform}`
  (tokens: campaign, input, clip, platform, variant, n).
- `variants`: up to 5 alternate versions of every file (another hook text,
  another track…). Each variant replaces the matching op or adds a new one.

Example recipe:

```json
{
  "version": 1,
  "mode": "auto",
  "clips": {"count": 5, "min_duration": 20, "max_duration": 60},
  "ops": [
    {"op": "reframe", "aspect": "9:16"},
    {"op": "logo", "asset": "acme_logo.png", "position": "top-right", "size_pct": 15},
    {"op": "captions", "style": "karaoke"},
    {"op": "audio.add_music", "track": "summer_track.mp3", "volume": 0.15},
    {"op": "text_overlay", "text": "Link in bio", "role": "cta", "from_end": 3}
  ],
  "export": ["tiktok", "reels"],
  "variants": [
    {"name": "hook_b", "ops": [{"op": "text_overlay", "role": "hook", "text": "Wait for it…"}]}
  ]
}
```

Unknown ops/settings are dropped with a warning, numbers outside their range
are clamped, and a missing required value becomes a question instead of a guess.

---

## 4. Tab 2 · Start a job

1. Pick the **Campaign**. Nothing is preselected. Below the dropdown you
   see exactly what the job will do (every edit step). A **Draft** campaign
   lists its open questions and cannot run until they are answered in Tab 1.
   Campaigns created since the page loaded appear as soon as you open the tab.
2. Add inputs: paste **one per line** (YouTube, Google Drive or direct links,
   local paths, or folders), and/or **upload videos**.
3. **Mode for all inputs**: `auto` (over 3 minutes → clip, otherwise edit),
   or force `clip` / `edit`.
4. Click **Check inputs**. The table shows duration, aspect, the chosen mode
   and why. A ⚠️ on a long video in **edit** mode means only its opening is
   used, which is usually a cold open or sponsor slate. Set override to
   `clip` to cut the best moments instead. To force one input, type `clip` or `edit` in its **override**
   cell. Links show `?` until they're downloaded.
5. Click **Run job** once. The job is queued and selected in the Jobs tab.
   Clicking again while the same job is queued or running is refused, so
   double-clicks no longer render everything twice.

---

## 5. Tab 3 · Progress: follow a job live

**Run job** brings you here automatically. A running job is opened even if
you didn't pick one. The page refreshes every 4 seconds:

- **Status card**: campaign, a status chip (Waiting / Running / Finished /
  Failed), a progress bar with %, how long it has been running, which video
  it's on, and a step checklist. Steps are ✅ done, ⏳ current (with counters
  such as "framing clip 4/13") or ○ still to come:
  Prepare → Transcribe → Understand the video → Find the best moments (AI) →
  Polish start & end → Frame for vertical → Add text & export → Quality check.
- **What's happening**: a live activity feed in plain words, newest first
  ("Exported …_04_tiktok.mp4", "Tracking faces in a clip…", "still
  transcribing speech… 3 min so far"). Long steps keep adding lines, so you
  can see it's still working.
- **Last activity N s ago**: if a step is quiet for a while, the card says
  it's still working. Framing and exporting can take several minutes per
  clip on this Mac.
- **Open in Review ▶** jumps to Tab 4 for this job. **Cancel job** stops
  after the current video. **Resume / retry failed videos** re-runs only
  unfinished or failed ones. Jobs cut off by a restart resume automatically.
- **Details** (collapsed): the videos in this job and the list of all jobs.

Long videos usually take 30–60 minutes. If the Mac sleeps, Docker pauses
and the job continues when it wakes.

## 6. Tab 4 · Review & download

1. Pick the **Job**. The **Deliverables** table lists every rendered file:
   platform/variant, duration, QA result and status.
2. **Open a file**: click its row in the table, use **◀ Prev / Next ▶**, or
   type a **Row #**. The line above the preview shows `#n of N`, the file
   name and its status. You get:
   - **Preview** player,
   - **Compliance** checks (✅ pass, ⚠️ warn, ❌ fail: duration, size and
     aspect against the platform rules, required elements present…),
   - **Posting text** (title and caption to copy),
   - **Recipe for THIS file**.
3. Click **✅ Approve** or **❌ Reject**. The confirmation appears right
   under the buttons, the table updates, and the next file opens
   automatically, so you can go through a job file by file.
4. **Edit a single file**: change the recipe JSON in the right-hand box (for
   example move the logo, change the CTA text, lower the music) and click
   **Re-render this file**. Only that file is rebuilt, with its own recipe.
   The campaign recipe is not changed.
5. **Approve all files that passed QA** approves every unreviewed file whose
   checks all passed, in one click. The line above the table counts files to
   review, approved, rejected and failed QA.
6. **Build zip** (keep **Approved files only** ticked) and then download.
7. **Re-export all files with the campaign's current settings** (collapsed
   section at the bottom) rebuilds a finished job's files after you change
   the recipe or after a tool update, e.g. to get the new text style or
   captions. It skips re-analysis and re-framing, takes about 1–2 min per
   clip, runs in the background (follow it in Tab 3) and resets approvals.
   The zip is also saved under `output/campaigns/<campaign>/<job>/`.

To change *every* future file, edit the recipe in **Tab 1** and save. To fix
one file, use the re-render in Tab 4.

---

## 7. Quick clips (no campaign)

A one-off run without creating a campaign.

1. **Video Input**: a YouTube/Drive link or a local path.
2. **Campaign Description**: optional free text (e.g. "add logo top right,
   podcast style, no music").
3. **Template**: None / podcast_clip / tiktok_reaction / motivational_reel.
4. **Vertical layout**:
   - `auto`: face-aware crop (recommended).
   - `speaker_crop`: follow the active speaker.
   - `stacked_split`: two people stacked.
   - `center_crop`
   - `branded_fit`: the whole frame on a branded background.
5. Click **🚀 Generate Clips**. The parsed config appears on the right and the
   clips appear in the gallery. Files are saved in
   `output/<videoId>_<shortId>/clip_N_final.mp4`.

---

## 8. Fix a clip (advanced)

For clips made by the clip pipeline.

1. Pick the video in **Pick a recent video** (or paste its ID), then click
   **Load review data**. You'll see the clips (hook, time range, layout) and
   all ranked candidates with scores.
2. **Open a clip**: click its row, use **◀ Prev / Next ▶**, or type the
   **Clip number**. Its details, start/end, captions and preview load. Then
   apply any fix:
   - **Boundaries**: new **Start** / **End** in seconds → **Apply boundaries**
     (your values override the automatic ones).
   - **Layout**: pick an override → **Apply layout**.
   - **Manual crop**: x, y, w, h in source-video pixels (a 9:16 box),
     optionally limited to a `t_start`–`t_end` window → **Apply manual crop**.
     Face detection will not override it.
   - **Captions**: edit the **text** column only (keep the same number of
     words per cue, because timings are fixed) → **Apply captions**.
3. Click **Re-render this clip**. Only that clip is rebuilt, and the new
   version plays in **Current final**.

---

## 9. Where things are

| What | Where (in `~/Desktop/senna-dev`) |
|---|---|
| Campaign deliverables + zips | `output/campaigns/<campaign>/<job>/` |
| Quick-clip outputs | `output/<videoId>_<shortId>/` |
| Downloads (source videos) | `downloads/` |
| Intermediate files / resume state | `work/` |
| Logs | `logs/` + `docker logs senna-dev-app-1` |
| Settings / API keys | `.env` (never commit), `config.yaml` |

---

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| Page doesn't load | `docker compose up -d`, then wait about 20 s; check `docker ps` |
| Video starts with someone else's intro / wrong text | Check the campaign named in Tab 2's summary before **Run job** |
| Long video gave one short clip of its opening | It ran in **edit** mode; rerun with override `clip` |
| Recipe stays 📝 Draft | Answer the **Questions** (edit brief → Parse, or edit JSON) → Save recipe |
| "asset not found" warning | Upload the asset in Tab 1, then parse again; names must match |
| Job stuck / failed input | Tab 3 → read the error → **Resume / retry failed inputs** |
| Logs say `falling back to Groq` | Claude was briefly unavailable; the run continues on Groq |
| Logs say `Claude auth failed` | Check `ANTHROPIC_API_KEY` in `.env`, then `docker compose up -d` |
| Clip starts/ends mid-sentence | Tab 6 → fix boundaries → re-render |
| Wrong face/crop | Tab 6 → layout override or manual crop → re-render |
| Changed code or requirements | `docker compose up -d --build` |
