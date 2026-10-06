"""preview_framing.py — render short layout previews + boundary demo (verification only).

Writes to output/previews/ (never touches existing finals).
Usage: .venv/Scripts/python.exe preview_framing.py
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))

from config import ffmpeg_path
from pipeline.framing import sample_clip, decide_layout, render_vertical
from pipeline.boundaries import load_words, build_sentences, refine_clip

SRC = "input/raw_video.mp4"
OUTDIR = os.path.join("output", "previews")
os.makedirs(OUTDIR, exist_ok=True)

# 8s demo window inside clip_1's range (30-55s per report.json)
START, DUR = 32.0, 8.0
sub = os.path.join(OUTDIR, "demo_source_8s.mp4")
if not os.path.exists(sub):
    r = subprocess.run([ffmpeg_path(), "-y", "-ss", str(START), "-t", str(DUR),
                        "-i", SRC, "-c:v", "libx264", "-preset", "fast",
                        "-crf", "23", "-c:a", "aac", sub], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]
    print("subclip ok:", sub, os.path.getsize(sub), "bytes")

cfg = {"sample_interval": 0.5, "layout_stability_frames": 4}  # faster demo sampling
analysis = sample_clip(sub, cfg)
n_with = sum(1 for s in analysis["samples"] if s["n"] > 0)
print(f"samples={len(analysis['samples'])} with_faces={n_with} "
      f"scenes={analysis['scenes']} size={analysis['width']}x{analysis['height']}")

for layout in ("auto", "speaker_crop", "stacked_split", "branded_fit"):
    out = os.path.join(OUTDIR, f"preview_{layout}.mp4")
    res = render_vertical(sub, out, layout=layout, cfg=cfg, analysis=analysis)
    print(f"{layout}: {res['layout']} | {res['layout_reason'][:90]} | frames={res['frames']}")

# boundary demo: candidate ending mid-sentence from the real transcript
words = load_words("transcripts/transcript.json")
sents = build_sentences(words)
print(f"transcript words={len(words)} sentences={len(sents)}")
# find a sentence longer than 4s starting after 30s, cut candidate mid-next-sentence
demo = None
for i, s in enumerate(sents):
    if s["start_time"] >= 30 and (s["end_time"] - s["start_time"]) > 3 \
            and i + 1 < len(sents):
        nxt = sents[i + 1]
        mid = (nxt["start_time"] + nxt["end_time"]) / 2
        demo = (s, nxt, mid)
        break
if demo:
    s, nxt, mid = demo
    clip = {"start_time": s["start_time"], "end_time": mid, "hook": "demo"}
    det_cfg = {"context_seconds": 20, "max_extension_seconds": 15, "min_duration": 5,
               "max_duration": 90, "start_padding": 0.15, "end_padding": 0.3,
               "llm_validation": False}
    refined, action = refine_clip(clip, words, sents, video_duration=324.2, cfg=det_cfg)
    print(f"candidate end={mid:.2f}s (mid-sentence '{nxt['text'][:50]}...')")
    print(f"refined end={refined['refined_end_time']:.2f}s action={action} "
          f"status={refined['refine_status']}")
    print("reason:", refined["refine_reason"][:160])
    with open(os.path.join(OUTDIR, "boundary_demo.json"), "w") as f:
        json.dump({"candidate": {"start": clip["start_time"], "end": mid},
                   "refined": {"start": refined["refined_start_time"],
                               "end": refined["refined_end_time"]},
                   "action": action, "reason": refined["refine_reason"],
                   "cut_sentence": nxt["text"],
                   "completed_sentence_end": nxt["end_time"]}, f, indent=2)
print("previews in", OUTDIR)
