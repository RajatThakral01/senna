"""Build synthetic face-positive demos + re-render layout previews.

1. single-face: 5s clip from a public-domain portrait (Obama, White House photo)
   -> speaker_crop must center the crop on the detected face.
2. two-face: side-by-side portrait + mirror -> stacked_split must produce
   top/bottom panels.
3. re-render the 4 real-footage previews with the fixed detector.
"""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
import cv2
from config import ffmpeg_path
from pipeline.framing import sample_clip, decide_layout, render_vertical

OUTDIR = os.path.join("output", "previews")
os.makedirs(OUTDIR, exist_ok=True)
portrait = os.path.join(tempfile.gettempdir(), "obama_test.jpg")
assert os.path.exists(portrait), "missing portrait"

# --- 1. single-face 5s 1920x1080 clip (slow zoom to emulate video) ---
single_img = os.path.join(OUTDIR, "face_single_src.mp4")
if not os.path.exists(single_img):
    r = subprocess.run(
        [ffmpeg_path(), "-y", "-loop", "1", "-i", portrait,
         "-vf", ("scale=2687:-1,crop=2687:1511:0:200,"
                 "scale=1920:1080,format=yuv420p"),
         "-t", "5", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-an", single_img], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]

cfg = {"sample_interval": 0.5, "layout_stability_frames": 4}
a1 = sample_clip(single_img, cfg)
n1 = sum(1 for s in a1["samples"] if s["n"] > 0)
d1 = decide_layout(a1, cfg, "auto")
print(f"single-face: samples={len(a1['samples'])} with_faces={n1} auto->{d1['layout']}")
r1 = render_vertical(single_img, os.path.join(OUTDIR, "demo_face_speaker_crop.mp4"),
                     layout="auto", cfg=cfg, analysis=a1)
print("rendered:", r1["layout"], r1["layout_reason"][:100])

# --- 2. two-face side-by-side -> stacked ---
pair = os.path.join(OUTDIR, "face_pair.png")
if not os.path.exists(pair):
    img = cv2.imread(portrait)
    h, w = img.shape[:2]
    half = cv2.resize(img, (w // 2, h))
    mirror = cv2.flip(half, 1)
    cv2.imwrite(pair, cv2.hconcat([half, mirror]))
pair_vid = os.path.join(OUTDIR, "face_pair_src.mp4")
if not os.path.exists(pair_vid):
    r = subprocess.run(
        [ffmpeg_path(), "-y", "-loop", "1", "-i", pair,
         "-vf", "scale=1920:1080,format=yuv420p", "-t", "5",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-an", pair_vid], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]
a2 = sample_clip(pair_vid, cfg)
n2 = sum(1 for s in a2["samples"] if s["n"] >= 2)
d2 = decide_layout(a2, cfg, "auto")
print(f"two-face: samples={len(a2['samples'])} with_2faces={n2} auto->{d2['layout']}")
r2 = render_vertical(pair_vid, os.path.join(OUTDIR, "demo_faces_stacked.mp4"),
                     layout="auto", cfg=cfg, analysis=a2)
print("rendered:", r2["layout"], r2["layout_reason"][:100])
print("done")
