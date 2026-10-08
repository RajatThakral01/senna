"""validate_fullscreen.py — re-render clips with full-screen vertical framing
and validate the actual outputs (req 6).

Usage:
    .venv/Scripts/python.exe validate_fullscreen.py <video_id> [clip_numbers...]

Steps:
  1. Invalidate the render checkpoint + delete existing finals for the clips.
  2. render_clips(video_id, clip_numbers=[...]) through the live main.py path.
  3. Per final: dims == 1080x1920, duration sane, audio present, edge bands
     are real content (not uniform fill), layout != branded_fit.
  4. Debug crop previews (crop rect on source) -> output/previews/crop_debug/.
  5. Prints a shot/transition report incl. needs_review flags.
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))

from config import ffprobe_path
from logger import setup_logging, get_logger

setup_logging()
log = get_logger("validate")

OUT_W, OUT_H = 1080, 1920


def probe(path):
    r = subprocess.run(
        [ffprobe_path(), "-v", "error", "-show_entries",
         "stream=width,height,codec_type,codec_name,duration",
         "-show_entries", "format=duration",
         "-of", "json", path], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {r.stderr[-300:]}")
    return json.loads(r.stdout)


def edge_uniformity(path, src_path=None, band=8, samples=8, fade_margin=1.5):
    """Detect renderer-generated fill bars (static across time).

    A fill bar is the SAME flat colour on every frame. Real content — even
    crushed-black walls — varies as the shot/crop moves. Returns
    {edge: min-std} where an edge counts as fill only if it is static
    (mean range < 3.0) AND flat (all std < 1.0) across all samples.
    Fade zones (first/last 1.5s) are excluded. src_path is accepted for
    API compatibility and ignored (crop offsets make row mapping invalid).
    """
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    margin = int(fade_margin * fps)
    lo, hi = margin, max(margin, n - margin)
    idxs = [int(lo + (hi - lo) * (i + 1) / (samples + 1)) for i in range(samples)]
    stats = {"top": [], "bottom": [], "left": [], "right": []}
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
        h, w = gray.shape
        stats["top"].append((gray[:band, :].mean(), gray[:band, :].std()))
        stats["bottom"].append((gray[-band:, :].mean(), gray[-band:, :].std()))
        stats["left"].append((gray[:, :band].mean(), gray[:, :band].std()))
        stats["right"].append((gray[:, -band:].mean(), gray[:, -band:].std()))
    cap.release()
    out = {}
    for k, v in stats.items():
        if not v:
            out[k] = -1.0
            continue
        means = [m for m, _ in v]
        stds = [s for _, s in v]
        static_fill = (max(means) - min(means)) < 3.0 and all(s < 1.0 for s in stds)
        # 0.0 = static flat bar; otherwise the across-time variation (high = live content)
        out[k] = 0.0 if static_fill else round(float(max(means) - min(means)), 2)
    return out


def main():
    from main import render_clips, invalidate_stage
    from pipeline.framing import render_crop_debug
    video_id = sys.argv[1]
    args = [a for a in sys.argv[2:] if not a.startswith("--")]
    numbers = [int(a) for a in args] or [1, 2, 3]
    check_only = "--check-only" in sys.argv

    if not check_only:
        invalidate_stage(video_id, "render")
        for n in numbers:
            for p in (f"output/clip_{n}_final.mp4", f"clips/clip_{n}_vertical.mp4"):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except OSError:
                    pass
        log.info("re-rendering clips=%s", numbers)
        finals = render_clips(video_id, clip_numbers=numbers)
        log.info("rendered=%s", finals)
        assert len(finals) == len(numbers), f"expected {len(numbers)}, got {len(finals)}"
    else:
        log.info("check-only on clips=%s (no re-render)", numbers)

    report = json.load(open("output/report.json"))
    by_no = {c["clip_number"]: c for c in report}
    ok, issues = True, []
    for n in numbers:
        c = by_no[n]
        if c.get("refine_status") == "rejected":
            print(f"--- clip {n}: SKIPPED (rejected: "
                  f"{(c.get('refine_reason') or '')[:100]}) ---")
            continue
        final = f"output/clip_{n}_final.mp4"
        info = probe(final)
        v = [s for s in info["streams"] if s["codec_type"] == "video"][0]
        a = [s for s in info["streams"] if s["codec_type"] == "audio"]
        dur = float(info["format"]["duration"])
        edges = edge_uniformity(final, src_path=f"clips/clip_{n}.mp4")
        flat = [k for k, s in edges.items() if 0 <= s < 0.5]
        checks = {
            "dims_1080x1920": (v["width"], v["height"]) == (OUT_W, OUT_H),
            "has_audio": bool(a),
            "duration_sane": abs(dur - float(c.get("output_duration", dur))) < 2.0,
            "no_fill_bars": not flat,
            "layout": c.get("layout"),
            "needs_review_plan": None,
        }
        # needs_review from latest edit plan
        try:
            from db.repositories import editplan_repo
            from db.repositories import clip_repo
            clip = next(x for x in clip_repo.get_clips_for_video(video_id)
                        if x["clip_number"] == n)
            plan = editplan_repo.get_latest_plan(clip["id"])
            pc = plan["plan"] if isinstance(plan.get("plan"), dict) else {}
            checks["needs_review_plan"] = (pc.get("framing") or {}).get("needs_review")
            checks["scene_count"] = len((pc.get("framing") or {}).get("scene_crops", []))
        except Exception as e:
            checks["plan_error"] = str(e)[:120]
        print(f"--- clip {n}: {final} ---")
        for k, val in checks.items():
            print(f"    {k}: {val}")
        print(f"    edge_std: {edges}")
        print(f"    layout_reason: {(c.get('layout_reason') or '')[:160]}")
        bad = [k for k, val in checks.items()
               if val is False or (k == "layout" and val == "branded_fit")]
        if bad or flat:
            ok = False
            issues.append(f"clip {n}: failed={bad} flat_edges={flat}")

        # debug previews: crop rect drawn on source frames
        os.makedirs("output/previews/crop_debug", exist_ok=True)
        try:
            from pipeline.framing import sample_clip
            analysis = sample_clip(f"clips/clip_{n}.mp4",
                                   {"sample_interval": 0.5,
                                    "layout_stability_frames": 4})
            paths = render_crop_debug(
                f"clips/clip_{n}.mp4",
                f"output/previews/crop_debug/clip_{n}_crop.jpg",
                analysis=analysis, cfg={"sample_interval": 0.5},
                manual_rois=[], n=4)
            print(f"    crop_debug: {paths}")
        except Exception as e:
            log.exception("crop debug failed clip=%s", n)
            issues.append(f"clip {n}: crop_debug error {e}")
            ok = False

    print("=== RESULT:", "PASS — all shots full-screen vertical" if ok else "FAIL")
    for i in issues:
        print("  -", i)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
