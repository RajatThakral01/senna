"""tools/framing_audit.py — measure framing quality of a rendered video.

Usage:
    python tools/framing_audit.py <video_id> [--clips 1,2,3] [--json out.json]

For every rendered clip it re-cuts the clip from the source (so it does not
need the intermediates render deletes), rebuilds the exact framing plan the
renderer uses (sample_clip -> plan_scene_crops -> _render_plan ->
_frame_rects) and checks it at every 0.1s sample against the detected faces:

  wrong person (1s window)  a big face (>= 6% width) clearly out-talks the rest
                            over +-0.5s while voiced, but is not in the crop.
                            A PROXY: lip motion is fooled by camera moves,
                            chewing and faces appearing — confirm by eye.
  sliced                    a face partly in / partly out of the crop
  no face in crop           faces in the picture, none inside the crop
  off-centre                followed face > 18% of crop width from centre
  head cut                  top of a face above the top of the crop
  split-screen / random     stacked panels; "random" = 3+ faces or < 2 big faces
  crowd / salient / holding share of samples in group, saliency, retained modes

Use it before/after framing changes on the same video; judge the flagged
moments in the --debug-framing videos.
"""
import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging  # noqa: E402

from config import get_config  # noqa: E402
from db.repositories import clip_repo, video_repo  # noqa: E402
from main import select_renderable_clips  # noqa: E402
from pipeline import framing as F  # noqa: E402
from pipeline import workspace  # noqa: E402
from pipeline.clipper import cut_clips  # noqa: E402


def audit_clip(path, cfg):
    a = F.sample_clip(path, cfg)
    b = a["content_bounds"]
    fps, W, H = a["fps"], a["width"], a["height"]
    sp = F.plan_scene_crops(a, cfg)
    lay = F.decide_layout(a, cfg, "auto")["layout"]
    plans = F._render_plan(a, sp, cfg, fps, int(round(a["duration"] * fps)) + 2, lay, b,
                           requested="auto")
    rows = []
    for s in a["samples"]:
        i = int(round(s["t"] * fps))
        p = next((p for p in plans if p["f0"] <= i < p["f1"]), plans[-1])
        k = min(max(i - p["f0"], 0), max(0, p["f1"] - p["f0"] - 1))
        rects, kind = F._frame_rects(p, k, b, None, W, H)
        faces = []
        for tr in s["tracks"]:
            bx = tr["box"]
            fw, fh = bx["x2"] - bx["x1"], bx["y2"] - bx["y1"]
            best, off, headcut = 0.0, None, False
            for (x, y, cw, ch, _) in rects:
                ix = max(0, min(bx["x2"], x + cw) - max(bx["x1"], x))
                iy = max(0, min(bx["y2"], y + ch) - max(bx["y1"], y))
                frac = ix * iy / max(1, fw * fh)
                if frac > best:
                    best, off = frac, abs(tr["cx"] - (x + cw / 2)) / cw
                    headcut = bx["y1"] < y - 0.1 * fh
            faces.append({"id": tr["id"], "inside": best, "off": off, "headcut": headcut,
                          "w": fw / W, "mouth": tr.get("mouth")})
        rows.append({"t": s["t"], "kind": kind, "mode": p["mode"], "voiced": s.get("voiced"),
                     "subj": p.get("shot", {}).get("state", p.get("track_id")), "faces": faces})
    return rows


def metrics(rows_by_clip):
    m = dict(n=0, faces=0, none_in=0, sliced=0, offc=0, single=0, head=0,
             wrongN=0, wrong=0, stacked=0, rand2=0, group=0, salient=0, retain=0)
    for rows in rows_by_clip.values():
        for i, r in enumerate(rows):
            m["n"] += 1
            m["group"] += r["kind"] == "group"
            m["salient"] += r["kind"] == "salient"
            m["retain"] += r["kind"] == "retain"
            F_ = r["faces"]
            if r["mode"] == "stacked":
                m["stacked"] += 1
                if len(F_) >= 3 or sum(1 for f in F_ if f["w"] >= 0.045) < 2:
                    m["rand2"] += 1
            if not F_:
                continue
            m["faces"] += 1
            m["none_in"] += max(f["inside"] for f in F_) < 0.5
            m["sliced"] += any(0.15 < f["inside"] < 0.85 and f["w"] > 0.04 for f in F_)
            m["head"] += any(f["inside"] > 0.3 and f["headcut"] for f in F_)
            if r["mode"] == "single" and r["kind"] == "face":
                m["single"] += 1
                m["offc"] += any(f["id"] == r["subj"] and f["off"] is not None and f["off"] > 0.18
                                 for f in F_)
            if r["voiced"]:
                ids = {f["id"]: f for f in F_}
                acc = {}
                for j in range(max(0, i - 5), min(len(rows), i + 6)):
                    for f in rows[j]["faces"]:
                        if f["id"] in ids and f["mouth"] is not None:
                            acc.setdefault(f["id"], []).append(f["mouth"])
                sc = sorted(((sum(v) / len(v), k) for k, v in acc.items() if len(v) >= 4),
                            reverse=True)
                if (len(sc) >= 2 and sc[0][0] > 0.15 and sc[0][0] > 2.0 * sc[1][0]
                        and ids[sc[0][1]]["w"] >= 0.06):
                    m["wrongN"] += 1
                    m["wrong"] += ids[sc[0][1]]["inside"] < 0.5
    return m


def main():
    ap = argparse.ArgumentParser(description="Framing quality audit for one video")
    ap.add_argument("video_id")
    ap.add_argument("--clips", default=None, help="comma-separated clip numbers")
    ap.add_argument("--json", default=None, help="also write per-sample rows here")
    args = ap.parse_args()
    logging.disable(logging.INFO)
    video = video_repo.get_video(args.video_id)
    if not video:
        raise SystemExit(f"unknown video {args.video_id}")
    workspace.activate(args.video_id)
    cfg = dict(get_config()["framing"])
    clips, _ = select_renderable_clips(clip_repo.get_clips_for_video(args.video_id))
    if args.clips:
        want = {int(x) for x in args.clips.split(",")}
        clips = [c for c in clips if c["clip_number"] in want]
    tmp = tempfile.mkdtemp(prefix="framing_audit_")
    cut_clips(video["raw_path"], clips, output_dir=tmp)
    rows = {}
    for c in clips:
        path = os.path.join(tmp, f"clip_{c['clip_number']}.mp4")
        if os.path.exists(path):
            rows[c["clip_number"]] = audit_clip(path, cfg)
            os.remove(path)
            print(f"clip {c['clip_number']} audited", flush=True)
    os.rmdir(tmp)
    m = metrics(rows)
    f, n = max(1, m["faces"]), max(1, m["n"])
    print(f"\nsamples={m['n']} (10/s) across {len(rows)} clips")
    print(f"  wrong person (1s window, big face) {m['wrong']}/{m['wrongN']}"
          f" = {m['wrong'] / max(1, m['wrongN']):.0%}   (proxy; check by eye)")
    print(f"  face sliced by crop edge   {m['sliced'] / f:.1%}")
    print(f"  faces present, none in crop {m['none_in'] / f:.1%}")
    print(f"  followed face off-centre   {m['offc'] / max(1, m['single']):.1%}")
    print(f"  head cut at top            {m['head'] / f:.1%}")
    print(f"  split-screen               {m['stacked'] / n:.1%}"
          f"  (random pairs {m['rand2'] / max(1, m['stacked']):.0%})")
    print(f"  crowd shots {m['group'] / n:.1%} | salient {m['salient'] / n:.1%}"
          f" | holding {m['retain'] / n:.1%}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, default=float)


if __name__ == "__main__":
    main()
