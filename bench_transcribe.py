"""bench_transcribe.py — transcription model benchmark (evidence, no default switch).

Compares WhisperX configs on a sample of real audio:
  runtime (load/transcribe/align), peak GPU memory, segment/word counts,
  language agreement, cross-model text agreement (difflib).

Models NOT already cached are skipped unless --allow-download is passed
(large-v3/turbo are multi-GB; never downloaded silently). Findings are
reported; defaults in config.yaml are never changed by this script.

Usage:
    python bench_transcribe.py [--audio downloads/audio.wav] [--offset 30]
        [--duration 60] [--models base,turbo] [--allow-download]
"""
import argparse
import difflib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

MODEL_REPOS = {
    "base": ["Systran/faster-whisper-base"],
    "small": ["Systran/faster-whisper-small"],
    "medium": ["Systran/faster-whisper-medium"],
    "large-v3": ["Systran/faster-whisper-large-v3"],
    "turbo": ["mobiuslabsgmbh/faster-whisper-large-v3-turbo",
              "deepdml/faster-whisper-large-v3-turbo-ct2"],
}


def is_cached(model):
    hub = os.path.expanduser("~/.cache/huggingface/hub")
    for repo in MODEL_REPOS.get(model, []):
        if os.path.isdir(os.path.join(hub, "models--" + repo.replace("/", "--"))):
            return True
    return False


def peak_gpu_mb():
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
    except Exception:
        pass
    return 0.0


def bench_one(audio_path, model, device, compute, language):
    import whisperx
    import librosa
    import numpy as np
    import torch
    rec = {"model": model, "device": device, "compute": compute}
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        m = whisperx.load_model(model, device=device, compute_type=compute)
        rec["load_s"] = round(time.monotonic() - t0, 1)
        y, _ = librosa.load(audio_path, sr=16000, mono=True)
        y = y.astype(np.float32)
        t0 = time.monotonic()
        res = m.transcribe(y, language=language, batch_size=16)
        rec["transcribe_s"] = round(time.monotonic() - t0, 1)
        rec["language"] = res.get("language")
        rec["segments"] = len(res.get("segments", []))
        try:
            ma, meta = whisperx.load_align_model(
                language_code=res["language"], device=device)
            t0 = time.monotonic()
            res = whisperx.align(res["segments"], ma, meta, y, device,
                                 return_char_alignments=False)
            rec["align_s"] = round(time.monotonic() - t0, 1)
            rec["words"] = sum(len(s.get("words", [])) for s in res.get("segments", []))
            del ma
        except Exception as e:
            rec["align_error"] = str(e)[:200]
            rec["words"] = 0
        rec["text"] = " ".join(s.get("text", "") for s in res.get("segments", []))
        rec["peak_gpu_mb"] = peak_gpu_mb()
        del m
        import gc
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    except Exception as e:
        rec["error"] = str(e)[:300]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="downloads/audio.wav")
    ap.add_argument("--offset", type=float, default=30.0)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--models", default="base")
    ap.add_argument("--device", default=None)  # default: configured device
    ap.add_argument("--allow-download", action="store_true")
    args = ap.parse_args()

    from config import get_config, ffmpeg_path
    import subprocess
    import tempfile
    cfg = (get_config().get("transcription", {}) or {})
    device = args.device or cfg.get("device", "cpu")
    if str(device).lower() == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    compute = cfg.get("compute_type", "float32")
    language = cfg.get("language", "en")

    sample = os.path.join(tempfile.gettempdir(), "bench_sample.wav")
    subprocess.run([ffmpeg_path(), "-y", "-ss", str(args.offset),
                    "-t", str(args.duration), "-i", args.audio,
                    "-ar", "16000", "-ac", "1", sample],
                   capture_output=True, check=True)
    print(f"sample: {args.audio} [{args.offset}-{args.offset + args.duration}s]")

    results, skipped = [], []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        if not is_cached(model) and not args.allow_download:
            skipped.append({"model": model,
                            "reason": "not cached; pass --allow-download to fetch"})
            print(f"SKIP {model}: not cached")
            continue
        print(f"bench {model} on {device}/{compute} ...", flush=True)
        results.append(bench_one(sample, model, device, compute, language))
    # agreement vs first successful result
    refs = [r for r in results if r.get("text")]
    if refs:
        base_text = refs[0]["text"]
        for r in results:
            t = r.get("text", "")
            r["agree_ratio_vs_first"] = round(
                difflib.SequenceMatcher(None, base_text, t).ratio(), 3) if t else 0.0
            r["word_count"] = len(t.split())
            del r["text"]
    out = {"sample": {"audio": args.audio, "offset": args.offset,
                      "duration": args.duration},
           "device": device, "compute": compute, "results": results,
           "skipped": skipped}
    os.makedirs("output", exist_ok=True)
    # merge with prior runs (keyed model+device) instead of overwriting
    path = "output/bench_transcription.json"
    try:
        prev = json.load(open(path))
        prior = {(r.get("model"), r.get("device")): r
                 for r in prev.get("results", [])}
        prior_skip = {s.get("model"): s for s in prev.get("skipped", [])}
    except (OSError, ValueError):
        prior, prior_skip = {}, {}
    for r in results:
        r["device"] = device
        if r.get("error"):
            continue  # never persist failed runs
        prior[(r.get("model"), device)] = r
    for s in skipped:
        prior_skip[s.get("model")] = s
    out["results"] = list(prior.values())
    out["skipped"] = list(prior_skip.values())
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"{'model':10} {'load':>6} {'transc':>7} {'align':>6} "
          f"{'segs':>5} {'words':>6} {'gpuMB':>7} {'agree':>6}")
    for r in results:
        print(f"{r['model']:10} {r.get('load_s', '?'):>6} "
              f"{r.get('transcribe_s', '?'):>7} {r.get('align_s', '?'):>6} "
              f"{r.get('segments', '?'):>5} {r.get('words', '?'):>6} "
              f"{r.get('peak_gpu_mb', '?'):>7} "
              f"{r.get('agree_ratio_vs_first', '?'):>6}  {r.get('error', '')}")
    print("wrote output/bench_transcription.json")


if __name__ == "__main__":
    main()
