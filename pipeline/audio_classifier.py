"""
pipeline/audio_classifier.py

Optional event-label adapter (laughter / applause / music / speech).

Backends:
  - "none" (default): disabled adapter. classify_events() returns events
    unchanged with status disabled — never fake labels.
  - "yamnet": YAMNet (Google, ~15MB TFLite + weights) when installed. Enable
    with audio_events.classifier.backend=yamnet and a model dir containing
    yamnet.tflite (see README section below). Requires `pip install
    tflite-runtime` (or tensorflow) + numpy; ~50MB RAM, CPU realtime-ish on
    short clips. Missing package/model/dir -> disabled status with a clear
    log line (graceful, never an exception to the caller).

Label mapping (YAMNet 521 classes -> our labels): laughter ("/m/01j4z9" etc:
classes containing 'augh'), applause (classes containing 'pplause' or
'rowd'), music (classes containing 'usic'), speech otherwise. Per-event label
= top mapped class by mean score inside the event window; label_confidence =
that mean. Uncertainty preserved: low-confidence means stay unlabeled (None).
"""
from logger import get_logger

log = get_logger("pipeline.audio_classifier")


def backend_status(cfg=None):
    """Report classifier backend availability without loading models."""
    cfg = (cfg or {}).get("classifier", {}) if isinstance(cfg, dict) else {}
    backend = str(cfg.get("backend", "none")).lower()
    if backend == "none":
        return {"backend": "none", "status": "disabled",
                "reason": "classifier backend not configured"}
    if backend == "yamnet":
        import os
        model_dir = cfg.get("model_dir", "models/yamnet")
        tflite = os.path.join(model_dir, "yamnet.tflite")
        if not os.path.exists(tflite):
            return {"backend": "yamnet", "status": "disabled",
                    "reason": f"model missing at {tflite}; download yamnet.tflite "
                              f"into models/yamnet/ to enable"}
        try:
            import tflite_runtime.interpreter  # noqa
        except ImportError:
            try:
                import tensorflow  # noqa
            except ImportError:
                return {"backend": "yamnet", "status": "disabled",
                        "reason": "tflite-runtime (or tensorflow) not installed; "
                                  "pip install tflite-runtime to enable"}
        return {"backend": "yamnet", "status": "ready", "model": tflite}
    return {"backend": backend, "status": "disabled",
            "reason": f"unknown classifier backend {backend!r}"}


def classify_events(events, audio=None, sr=16000, cfg=None):
    """Attach optional labels. Returns (events, status_dict). Never raises."""
    import traceback
    try:
        st = backend_status(cfg)
        if st["status"] != "ready" or not events:
            if events and st["status"] != "ready":
                log.info("classifier %s", st.get("reason", "disabled"))
            return events, st
        backend = st["backend"]
        if backend == "yamnet":
            return _classify_yamnet(events, audio, sr, cfg, st)
        return events, st
    except Exception:
        log.warning("classifier failed, leaving unlabeled: %s",
                    traceback.format_exc(limit=3))
        return events, {"backend": "?", "status": "disabled",
                        "reason": "classifier exception"}


def _yamnet_scores(audio, sr, model_path):
    """Run YAMNet over 16kHz mono audio. Returns (scores, class_names)."""
    import numpy as np
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter
    import csv
    import os
    it = Interpreter(model_path=model_path)
    inp = it.get_input_details()[0]
    n = inp["shape"][1] if len(inp["shape"]) > 1 else 16000
    it.resize_tensor_input(inp["index"], [n], strict=False)
    it.allocate_tensors()
    out = it.get_output_details()[0]
    scores = []
    step = n  # YAMNet: 0.96s windows, 50% overlap handled by model stride
    for off in range(0, max(1, len(audio) - n + 1), n // 2):
        seg = np.zeros(n, dtype=np.float32)
        chunk = audio[off:off + n].astype(np.float32)
        seg[:len(chunk)] = chunk
        it.set_tensor(inp["index"], seg.reshape(it.get_input_details()[0]["shape"]))
        it.invoke()
        scores.append(it.get_tensor(out["index"])[0].copy())
    scores = np.asarray(scores)
    names, mp = [], os.path.join(os.path.dirname(model_path), "yamnet_class_map.csv")
    if os.path.exists(mp):
        with open(mp) as f:
            for row in csv.DictReader(f):
                names.append(row.get("display_name", ""))
    return scores, names


def _map_label(name):
    n = (name or "").lower()
    if "augh" in n or "giggle" in n or "chuckle" in n:
        return "laughter"
    if "pplause" in n or ("rowd" in n and "heer" in n):
        return "applause"
    if "usic" in n:
        return "music"
    if "peech" in n or "alk" in n or "onversation" in n:
        return "speech"
    return None


def _classify_yamnet(events, audio, sr, cfg, st):
    import numpy as np
    cfg = (cfg or {}).get("classifier", {})
    min_conf = float(cfg.get("min_label_confidence", 0.3))
    scores, names = _yamnet_scores(np.asarray(audio, dtype=np.float32),
                                   sr, st["model"])
    hop = 0.48  # YAMNet frame hop for 0.96s windows at 50% overlap
    for ev in events:
        i0 = max(0, int(ev["start_time"] / hop))
        i1 = min(len(scores), max(i0 + 1, int(ev["end_time"] / hop)))
        win = scores[i0:i1]
        if win.size == 0:
            continue
        mean = win.mean(axis=0)
        order = np.argsort(mean)[::-1][:5]
        for idx in order:
            label = _map_label(names[int(idx)] if int(idx) < len(names) else "")
            if label and float(mean[int(idx)]) >= min_conf:
                ev["label"] = label
                ev["label_confidence"] = round(float(mean[int(idx)]), 3)
                break
    log.info("yamnet labeled %d/%d events",
             sum(1 for e in events if e.get("label")), len(events))
    return events, dict(st, status="labeled")
