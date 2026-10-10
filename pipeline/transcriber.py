# transcriber.py
import time
import whisperx
import json
import os
from logger import get_logger, log_stage
from config import WHISPER_MODEL, WHISPER_LANGUAGE, WHISPER_DEVICE, WHISPER_COMPUTE

log = get_logger("pipeline.transcriber")

# Optional model cache for batch work (campaign edit mode transcribes many
# short videos; loading Whisper costs ~7 s each time). Off by default so the
# long-video pipeline still frees memory right after transcription.
_MODEL_CACHE = {}


def _load_model(model, device, compute, keep):
    key = ("asr", model, device, compute)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    m = whisperx.load_model(model, device=device, compute_type=compute)
    if keep:
        _MODEL_CACHE[key] = m
    return m


def _load_align(language, device, keep):
    key = ("align", language, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    pair = whisperx.load_align_model(language_code=language, device=device)
    if keep:
        _MODEL_CACHE[key] = pair
    return pair


def release_models():
    """Drop cached Whisper/align models (call after a batch)."""
    _MODEL_CACHE.clear()
    import gc
    gc.collect()

def _transcription_settings():
    """Model/device/compute/language from config.yaml [transcription] + env.

    Falls back to legacy WHISPER_* module constants when config is absent.
    """
    try:
        from config import get_config
        t = get_config().get("transcription", {}) or {}
        device = t.get("device") or WHISPER_DEVICE
        if str(device).lower() == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        compute = t.get("compute_type") or WHISPER_COMPUTE
        if str(compute).lower() == "auto":
            # int8 on CPU (~1.2x faster than float32, same words in our
            # benchmark); float16 on CUDA
            compute = "float16" if device == "cuda" else "int8"
        return (t.get("model") or WHISPER_MODEL,
                t.get("language") or WHISPER_LANGUAGE,
                device,
                compute,
                t.get("glossary") or [])
    except Exception:
        return WHISPER_MODEL, WHISPER_LANGUAGE, WHISPER_DEVICE, WHISPER_COMPUTE, []


def transcribe_audio(audio_path, output_dir="transcripts", keep_models=False):
    """
    Transcribes audio using WhisperX with word-level timestamps.
    Returns path to transcript JSON file.
    """
    log.info("transcribe start audio=%s model=%s lang=%s device=%s", audio_path, WHISPER_MODEL, WHISPER_LANGUAGE, WHISPER_DEVICE)
    if not os.path.exists(audio_path):
        log.error("audio file missing path=%s", audio_path)
        raise FileNotFoundError(audio_path)
    MODEL, LANG, DEVICE, COMPUTE, GLOSSARY = _transcription_settings()
    log.info("transcribe settings model=%s lang=%s device=%s compute=%s glossary=%d",
             MODEL, LANG, DEVICE, COMPUTE, len(GLOSSARY or []))
    t0 = time.monotonic()
    with log_stage("pipeline.transcriber", "load_model"):
        log.info("loading WhisperX model=%s device=%s", MODEL, DEVICE)
        # Load WhisperX model
        model = _load_model(MODEL, DEVICE, COMPUTE, keep_models)

    with log_stage("pipeline.transcriber", "transcribe"):
        log.info("transcribing audio (may take minutes) path=%s", audio_path)

        # Load audio using librosa (bypasses torchcodec issues)
        import librosa
        import numpy as np
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        audio = audio.astype(np.float32)
        log.debug("audio loaded samples=%d", len(audio))
        result = model.transcribe(audio, language=LANG, batch_size=16)
    log.info("transcribe raw segments=%d lang=%s", len(result.get("segments", [])), result.get("language"))

    # Align for word-level timestamps
    with log_stage("pipeline.transcriber", "align"):
        log.info("aligning word-level timestamps...")
        model_a, metadata = _load_align(result["language"], DEVICE, keep_models)
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            DEVICE,
            return_char_alignments=False
        )
    nwords = sum(len(s.get("words", [])) for s in result.get("segments", []))
    log.info("aligned segments=%d words=%d", len(result.get("segments", [])), nwords)

    # Free WhisperX GPU memory before downstream stages (embeddings need VRAM)
    try:
        del model
    except NameError:
        pass
    try:
        del model_a
    except NameError:
        pass
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    log.debug("freed WhisperX GPU memory")

    # Glossary display corrections (traceable; timings + source untouched)
    if GLOSSARY:
        from pipeline.glossary import apply_glossary
        import json as _json
        corrections = []
        for seg in result.get("segments", []):
            seg_words, seg_corr = apply_glossary(seg.get("words", []) or [], GLOSSARY)
            seg["words"] = seg_words
            corrections.extend(seg_corr)
        try:
            with open(os.path.join(output_dir, "glossary.json"), "w", encoding="utf-8") as f:
                _json.dump(corrections, f, indent=2, ensure_ascii=False)
        except OSError:
            log.exception("glossary.json write failed")
        log.info("glossary corrections=%d", len(corrections))

    # Save transcript to JSON
    transcript_path = os.path.join(output_dir, "transcript.json")
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Also save a readable .txt version
    txt_path = os.path.join(output_dir, "transcript.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for segment in result["segments"]:
            start = format_time(segment["start"])
            end = format_time(segment["end"])
            text = segment["text"].strip()
            f.write(f"[{start} --> {end}] {text}\n")

    log.info("transcript saved json=%s txt=%s segments=%d elapsed=%.1fs",
             transcript_path, txt_path, len(result.get("segments", [])), time.monotonic() - t0)
    return transcript_path, result

def format_time(seconds):
    """Converts seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"