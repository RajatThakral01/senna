# transcriber.py
import time
import whisperx
import json
import os
from logger import get_logger, log_stage
from config import WHISPER_MODEL, WHISPER_LANGUAGE, WHISPER_DEVICE

log = get_logger("pipeline.transcriber")

def transcribe_audio(audio_path, output_dir="transcripts"):
    """
    Transcribes audio using WhisperX with word-level timestamps.
    Returns path to transcript JSON file.
    """
    log.info("transcribe start audio=%s model=%s lang=%s device=%s", audio_path, WHISPER_MODEL, WHISPER_LANGUAGE, WHISPER_DEVICE)
    if not os.path.exists(audio_path):
        log.error("audio file missing path=%s", audio_path)
        raise FileNotFoundError(audio_path)
    t0 = time.monotonic()
    with log_stage("pipeline.transcriber", "load_model"):
        log.info("loading WhisperX model=%s device=%s", WHISPER_MODEL, WHISPER_DEVICE)
        # Load WhisperX model
        model = whisperx.load_model(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type="float32"   # Use "float16" if you have a GPU
        )

    with log_stage("pipeline.transcriber", "transcribe"):
        log.info("transcribing audio (may take minutes) path=%s", audio_path)

        # Load audio using librosa (bypasses torchcodec issues)
        import librosa
        import numpy as np
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        audio = audio.astype(np.float32)
        log.debug("audio loaded samples=%d", len(audio))
        result = model.transcribe(audio, language=WHISPER_LANGUAGE, batch_size=16)
    log.info("transcribe raw segments=%d lang=%s", len(result.get("segments", [])), result.get("language"))

    # Align for word-level timestamps
    with log_stage("pipeline.transcriber", "align"):
        log.info("aligning word-level timestamps...")
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"],
            device=WHISPER_DEVICE
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            WHISPER_DEVICE,
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