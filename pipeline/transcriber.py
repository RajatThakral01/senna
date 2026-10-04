# transcriber.py
import whisperx
import json
import os
from config import WHISPER_MODEL, WHISPER_LANGUAGE, WHISPER_DEVICE

def transcribe_audio(audio_path, output_dir="transcripts"):
    """
    Transcribes audio using WhisperX with word-level timestamps.
    Returns path to transcript JSON file.
    """

    print(f"🎙️  Loading WhisperX model ({WHISPER_MODEL})...")

    # Load WhisperX model
    model = whisperx.load_model(
        WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type="float32"   # Use "float16" if you have a GPU
    )

    print("📝  Transcribing audio (this may take a few minutes)...")

    # Load audio using librosa (bypasses torchcodec issues)
    import librosa
    import numpy as np
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    audio = audio.astype(np.float32)
    result = model.transcribe(audio, language=WHISPER_LANGUAGE, batch_size=16)

    # Align for word-level timestamps
    print("🔍  Aligning word-level timestamps...")
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

    print(f"✅  Transcript saved to {transcript_path}")
    return transcript_path, result

def format_time(seconds):
    """Converts seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"