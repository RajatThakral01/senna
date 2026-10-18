# test_transcriber.py
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(__file__))
from pipeline.transcriber import transcribe_audio, format_time


class TestTranscriber:
    def test_format_time(self):
        assert format_time(0) == "00:00:00"
        assert format_time(3661) == "01:01:01"
        assert format_time(7200) == "02:00:00"
        assert format_time(45) == "00:00:45"
        assert format_time(3700) == "01:01:40"

    def test_missing_audio_raises(self):
        import pytest
        with pytest.raises(FileNotFoundError):
            transcribe_audio("no-such-file.wav", "transcripts")

    @patch("pipeline.transcriber.whisperx.load_model")
    @patch("pipeline.transcriber.whisperx.load_align_model")
    @patch("pipeline.transcriber.whisperx.align")
    def test_transcribe_audio_success(self, mock_align, mock_load_align, mock_load_model):
        import numpy as np
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        mock_result = {
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "Hello world"},
                {"start": 5.0, "end": 10.0, "text": "This is a test"}
            ]
        }
        mock_model.transcribe.return_value = mock_result
        mock_load_align.return_value = (MagicMock(), MagicMock())
        mock_align.return_value = mock_result

        fake_audio = np.zeros(16000, dtype="float32")
        with patch("pipeline.transcriber.os.path.exists", return_value=True):
            with patch("librosa.load", return_value=(fake_audio, 16000)):
                with patch("pipeline.transcriber.os.path.join",
                           side_effect=lambda *a: "/".join(a)):
                    with patch("builtins.open", MagicMock()):
                        with patch("json.dump"):
                            path, result = transcribe_audio("audio.mp3", "transcripts")

        assert path == "transcripts/transcript.json"
        assert result["language"] == "en"
        assert len(result["segments"]) == 2


def test_subprocess_policy(monkeypatch):
    from pipeline import transcriber as T
    monkeypatch.delenv("TRANSCRIBE_SUBPROCESS", raising=False)
    assert T._use_subprocess(keep_models=True) is False          # batch cache stays
    assert T._use_subprocess(keep_models=False) is False         # under pytest: in-process
    monkeypatch.setenv("TRANSCRIBE_SUBPROCESS", "1")
    assert T._use_subprocess(keep_models=False) is True
    assert T._use_subprocess(keep_models=True) is False
    monkeypatch.setenv("TRANSCRIBE_SUBPROCESS", "0")
    assert T._use_subprocess(keep_models=False) is False


def test_subprocess_missing_file_raises_before_spawning(monkeypatch):
    from pipeline import transcriber as T
    monkeypatch.setenv("TRANSCRIBE_SUBPROCESS", "1")
    import pytest
    with pytest.raises(FileNotFoundError):
        T.transcribe_audio("no-such-file.wav", "transcripts")


def test_subprocess_reads_child_transcript(monkeypatch, tmp_path):
    import json
    import subprocess
    from pipeline import transcriber as T
    monkeypatch.setenv("TRANSCRIBE_SUBPROCESS", "1")
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    seen = {}

    class FakePopen:
        rc = 0

        def __init__(self, cmd, cwd=None):
            seen["cmd"] = cmd
            seen["waits"] = 0
            (tmp_path / "transcript.json").write_text(
                json.dumps({"segments": [{"text": "hi"}]}))

        def wait(self, timeout=None):
            seen["waits"] += 1
            if seen["waits"] == 1:                     # one heartbeat, then finish
                raise subprocess.TimeoutExpired("cmd", timeout)
            return FakePopen.rc
    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    path, res = T.transcribe_audio(str(wav), str(tmp_path))
    assert res == {"segments": [{"text": "hi"}]} and path.endswith("transcript.json")
    assert "_transcribe_here" in seen["cmd"][2] and seen["cmd"][3] == str(wav)
    assert seen["waits"] == 2

    FakePopen.rc = -9
    import pytest
    with pytest.raises(RuntimeError, match="out of memory"):
        T.transcribe_audio(str(wav), str(tmp_path))


def test_embedder_release_model(monkeypatch):
    from pipeline import embedder
    monkeypatch.setattr(embedder, "_MODEL", object())
    monkeypatch.setattr(embedder, "_MODEL_KEY", "k")
    embedder.release_model()
    assert embedder._MODEL is None and embedder._MODEL_KEY is None
    embedder.release_model()                                     # idempotent
