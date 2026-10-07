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
