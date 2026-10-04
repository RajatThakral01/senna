# test_transcriber.py
import pytest
import os
import sys
import json
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

    @patch("transcriber.whisperx.load_model")
    @patch("transcriber.whisperx.load_audio")
    @patch("transcriber.whisperx.load_align_model")
    @patch("transcriber.whisperx.align")
    def test_transcribe_audio_success(self, mock_align, mock_load_align, mock_load_audio, mock_load_model):
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        mock_audio = MagicMock()
        mock_load_audio.return_value = mock_audio

        mock_align_model = MagicMock()
        mock_metadata = MagicMock()
        mock_load_align.return_value = (mock_align_model, mock_metadata)

        mock_result = {
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "Hello world"},
                {"start": 5.0, "end": 10.0, "text": "This is a test"}
            ]
        }
        mock_model.transcribe.return_value = mock_result
        mock_align.return_value = mock_result

        with patch("transcriber.os.path.join", side_effect=lambda *args: "/".join(args)):
            with patch("builtins.open", MagicMock()):
                with patch("json.dump"):
                    path, result = transcribe_audio("audio.mp3", "transcripts")

        assert path == "transcripts/transcript.json"
        assert result["language"] == "en"
        assert len(result["segments"]) == 2