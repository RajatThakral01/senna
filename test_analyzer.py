# test_analyzer.py
import pytest
import os
import sys
import json
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(__file__))
from pipeline.analyzer import analyze_transcript, build_transcript_text, format_time


class TestAnalyzer:
    def test_format_time(self):
        assert format_time(0) == "00:00:00"
        assert format_time(125) == "00:02:05"
        assert format_time(3695) == "01:01:35"

    def test_build_transcript_text(self):
        mock_result = {
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "Hello world"},
                {"start": 5.0, "end": 10.0, "text": "This is a test"},
                {"start": 10.0, "end": 15.0, "text": "Testing one two three"}
            ]
        }
        text = build_transcript_text(mock_result)

        assert "[00:00:00 --> 00:00:05] Hello world" in text
        assert "[00:00:05 --> 00:00:10] This is a test" in text
        assert "[00:00:10 --> 00:00:15] Testing one two three" in text

    def test_build_transcript_text_empty(self):
        mock_result = {"segments": []}
        text = build_transcript_text(mock_result)
        assert text == ""

    @patch("analyzer.requests.post")
    @patch("analyzer.json.loads")
    def test_analyze_transcript_success(self, mock_json_loads, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"clip_number": 1}'}}]
        }
        mock_post.return_value = mock_response
        mock_json_loads.return_value = [
            {
                "clip_number": 1,
                "start_time": "00:01:00",
                "end_time": "00:01:30",
                "duration_seconds": 30,
                "hook": "Test hook",
                "reason": "Test reason",
                "suggested_title": "Test Title",
                "suggested_hashtags": "#test #viral"
            }
        ]

        mock_transcript = {
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "Hello"}
            ]
        }

        with patch("analyzer.build_transcript_text", return_value="mock transcript"):
            with patch("builtins.open", MagicMock()):
                clips = analyze_transcript(mock_transcript)

        assert len(clips) == 1
        assert clips[0]["clip_number"] == 1
        assert clips[0]["start_time"] == "00:01:00"

    @patch("analyzer.requests.post")
    def test_analyze_transcript_api_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_post.return_value = mock_response

        mock_transcript = {"segments": []}

        with pytest.raises(Exception) as exc_info:
            analyze_transcript(mock_transcript)

        assert "401" in str(exc_info.value)

    @patch("analyzer.requests.post")
    @patch("analyzer.json.loads")
    def test_analyze_transcript_cleans_markdown_fences(self, mock_json_loads, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "```json\n{\"clip_number\": 1}\n```"}}]
        }
        mock_post.return_value = mock_response
        mock_json_loads.return_value = [
            {"clip_number": 1, "start_time": "00:00:00", "end_time": "00:00:30",
             "hook": "Test", "reason": "Test", "suggested_title": "Test", "suggested_hashtags": "#test"}
        ]

        mock_transcript = {"segments": []}

        with patch("analyzer.build_transcript_text", return_value=""):
            with patch("builtins.open", MagicMock()):
                analyze_transcript(mock_transcript)

        mock_json_loads.assert_called_once()
        call_arg = mock_json_loads.call_args[0][0]
        assert "clip_number" in call_arg
        assert "```" not in call_arg