# test_subtitler.py
import pytest
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(__file__))
from subtitler import (
    time_to_seconds, seconds_to_srt_time, get_clip_transcript_segments,
    generate_srt_with_m27, save_srt, burn_subtitles, process_clip_subtitles
)


class TestSubtitler:
    def test_time_to_seconds(self):
        assert time_to_seconds("00:00:00") == 0
        assert time_to_seconds("00:05:30") == 330
        assert time_to_seconds("01:00:00") == 3600

    def test_seconds_to_srt_time(self):
        assert seconds_to_srt_time(0) == "00:00:00,000"
        assert seconds_to_srt_time(2.5) == "00:00:02,500"
        assert seconds_to_srt_time(65.3) == "00:01:05,299"
        assert seconds_to_srt_time(3661.999) == "01:01:01,998"

    def test_get_clip_transcript_segments(self):
        transcript_result = {
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "Hello"},
                {"start": 5.0, "end": 10.0, "text": "World"},
                {"start": 10.0, "end": 15.0, "text": "Test"},
                {"start": 20.0, "end": 25.0, "text": "Outside"}
            ]
        }

        segments = get_clip_transcript_segments(transcript_result, "00:00:05", "00:00:15")

        assert len(segments) == 3
        assert segments[0]["text"] == "Hello"
        assert segments[1]["text"] == "World"
        assert segments[2]["text"] == "Test"

    def test_get_clip_transcript_segments_edge_cases(self):
        transcript_result = {
            "segments": [
                {"start": 5.0, "end": 10.0, "text": "Middle"},
                {"start": 15.0, "end": 20.0, "text": "After"}
            ]
        }

        segments = get_clip_transcript_segments(transcript_result, "00:00:00", "00:00:30")
        assert len(segments) == 2

        segments = get_clip_transcript_segments(transcript_result, "00:00:00", "00:00:05")
        assert len(segments) == 1

    @patch("subtitler.requests.post")
    def test_generate_srt_with_m27_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "1\n00:00:00,000 --> 00:00:02,500\nHello world"}}]
        }
        mock_post.return_value = mock_response

        segments = [{"start": 0.0, "end": 2.5, "text": "Hello world"}]
        result = generate_srt_with_m27(segments, 1)

        assert "00:00:00,000" in result
        assert "Hello world" in result

    @patch("subtitler.requests.post")
    def test_generate_srt_cleans_code_fences(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "```srt\n1\n00:00:00,000 --> 00:00:02,500\nTest\n```"}}]
        }
        mock_post.return_value = mock_response

        segments = [{"start": 0.0, "end": 2.5, "text": "Test"}]
        result = generate_srt_with_m27(segments, 1)

        assert "```" not in result
        assert "Test" in result

    @patch("subtitler.requests.post")
    def test_generate_srt_api_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        segments = [{"start": 0.0, "end": 2.5, "text": "Test"}]

        with pytest.raises(Exception) as exc_info:
            generate_srt_with_m27(segments, 1)

        assert "500" in str(exc_info.value)

    def test_save_srt(self):
        srt_content = "1\n00:00:00,000 --> 00:00:02,500\nTest"
        with patch("subtitler.os.path.join", return_value="clips/clip_1.srt"):
            with patch("builtins.open", MagicMock()):
                path = save_srt(srt_content, 1, "clips")

        assert "clip_1.srt" in path

    @patch("subtitler.subprocess.run")
    def test_burn_subtitles_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        with patch("subtitler.os.path.join", side_effect=lambda *args: "/".join(args)):
            result = burn_subtitles("clip.mp4", "clip.srt", 1, "output")

        assert result is not None
        assert "clip_1_final.mp4" in result

    @patch("subtitler.subprocess.run")
    def test_burn_subtitles_ffmpeg_error(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="Error")

        with patch("subtitler.os.path.join", side_effect=lambda *args: "/".join(args)):
            result = burn_subtitles("clip.mp4", "clip.srt", 1, "output")

        assert result is None

    @patch("subtitler.generate_srt_with_m27")
    @patch("subtitler.save_srt")
    @patch("subtitler.burn_subtitles")
    def test_process_clip_subtitles_success(self, mock_burn, mock_save, mock_generate):
        mock_generate.return_value = "1\n00:00:00,000 --> 00:00:02,500\nTest"
        mock_save.return_value = "clips/clip_1.srt"
        mock_burn.return_value = "output/clip_1_final.mp4"

        clip_info = {
            "clip_number": 1,
            "path": "clips/clip_1.mp4",
            "start_time": "00:00:00",
            "end_time": "00:00:30"
        }
        transcript_result = {
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "Hello"}
            ]
        }

        result = process_clip_subtitles(clip_info, transcript_result, "clips", "output")

        assert result == "output/clip_1_final.mp4"
        mock_generate.assert_called_once()
        mock_save.assert_called_once()
        mock_burn.assert_called_once()

    @patch("subtitler.get_clip_transcript_segments")
    def test_process_clip_subtitles_no_segments(self, mock_get_segments):
        mock_get_segments.return_value = []

        clip_info = {
            "clip_number": 1,
            "path": "clips/clip_1.mp4",
            "start_time": "00:00:00",
            "end_time": "00:00:30"
        }
        transcript_result = {"segments": []}

        result = process_clip_subtitles(clip_info, transcript_result)

        assert result is None