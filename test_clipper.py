# test_clipper.py
import pytest
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(__file__))
from pipeline.clipper import cut_clips, time_to_seconds


class TestClipper:
    def test_time_to_seconds(self):
        assert time_to_seconds("00:00:00") == 0
        assert time_to_seconds("00:01:00") == 60
        assert time_to_seconds("01:00:00") == 3600
        assert time_to_seconds("01:30:30") == 5430
        assert time_to_seconds("00:05:15") == 315

    @patch("clipper.subprocess.run")
    def test_cut_clips_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        clips = [
            {
                "clip_number": 1,
                "start_time": "00:01:00",
                "end_time": "00:01:30",
                "hook": "Test hook",
                "suggested_title": "Test",
                "suggested_hashtags": "#test",
                "reason": "Test reason"
            }
        ]

        with patch("clipper.os.path.join", side_effect=lambda *args: "/".join(args)):
            result = cut_clips("video.mp4", clips, "clips")

        assert len(result) == 1
        assert result[0]["clip_number"] == 1
        assert "clip_1.mp4" in result[0]["path"]
        mock_run.assert_called_once()

    @patch("clipper.subprocess.run")
    def test_cut_clips_ffmpeg_error(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="Error")

        clips = [
            {
                "clip_number": 1,
                "start_time": "00:01:00",
                "end_time": "00:01:30",
                "hook": "Test",
                "suggested_title": "Test",
                "suggested_hashtags": "#test",
                "reason": "Test"
            }
        ]

        with patch("clipper.os.path.join", side_effect=lambda *args: "/".join(args)):
            result = cut_clips("video.mp4", clips, "clips")

        assert len(result) == 0

    @patch("clipper.subprocess.run")
    def test_cut_clips_multiple(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        clips = [
            {"clip_number": 1, "start_time": "00:00:00", "end_time": "00:00:30", "hook": "", "suggested_title": "", "suggested_hashtags": "", "reason": ""},
            {"clip_number": 2, "start_time": "00:01:00", "end_time": "00:01:30", "hook": "", "suggested_title": "", "suggested_hashtags": "", "reason": ""},
            {"clip_number": 3, "start_time": "00:02:00", "end_time": "00:02:30", "hook": "", "suggested_title": "", "suggested_hashtags": "", "reason": ""},
        ]

        with patch("clipper.os.path.join", side_effect=lambda *args: "/".join(args)):
            result = cut_clips("video.mp4", clips, "clips")

        assert len(result) == 3
        assert mock_run.call_count == 3