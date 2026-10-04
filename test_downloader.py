# test_downloader.py
import pytest
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(__file__))
from pipeline.downloader import download_video


class TestDownloader:
    @patch("downloader.yt_dlp.YoutubeDL")
    def test_download_video_success(self, mock_yt_dlp):
        mock_ydl_instance = MagicMock()
        mock_yt_dlp.return_value.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_yt_dlp.return_value.__exit__ = MagicMock(return_value=False)

        mock_ydl_instance.extract_info.return_value = {
            "title": "Test Video",
            "duration": 300
        }

        with patch("os.path.join", side_effect=lambda *args: "/".join(args)):
            video_path, audio_path, title = download_video("https://youtube.com/watch?v=test")

        assert title == "Test Video"
        assert "video.mp4" in video_path
        assert "audio.mp3" in audio_path

    @patch("downloader.yt_dlp.YoutubeDL")
    def test_download_video_no_title(self, mock_yt_dlp):
        mock_ydl_instance = MagicMock()
        mock_yt_dlp.return_value.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_yt_dlp.return_value.__exit__ = MagicMock(return_value=False)

        mock_ydl_instance.extract_info.return_value = {
            "title": "Unknown",
            "duration": 0
        }

        with patch("os.path.join", side_effect=lambda *args: "/".join(args)):
            video_path, audio_path, title = download_video("https://youtube.com/watch?v=test")

        assert title == "Unknown"