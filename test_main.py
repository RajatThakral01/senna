# test_main.py
import pytest
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(__file__))


class TestMain:
    def test_print_banner(self):
        from main import print_banner
        with patch("builtins.print") as mock_print:
            print_banner()
            assert mock_print.call_count > 0

    def test_print_summary(self):
        from main import print_summary
        with patch("builtins.print") as mock_print:
            print_summary([], "Test Video")
            assert mock_print.call_count > 0

    def test_print_summary_with_clips(self):
        from main import print_summary
        mock_clips = [
            {
                "clip_number": 1,
                "final_path": "output/clip_1_final.mp4",
                "suggested_title": "Test Title",
                "suggested_hashtags": "#test #viral",
                "reason": "Test reason"
            }
        ]
        with patch("builtins.print") as mock_print:
            print_summary(mock_clips, "Test Video")

        call_args = [str(c) for c in mock_print.call_args_list]
        assert any("Test Title" in c for c in call_args)
        assert any("#test #viral" in c for c in call_args)

    @patch("main.download_video")
    @patch("main.transcribe_audio")
    @patch("main.analyze_transcript")
    @patch("main.cut_clips")
    @patch("main.process_clip_subtitles")
    @patch("builtins.open", MagicMock())
    @patch("json.dump")
    def test_main_flow(self, mock_json_dump, mock_process, mock_cut, mock_analyze, mock_transcribe, mock_download):
        mock_download.return_value = ("video.mp4", "audio.mp3", "Test Video")
        mock_transcribe.return_value = ("transcript.json", {"segments": []})
        mock_analyze.return_value = [
            {"clip_number": 1, "start_time": "00:00:00", "end_time": "00:00:30",
             "hook": "Test", "suggested_title": "Test", "suggested_hashtags": "#test", "reason": "Test"}
        ]
        mock_cut.return_value = [
            {"clip_number": 1, "path": "clips/clip_1.mp4", "start_time": "00:00:00", "end_time": "00:00:30",
             "hook": "Test", "suggested_title": "Test", "suggested_hashtags": "#test", "reason": "Test"}
        ]
        mock_process.return_value = "output/clip_1_final.mp4"

        from main import main

        with patch("builtins.input", return_value="https://youtube.com/watch?v=test"):
            with patch("sys.argv", ["main.py"]):
                main()

        mock_download.assert_called_once()
        mock_transcribe.assert_called_once()
        mock_analyze.assert_called_once()
        mock_cut.assert_called_once()
        mock_process.assert_called_once()

    @patch("sys.argv", ["main.py"])
    @patch("builtins.input", return_value="")
    @patch("builtins.print")
    def test_main_no_url(self, mock_print, mock_input):
        from main import main
        main()
        assert any("No URL" in str(c) for c in mock_print.call_args_list)

    def test_main_with_url_arg(self):
        from main import main

        with patch("main.download_video") as mock_download:
            mock_download.side_effect = Exception("stop")
            with patch("sys.argv", ["main.py", "https://youtube.com/watch?v=test"]):
                try:
                    main()
                except Exception:
                    pass

        mock_download.assert_called_once_with("https://youtube.com/watch?v=test")