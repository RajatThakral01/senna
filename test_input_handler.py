import unittest
from unittest.mock import patch, MagicMock
import os

# Add the parent directory to the path so we can import input_handler
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from input import input_handler

class TestInputHandler(unittest.TestCase):

    def test_detect_source_type(self):
        print("Testing detect_source_type...")
        self.assertEqual(input_handler.detect_source_type("https://www.youtube.com/watch?v=dQw4w9WgXcQ"), "youtube")
        self.assertEqual(input_handler.detect_source_type("https://youtu.be/dQw4w9WgXcQ"), "youtube")
        self.assertEqual(input_handler.detect_source_type("https://www.youtube.com/live/12345"), "youtube_live")
        self.assertEqual(input_handler.detect_source_type("https://drive.google.com/file/d/12345/view"), "google_drive")
        self.assertEqual(input_handler.detect_source_type("/path/to/my/video.mp4"), "local_file")
        self.assertEqual(input_handler.detect_source_type("./local_video.mov"), "local_file")
        self.assertEqual(input_handler.detect_source_type("http://example.com/video.webm"), "direct_url")
        self.assertEqual(input_handler.detect_source_type("invalid_input"), "unknown")
        print("✅ detect_source_type tests passed!")

    @patch('input.input_handler._handle_youtube')
    def test_handle_input_youtube(self, mock_download):
        print("Testing handle_input with YouTube URL...")
        input_handler.handle_input("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        mock_download.assert_called_once_with("https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                                              os.path.join("downloads", "dQw4w9WgXcQ.mp4"))
        print("✅ YouTube handler called correctly.")

    @patch('input.input_handler._handle_youtube_live')
    def test_handle_input_youtube_live(self, mock_download):
        print("Testing handle_input with YouTube Live URL...")
        input_handler.handle_input("https://www.youtube.com/live/abcdefghijk")
        mock_download.assert_called_once_with("https://www.youtube.com/live/abcdefghijk",
                                              os.path.join("downloads", "abcdefghijk_live.mp4"))
        print("✅ YouTube Live handler called correctly.")

    @patch('input.input_handler.download_from_drive')
    def test_handle_input_google_drive(self, mock_download):
        print("Testing handle_input with Google Drive URL...")
        input_handler.handle_input("https://drive.google.com/file/d/123/view")
        mock_download.assert_called_once_with("https://drive.google.com/file/d/123/view",
                                              os.path.join("downloads", "view.mp4"))
        print("✅ Google Drive handler called correctly.")

    def test_handle_input_local_file_used_in_place(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            self.assertEqual(input_handler.handle_input(f.name), f.name)
        with self.assertRaises(FileNotFoundError):
            input_handler.handle_input("/path/to/missing_video.mp4")

    def test_relative_existing_path_is_local(self):
        import tempfile
        d = tempfile.mkdtemp(dir=".")
        p = os.path.join(os.path.basename(d), "clip.mp4")
        open(p, "wb").close()
        try:
            self.assertEqual(input_handler.detect_source_type(p), "local_file")
        finally:
            os.remove(p); os.rmdir(d)

    @patch('input.input_handler._handle_youtube')
    @patch('input.input_handler._cached_source_ok', return_value=True)
    def test_cached_download_reused(self, _ok, mock_download):
        out = input_handler.handle_input("https://youtu.be/dQw4w9WgXcQ")
        self.assertEqual(out, os.path.join("downloads", "dQw4w9WgXcQ.mp4"))
        mock_download.assert_not_called()

    @patch('input.input_handler.download_direct_url')
    def test_handle_input_direct_url(self, mock_download):
        print("Testing handle_input with direct URL...")
        input_handler.handle_input("http://example.com/video.mp4")
        args = mock_download.call_args[0]
        self.assertEqual(args[0], "http://example.com/video.mp4")
        self.assertTrue(args[1].startswith(os.path.join("downloads", "video_")))
        print("✅ Direct URL handler called correctly.")

    def test_handle_input_unknown(self):
        print("Testing handle_input with unknown source...")
        with self.assertRaises(ValueError):
            input_handler.handle_input("this is not a valid source")
        print("✅ Unknown source correctly raises ValueError.")

    def test_js_runtime_args_shape(self):
        args = input_handler._js_runtime_args()
        self.assertIsInstance(args, list)
        if args:
            self.assertEqual(args[0], "--js-runtimes")
            self.assertIn(args[1], ("deno", "node"))

    def test_sd_source_refused(self):
        print("Testing SD refusal...")
        import subprocess
        from unittest.mock import call
        with patch("input.input_handler._run_yt_dlp") as mock_run, \
             patch("input.input_handler._probe_height", return_value=360), \
             patch("os.path.exists", return_value=True), \
             patch("os.remove"), \
             patch("os.path.getsize", return_value=1):
            with self.assertRaises(RuntimeError):
                input_handler._handle_youtube("https://www.youtube.com/watch?v=123")
            # both clients attempted
            self.assertEqual(mock_run.call_count, 2)
        print("✅ SD source fails loudly after both clients.")

    def test_hq_retried_after_failure(self):
        # audio URL expired after a long video download (HTTP 403):
        # the same HQ command is retried before dropping to the SD client
        import subprocess
        err = subprocess.CalledProcessError(1, ["yt-dlp"])
        with patch("input.input_handler._run_yt_dlp",
                   side_effect=[err, None]) as mock_run, \
             patch("input.input_handler._probe_height", return_value=1080), \
             patch("os.path.exists", return_value=True), \
             patch("os.remove"), \
             patch("os.path.getsize", return_value=1):
            input_handler._handle_youtube("https://www.youtube.com/watch?v=123")
            self.assertEqual(mock_run.call_count, 2)
            self.assertEqual(mock_run.call_args_list[0][0][0],
                             mock_run.call_args_list[1][0][0])

    def test_hd_source_accepted_first_try(self):
        print("Testing HD accept...")
        with patch("input.input_handler._run_yt_dlp") as mock_run, \
             patch("input.input_handler._probe_height", return_value=1080), \
             patch("os.path.exists", return_value=True), \
             patch("os.remove"), \
             patch("os.path.getsize", return_value=1):
            input_handler._handle_youtube("https://www.youtube.com/watch?v=123")
            self.assertEqual(mock_run.call_count, 1)
            cmd = mock_run.call_args[0][0]
            flat = " ".join(cmd)
            self.assertIn("--js-runtimes", flat)
            self.assertIn("height>=720", flat)
        print("✅ HD source accepted, JS runtime passed.")


if __name__ == '__main__':
    unittest.main()
