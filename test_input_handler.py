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
        input_handler.handle_input("https://www.youtube.com/watch?v=123")
        mock_download.assert_called_once_with("https://www.youtube.com/watch?v=123")
        print("✅ YouTube handler called correctly.")

    @patch('input.input_handler._handle_youtube_live')
    def test_handle_input_youtube_live(self, mock_download):
        print("Testing handle_input with YouTube Live URL...")
        input_handler.handle_input("https://www.youtube.com/live/123")
        mock_download.assert_called_once_with("https://www.youtube.com/live/123")
        print("✅ YouTube Live handler called correctly.")

    @patch('input.input_handler.download_from_drive')
    def test_handle_input_google_drive(self, mock_download):
        print("Testing handle_input with Google Drive URL...")
        input_handler.handle_input("https://drive.google.com/file/d/123/view")
        mock_download.assert_called_once_with("https://drive.google.com/file/d/123/view", "input/raw_video.mp4")
        print("✅ Google Drive handler called correctly.")

    @patch('input.input_handler.copy_local_file')
    def test_handle_input_local_file(self, mock_copy):
        print("Testing handle_input with local file path...")
        input_handler.handle_input("/path/to/video.mp4")
        mock_copy.assert_called_once_with("/path/to/video.mp4", "input/raw_video.mp4")
        print("✅ Local file handler called correctly.")

    @patch('input.input_handler.download_direct_url')
    def test_handle_input_direct_url(self, mock_download):
        print("Testing handle_input with direct URL...")
        input_handler.handle_input("http://example.com/video.mp4")
        mock_download.assert_called_once_with("http://example.com/video.mp4", "input/raw_video.mp4")
        print("✅ Direct URL handler called correctly.")

    def test_handle_input_unknown(self):
        print("Testing handle_input with unknown source...")
        with self.assertRaises(ValueError):
            input_handler.handle_input("this is not a valid source")
        print("✅ Unknown source correctly raises ValueError.")

if __name__ == '__main__':
    unittest.main()
