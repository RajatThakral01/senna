# test_analyzer.py — tests for the current pipeline/analyzer.py API.
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(__file__))
from pipeline.analyzer import (
    format_time, to_sec, extract_json_from_response,
    _extract_clips_from_chunk, _deduplicate_clips,
)


def _llm_response(content):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": content}}]
    }
    mock_response.raise_for_status.return_value = None
    return mock_response


class TestAnalyzerHelpers:
    def test_format_time(self):
        assert format_time(0) == "00:00:00"
        assert format_time(125) == "00:02:05"
        assert format_time(3695) == "01:01:35"

    def test_to_sec(self):
        assert to_sec("00:01:00") == 60
        assert to_sec("01:30.5") == 90.5

    def test_extract_json_plain_array(self):
        raw = '[{"start_time": "00:01:00", "end_time": "00:01:30"}]'
        assert extract_json_from_response(raw) == raw

    def test_extract_json_markdown_fences(self):
        raw = '```json\n[{"start_time": "00:01:00"}]\n```'
        out = extract_json_from_response(raw)
        assert "```" not in out
        assert "start_time" in out

    def test_extract_json_think_block(self):
        raw = '<think>reasoning here</think>\n[{"start_time": "00:01:00"}]'
        out = extract_json_from_response(raw)
        assert "<think>" not in out
        assert "start_time" in out


class TestExtractClips:
    def _chunk(self):
        return {"chunk_index": 0, "start_time": 0.0, "end_time": 120.0,
                "text": "Hello world. This is a test of the clip extractor."}

    @patch("pipeline.analyzer.requests.post")
    def test_success(self, mock_post):
        mock_post.return_value = _llm_response(
            '[{"start_time": "00:01:00", "end_time": "00:01:30", '
            '"hook": "Test hook", "reason": "Test reason", '
            '"suggested_title": "Test Title", "suggested_hashtags": "#test #viral"}]')
        clips = _extract_clips_from_chunk(self._chunk(), {})
        assert len(clips) == 1
        assert clips[0]["hook"] == "Test hook"
        assert clips[0]["duration_seconds"] == 30

    @patch("pipeline.analyzer.requests.post")
    def test_short_clip_dropped(self, mock_post):
        mock_post.return_value = _llm_response(
            '[{"start_time": "00:01:00", "end_time": "00:01:05", "hook": "x"}]')
        assert _extract_clips_from_chunk(self._chunk(), {}) == []

    @patch("pipeline.analyzer.requests.post")
    def test_api_error_returns_empty(self, mock_post):
        import requests as _rq
        mock_post.side_effect = _rq.exceptions.ConnectionError("down")
        assert _extract_clips_from_chunk(self._chunk(), {}) == []

    @patch("pipeline.analyzer.requests.post")
    def test_empty_response_returns_empty(self, mock_post):
        mock_post.return_value = _llm_response('')
        assert _extract_clips_from_chunk(self._chunk(), {}) == []


class TestDeduplicate:
    def test_keeps_longer_on_near_start(self):
        clips = [
            {"start_time": "00:01:00", "end_time": "00:01:30", "duration_seconds": 30},
            {"start_time": "00:01:02", "end_time": "00:02:00", "duration_seconds": 58},
        ]
        out = _deduplicate_clips(clips, overlap_threshold_seconds=5.0)
        assert len(out) == 1
        assert out[0]["duration_seconds"] == 58

    def test_keeps_distinct(self):
        clips = [
            {"start_time": "00:01:00", "end_time": "00:01:30", "duration_seconds": 30},
            {"start_time": "00:05:00", "end_time": "00:05:30", "duration_seconds": 30},
        ]
        assert len(_deduplicate_clips(clips, overlap_threshold_seconds=5.0)) == 2

    def test_empty(self):
        assert _deduplicate_clips([], overlap_threshold_seconds=5.0) == []
