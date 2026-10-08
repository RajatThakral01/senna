# test_llm_client.py — slot routing, fallback rotation, pool guards.
import os
import sys
from unittest.mock import patch, MagicMock

import pytest
import requests

sys.path.insert(0, os.path.dirname(__file__))

from pipeline import llm_client
from pipeline.llm_client import (post_chat, key_pool, has_keys, pool_usable,
                                 SLOT_FOR_STAGE)


def _resp(status=200, content="hi"):
    m = MagicMock()
    m.status_code = status
    m.json.return_value = {"choices": [{"message": {"content": content}}]}
    m.raise_for_status.return_value = None
    return m


def _keys(**kw):
    base = {"GROQ_API_KEY": "", "GROQ_API_KEY_2": "",
            "GROQ_API_KEY_3": "", "GROQ_API_KEY_FALLBACK": "",
            "GROQ_API_KEY_4": ""}
    base.update(kw)
    return patch.dict(os.environ, base, clear=False)


class TestPool:
    def test_empty_pool(self):
        with _keys():
            assert key_pool() == []
            assert has_keys() is False
            assert pool_usable() is False

    def test_placeholders_excluded(self):
        with _keys(GROQ_API_KEY="your_groq_api_key_1_here",
                   GROQ_API_KEY_2="real-key-2"):
            pool = key_pool()
            assert [lb for lb, _ in pool] == ["key2"]
            assert pool_usable() is True

    def test_dupes_collapsed(self):
        with _keys(GROQ_API_KEY="same", GROQ_API_KEY_2="same",
                   GROQ_API_KEY_FALLBACK="same"):
            assert len(key_pool()) == 1

    def test_slot_map_covers_stages(self):
        assert set(SLOT_FOR_STAGE) == {"outline", "discovery", "enrich",
                                       "similarity", "campaign",
                                       "analyzer", "boundaries"}


class TestRotation:
    def test_primary_success_single_call(self):
        with _keys(GROQ_API_KEY="k1", GROQ_API_KEY_FALLBACK="k4"):
            with patch("pipeline.llm_client.requests.post") as mp:
                mp.return_value = _resp(200)
                r = post_chat({"model": "m"}, timeout=10, slot=1)
                assert r.status_code == 200
                assert mp.call_count == 1
                sent = mp.call_args[1]["headers"]["Authorization"]
                assert sent == "Bearer k1"

    def test_429_falls_to_key4(self):
        with _keys(GROQ_API_KEY="k1", GROQ_API_KEY_FALLBACK="k4"):
            with patch("pipeline.llm_client.requests.post") as mp:
                mp.side_effect = [_resp(429), _resp(200)]
                r = post_chat({"model": "m"}, timeout=10, slot=1)
                assert r.status_code == 200
                assert mp.call_count == 2
                labels = [c[1]["headers"]["Authorization"] for c in mp.call_args_list]
                assert labels == ["Bearer k1", "Bearer k4"]

    def test_dead_key_skipped(self):
        with _keys(GROQ_API_KEY="dead", GROQ_API_KEY_2="k2"):
            with patch("pipeline.llm_client.requests.post") as mp:
                mp.side_effect = [_resp(403), _resp(200)]
                r = post_chat({"model": "m"}, timeout=10, slot=1)
                assert r.status_code == 200
                labels = [c[1]["headers"]["Authorization"] for c in mp.call_args_list]
                assert labels == ["Bearer dead", "Bearer k2"]

    def test_all_fail_raises_last(self):
        with _keys(GROQ_API_KEY="k1"):
            with patch("pipeline.llm_client.requests.post") as mp:
                mp.return_value = _resp(429)
                with pytest.raises(requests.exceptions.HTTPError):
                    post_chat({"model": "m"}, timeout=10, slot=2)
                assert mp.call_count == 1  # single-key pool, no point looping

    def test_no_keys_raises_connection_error(self):
        with _keys():
            with pytest.raises(requests.exceptions.ConnectionError):
                post_chat({"model": "m"}, timeout=10, slot=1)
