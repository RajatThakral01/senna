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
        # waiting for rate-limit resets disabled: fail straight away
        with _keys(GROQ_API_KEY="k1"), \
                patch.dict("os.environ", {"LLM_RATE_LIMIT_MAX_WAIT": "0"}):
            with patch("pipeline.llm_client.requests.post") as mp:
                mp.return_value = _resp(429)
                with pytest.raises(requests.exceptions.HTTPError):
                    post_chat({"model": "m"}, timeout=10, slot=2)
                assert mp.call_count == 1  # single-key pool, no point looping

    def test_no_keys_raises_connection_error(self):
        with _keys():
            with pytest.raises(requests.exceptions.ConnectionError):
                post_chat({"model": "m"}, timeout=10, slot=1)


class TestReasoningHeadroomAndJsonMode:
    def _resp(self, status=200, body=None):
        from unittest.mock import MagicMock
        r = MagicMock()
        r.status_code = status
        r.json.return_value = body or {"choices": [{"finish_reason": "stop",
                                                     "message": {"content": "{}"}}]}
        r.raise_for_status.return_value = None
        return r

    def test_headroom_added_for_reasoning_model(self):
        from pipeline.llm_client import _prepare_payload
        p = _prepare_payload({"model": "openai/gpt-oss-120b", "max_tokens": 400}, False)
        assert p["max_tokens"] == 4400 and "response_format" not in p
        q = _prepare_payload({"model": "llama-3.3-70b", "max_tokens": 400}, True)
        assert q["max_tokens"] == 400
        assert q["response_format"] == {"type": "json_object"}

    def test_json_mode_rejection_retries_without_json_mode(self, monkeypatch):
        from unittest.mock import patch
        import pipeline.llm_client as L
        monkeypatch.setattr(L, "_ordered_keys", lambda slot: [("key1", "k")])
        bad = self._resp(400, {"error": {"code": "json_validate_failed",
                                         "message": "Failed to generate JSON"}})
        good = self._resp()
        with patch.object(L.requests, "post", side_effect=[bad, good]) as post:
            r = L.post_chat({"model": "openai/gpt-oss-120b", "max_tokens": 10},
                            timeout=5, slot=1, json_mode=True)
        assert r is good and post.call_count == 2
        assert "response_format" in post.call_args_list[0].kwargs["json"]
        assert "response_format" not in post.call_args_list[1].kwargs["json"]


class TestRateLimitWait:
    def _r(self, status, msg="", headers=None):
        from unittest.mock import MagicMock
        r = MagicMock()
        r.status_code = status
        r.headers = headers or {}
        r.json.return_value = ({"error": {"message": msg}} if status == 429 else
                               {"choices": [{"finish_reason": "stop",
                                             "message": {"content": "{}"}}]})
        r.raise_for_status.return_value = None
        return r

    def test_parse_wait(self):
        from pipeline.llm_client import _rate_wait, _parse_duration
        assert _parse_duration("1m2.5s") == 62.5 and _parse_duration("450ms") == 0.45
        w, daily = _rate_wait(self._r(429, "Rate limit reached on tokens per minute "
                                           "(TPM). Please try again in 4.5s."))
        assert w == 5.0 and daily is False
        w, daily = _rate_wait(self._r(429, "", {"retry-after": "12"}))
        assert w == 12.5
        assert _rate_wait(self._r(429, "Limit on tokens per day (TPD)"))[1] is True

    def test_waits_then_succeeds_when_all_keys_limited(self, monkeypatch):
        from unittest.mock import patch
        import pipeline.llm_client as L
        monkeypatch.setattr(L, "_ordered_keys", lambda s: [("key1", "a"), ("key2", "b")])
        slept = []
        monkeypatch.setattr(L.time, "sleep", lambda s: slept.append(s))
        lim = self._r(429, "try again in 3s")
        ok = self._r(200)
        with patch.object(L.requests, "post", side_effect=[lim, lim, ok]):
            r = L.post_chat({"model": "m", "max_tokens": 5}, timeout=5, slot=1)
        assert r is ok and slept == [3.5]

    def test_daily_limit_fails_fast(self, monkeypatch):
        import pytest
        from unittest.mock import patch
        import pipeline.llm_client as L
        monkeypatch.setattr(L, "_ordered_keys", lambda s: [("key1", "a")])
        monkeypatch.setattr(L.time, "sleep", lambda s: (_ for _ in ()).throw(AssertionError("slept")))
        with patch.object(L.requests, "post",
                          return_value=self._r(429, "Limit reached on tokens per day (TPD)")):
            with pytest.raises(L.requests.exceptions.HTTPError):
                L.post_chat({"model": "m", "max_tokens": 5}, timeout=5, slot=1)

    def test_wait_budget_respected(self, monkeypatch):
        import pytest
        from unittest.mock import patch
        import pipeline.llm_client as L
        monkeypatch.setattr(L, "_ordered_keys", lambda s: [("key1", "a")])
        monkeypatch.setenv("LLM_RATE_LIMIT_MAX_WAIT", "10")
        slept = []
        monkeypatch.setattr(L.time, "sleep", lambda s: slept.append(s))
        with patch.object(L.requests, "post", return_value=self._r(429, "try again in 4s")):
            with pytest.raises(L.requests.exceptions.HTTPError):
                L.post_chat({"model": "m", "max_tokens": 5}, timeout=5, slot=1)
        assert sum(slept) <= 10
